import logging
import os
from typing import Any, Dict, Optional, Tuple, Collection, List
from functools import partial
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import einops
from einops import rearrange, repeat
from torch import einsum
from einops_exts import rearrange_many
import wandb
from timm.layers.mlp import Mlp
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only


from beast.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from beast.callbacks.ema import EMA
from beast.models.utils import generate_policy_prompt

logger = logging.getLogger(__name__)


def create_bidirectional_mask(batch_size, seq_length, device):
    """
    Creates a bidirectional attention mask for Florence-2 decoder.
    
    In a bidirectional mask, every token can attend to every other token,
    allowing full visibility in both directions.
    
    Args:
        batch_size (int): Batch size
        seq_length (int): Sequence length (both target and source length for self-attention)
        device: Device to create tensor on
        
    Returns:
        torch.FloatTensor: Bidirectional mask with shape (batch_size, 1, seq_length, seq_length)
    """
    # For bidirectional attention, we want all positions to be visible
    # This means the mask should be all zeros (allowing attention everywhere)
    
    # Create a tensor with shape (batch_size, 1, seq_length, seq_length) filled with zeros
    # In attention masks, 0.0 means "attend to this position"
    bidirectional_mask = torch.zeros((batch_size, 1, seq_length, seq_length), device=device)
    
    return bidirectional_mask


class BEASTF(pl.LightningModule):
    def __init__(
        self,
        # VLM Configuration
        vlm_path: str = "microsoft/Florence-2-base",
        freeze_florence: bool = False,
        freeze_vision_tower: bool = False,
        vlm_prompt_style: str = "default",
        token_dropout: float = 0.2,
        
        # Model Structure
        multistep: int = 10,
        num_sampling_steps: int = 5,
        lowdim_obs_dim: int = 7,
        action_dim: int = 7,
        act_window_size: int = 10,
        
        # Model flags
        use_second_view: bool = False,
        second_view_key: str = 'image_wrist',
        use_proprio: bool = False,
        return_act_chunk: bool = False,
        
        # Optimizer Configuration
        optimizer_type: str = "adamw",
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,

        # MP tokenizer
        mp_tokenizer: DictConfig = None,
        update_w_bound: bool = False,
        pre_compute_w_bound: bool = False,
        pre_compute_w_bound_steps: int = 50000,

        load_pretrained: bool = False,
        pretrained_model_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        # self.automatic_optimization = False
        # Initialize model flags and configurations
        self._init_flags(
            use_second_view=use_second_view,
            vlm_prompt_style=vlm_prompt_style,
            token_dropout=token_dropout,
            use_proprio=use_proprio,
            return_act_chunk=return_act_chunk,
            second_view_key=second_view_key,
        )
        self.obs_modalities = []
        # Initialize model dimensions
        self._init_dimensions(
            lowdim_obs_dim=lowdim_obs_dim,
            action_dim=action_dim,
            act_window_size=act_window_size,
            multistep=multistep,
            num_sampling_steps=num_sampling_steps,
        )
        self.target_modality = "actions"
        # Setup VLM and core components
        self._setup_vlm(vlm_path, freeze_vision_tower, freeze_florence)

        hidden_dim = self.vlm.config.text_config.d_model
        self.vlm_latent_dim = hidden_dim
        self.use_proprio = use_proprio
        
        # Initialize state tracking
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self.modality_scope = "lang"
        # Save optimizer config
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.optimizer_type = optimizer_type

        self.action_tokenizer = hydra.utils.instantiate(mp_tokenizer)
        self.num_dof = self.action_tokenizer.num_dof
        self.num_basis = self.action_tokenizer.num_basis
        # self.tokenizer_vocab_size = self.tokenizer.vocab_size
        self.vlm_vocab_size = self.vlm.config.vocab_size - 1
        self.action_tokenizer.update_vlm_vocab_size(self.vlm_vocab_size)

        self.update_w_bound = update_w_bound
        self.precompute_w_bound = pre_compute_w_bound
        self.precompute_w_bound_steps = pre_compute_w_bound_steps

        if load_pretrained and pretrained_model_path is not None:
            self._load_pretrained_weights(pretrained_model_path)
    

    def _load_pretrained_weights(self, pretrained_model_path: str, mean_resizing: bool = False):
        """Loads pretrained weights, handling key mismatches (e.g., different prefixes)."""
        print(f"Loading pretrained weights from {pretrained_model_path}...")

        # Load checkpoint
        checkpoint = torch.load(pretrained_model_path, map_location=self.device)

        # Extract the state dict (handle PyTorch Lightning or plain models)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Fix key mismatches: remove 'agent.' prefix if it exists
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("agent.", "")  # Remove 'agent.' if it exists
            new_state_dict[new_key] = value

        # Load the weights, allowing partial matches
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)

        # Log mismatches for debugging
        print(f"Pretrained weights loaded with the following issues:")
        if missing_keys:
            print(f"  ⚠️ Missing keys (not found in checkpoint, using default init): {len(missing_keys)}")
            print(f"    {missing_keys[:30]} ...")  # Show first 30 for brevity
        if unexpected_keys:
            print(f"  ⚠️ Unexpected keys (ignored): {len(unexpected_keys)}")
            print(f"    {unexpected_keys[:30]} ...")  # Show first 30 for brevity
        if not missing_keys and not unexpected_keys:
            print("  ✅ All keys matched successfully!")

        # Handle mean-resizing for missing embeddings if enabled
        if mean_resizing:
            self._initialize_new_embeddings(new_state_dict)

        return missing_keys, unexpected_keys

    def _init_flags(self, **kwargs):
        """Initialize model flags and configurations"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if self.vlm_prompt_style not in ["default", "feature_focused", "state_oriented"]:
            raise ValueError("Invalid VLM prompt style")
        
        self.format_instruction = functools.partial(
                             generate_policy_prompt,
                             robot_name="Franka Panda",
                             action_space="Delta End-Effector",
                             num_arms="1",
                             prompt_style='minimal')
        
        self.use_proprio = self.use_proprio 
        self.use_second_view = self.use_second_view and self.second_view_key is not None
        self.vlm_prompt_style = self.vlm_prompt_style
        self.return_act_chunk = False

    def _init_dimensions(self, **kwargs):
        """Initialize model dimensions"""
        for key, value in kwargs.items():
            setattr(self, key, value)
            

    def _setup_vlm(self, vlm_path: str, freeze_vision_tower: bool, freeze_florence: bool):
        """Initialize and configure the Florence-2 VLM"""
        print(f"Loading Florence-2 from {vlm_path}")
        
        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_path, trust_remote_code=True)
        
        
        # Handle parameter freezing
        if freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif not freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True

        # Setup processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        
        # Create prompt embedding
        self.prompt_embeds = self._create_prompt_embed("<Primitives>")
        
        # Setup token dropout
        self.vlm_token_dropout = nn.Dropout(self.token_dropout)


    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Get parameter groups
        optim_groups = self._get_param_groups()

        # Initialize optimizer
        optimizer = torch.optim.AdamW(
                optim_groups,
                lr=self.optimizer_config.learning_rate,
                betas=self.optimizer_config.betas
            )

        # Initialize scheduler
        scheduler = TriStageLRScheduler(
            optimizer,
            OmegaConf.create(self.lr_scheduler_config)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def _get_param_groups(self):
        """Get parameter groups for optimizer"""
        no_decay = ['bias', 'LayerNorm', 'layernorm', 'ln', 'norm']
        decay_group = []
        no_decay_group = []

        # Collect all parameters, excluding VLM if frozen
        for name, param in self.named_parameters():
            if param.requires_grad:
                if any(nd in name.lower() for nd in no_decay):
                    no_decay_group.append(param)
                else:
                    decay_group.append(param)

        return [
            {"params": decay_group, "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": no_decay_group, "weight_decay": 0.0}
        ]

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        """Lightning training step"""

        # Get optimizer
        opt = self.optimizers()
        
        # Compute loss
        total_loss = torch.tensor(0.0, device=self.device)
        action_loss = torch.tensor(0.0, device=self.device) 
        training_loss = torch.tensor(0.0, device=self.device)
        total_bs = 0
        token_predict_accuracy = 0.0

        for modality_scope, dataset_batch in batch.items():
            self.modality_scope = modality_scope
            llm_output_dict = self.compute_llm_outputs(dataset_batch)
            act_loss = llm_output_dict["reconstruct_action_mse"]
            predict_accuracy = llm_output_dict["token_predict_accuarcy"]
            llm_loss = llm_output_dict["llm_loss"]


            action_loss = action_loss + act_loss
            total_bs = total_bs + len(dataset_batch["actions"])
            training_loss = training_loss + llm_loss
            token_predict_accuracy = token_predict_accuracy + predict_accuracy

        token_predict_accuracy = token_predict_accuracy / len(batch)
        action_loss = action_loss / len(batch)
        training_loss = training_loss / len(batch)

        # Log metrics
        self._log_training_metrics(llm_loss=training_loss, token_pred_acc=token_predict_accuracy, 
                                   reconstruct_mse=action_loss,total_bs=total_bs)

        return training_loss

    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Lightning validation step"""
        output = {}
        with torch.no_grad():
            llm_ouput_dict = self.compute_llm_outputs(batch)
            
            # Log metrics
            self._log_validation_metrics(llm_loss=llm_ouput_dict["llm_loss"],
                                         token_pred_acc=llm_ouput_dict["token_predict_accuarcy"],
                                         reconstruct_mse=llm_ouput_dict["reconstruct_action_mse"])
            
            output["validation_loss"] = llm_ouput_dict["llm_loss"] / len(batch)
            return output
            

    def _create_prompt_embed(self, prompt_text):
        """Create embeddings for prompt tokens"""
        # Add special token if not in vocabulary
        self.tokenizer.add_special_tokens({'additional_special_tokens': [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        
        # Get token ID and create embedding
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)), 
            requires_grad=False
        )
    
        return prompt_embed.unsqueeze(0).unsqueeze(0)

    def token_prediction_accuracy(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Computes token-level prediction accuracy for a batch.

        Args:
            logits (torch.Tensor): Model output logits of shape (batch_size, num_classes, seq_len, ...)
            targets (torch.Tensor): Ground-truth class indices of shape (batch_size, seq_len, ...)

        Returns:
            float: Accuracy as a percentage (0-100).
        """

        # Compute accuracy
        correct = (preds == targets).sum().item()
        total = targets.numel()  # Total number of tokens

        return 100.0 * correct / total if total > 0 else 0.0  # Return accuracy as percentage
    

    def compute_input_features(self, batch: Dict) -> torch.Tensor:
        device = self.device
        default_type = next(self.parameters()).dtype
        
        # Process primary image
        image_tensor = batch["rgb_obs"]['rgb_static']
        B, T, C, H, W = image_tensor.shape
        
        # Extract visual features
        image_features = self.vlm._encode_image(
            image_tensor.view(-1, C, H, W).to(device).to(default_type)
        ).to(default_type)
        image_features = image_features.view(B, T * image_features.shape[1], -1)
        
        # Process second view if enabled
        if self.use_second_view:
            image2_tensor = batch["rgb_obs"]['rgb_gripper']
            image2_features = self.vlm._encode_image(
                image2_tensor.view(-1, C, H, W).to(device).to(default_type)
            ).to(default_type)
            image2_features = image2_features.view(B, T * image2_features.shape[1], -1)
            image_features = torch.cat([image_features, image2_features], dim=1)
        
        # Get text embeddings
        # Get text embeddings once to reuse
        constructed_prompts = self.construct_prompts(batch)
        text_embeds = self._get_text_embeddings(constructed_prompts, device)
        
        # Add task prompt and aggregation tokens
        task_prompt = self.prompt_embeds.expand(B, -1, -1).to(image_features.device)
        
        # Merge sequence
        merged_embeds = torch.cat([
            image_features,
            task_prompt,
            text_embeds.to(image_features.device)
        ], dim=1)
        
        # Create attention mask
        attention_mask = torch.ones(merged_embeds.shape[:2], device=merged_embeds.device)

        # Process through encoder
        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask
        ).last_hidden_state

        return features, attention_mask
        

    def compute_llm_outputs(self, batch: Dict) -> torch.Tensor:

        features, encoder_attn_mask = self.compute_input_features(batch)

        llm_input_ids = None

        if "actions" in batch.keys():

            action_tokens, params = self.action_tokenizer.encode(batch["actions"], update_bounds=self.update_w_bound)

            llm_label_ids = self.action_tokenizer.tokens_to_llm_tokens(action_tokens)

            input_tokens = self.action_tokenizer.vocab_size//2 * torch.ones_like(llm_label_ids, dtype=torch.long, device=self.device)
            llm_input_ids = self.action_tokenizer.tokens_to_llm_tokens(input_tokens)

            
            ### Sanity Check, check if the reconstructed tokens are correct
            # for i in range(len(batch["actions"])):
                # self.action_tokenizer.visualize_reconstruction_error_with_llm_tokenizer(batch["actions"][i])

        bidirectional_mask = create_bidirectional_mask(
            batch_size=llm_label_ids.shape[0],
            seq_length=llm_label_ids.shape[1],
            device=self.device
        ) 

        decoder_outputs = self.vlm.get_decoder()(
            input_ids=llm_input_ids, # fix this, this should be the empty action token
            encoder_hidden_states=features,
            encoder_attention_mask=encoder_attn_mask,
            attention_mask=bidirectional_mask, # bidirectional attention for the decoder
            use_cache=True,
        )

        lm_logits = self.vlm.language_model.get_output_embeddings()(decoder_outputs[0])
        lm_logits = lm_logits + self.vlm.language_model.final_logits_bias.to(lm_logits.device)

        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(
            lm_logits.view(-1, self.vlm.config.vocab_size),
            llm_label_ids.view(-1),
        )

        ### Add compute reconstruction loss
        token_predict_accuracy = 0.0
        action_mse = 0.0
        if "actions" in batch.keys():
            pred_tokens = torch.argmax(lm_logits, dim=-1)
            token_predict_accuracy = self.token_prediction_accuracy(pred_tokens, llm_label_ids)
            reconstruct_traj = self.action_tokenizer.reconstruct_from_llm_tokens(pred_tokens, times=None)
            action_mse = F.mse_loss(reconstruct_traj, batch["actions"])


        return {
            'llm_loss': masked_lm_loss,
            'token_predict_accuarcy': token_predict_accuracy,
            'reconstruct_action_mse': action_mse,
        }


    def llm_generates(self, batch: Dict) -> torch.Tensor:
        """Encode observations using Florence-2"""
        features, encoder_attn_mask = self.compute_input_features(batch)

        input_tokens = self.action_tokenizer.vocab_size//2 * torch.ones((1, self.num_dof, self.num_basis), 
                                                                        dtype=torch.long, device=self.device)
        llm_input_ids = self.action_tokenizer.tokens_to_llm_tokens(input_tokens)

        bidirectional_mask = create_bidirectional_mask(
            batch_size=llm_input_ids.shape[0],
            seq_length=llm_input_ids.shape[1],
            device=self.device
        ) 

        decoder_outputs = self.vlm.get_decoder()(
            input_ids=llm_input_ids, # fix this, this should be the empty action token
            encoder_hidden_states=features,
            encoder_attention_mask=encoder_attn_mask,
            attention_mask=bidirectional_mask, # bidirectional attention for the decoder
            use_cache=True,
        )

        lm_logits = self.vlm.language_model.get_output_embeddings()(decoder_outputs[0])
        lm_logits = lm_logits + self.vlm.language_model.final_logits_bias.to(lm_logits.device)

        output_tokens = torch.argmax(lm_logits, dim=-1)

        return output_tokens


    def forward(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """
        Forward pass for inference.
        
        Args:
            obs: Dictionary of observations
            goal: Dictionary containing goal info
            
        Returns:
            Predicted action sequence
        """
        # batch = {'rgb_obs': obs, '"lang_text"': goal}
        rgb_static = obs["rgb_obs"]['rgb_static']
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']

        # Create batch for observation encoding
        batch = {
            "rgb_obs": {
                "rgb_static": rgb_static,
                "rgb_gripper": rgb_gripper
            },
            "lang_text": [goal["lang_text"]]
        }

        llm_action_tokens = self.llm_generates(batch)

        if not self.action_tokenizer.init_pos:
            actions = self.action_tokenizer.reconstruct_from_llm_tokens(llm_action_tokens, times=None)
        else:
            init_pos = self.pred_action_seq[:, -1, ...] if self.pred_action_seq is not None else None
            actions = self.action_tokenizer.reconstruct_from_llm_tokens(llm_action_tokens, times=None, init_p=init_pos)
        return actions

    def step(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """
        Do one step of inference, handling action chunking.
        
        Args:
            obs: Dictionary of observations
            goal: Dictionary containing goal info
            
        Returns:
            Current action prediction
        """
        if self.rollout_step_counter % self.multistep == 0:
            self.pred_action_seq = self(obs, goal)
        
        if not self.return_act_chunk:
            # Default: return current action
            current_action = self.pred_action_seq[0, self.rollout_step_counter]
            if len(current_action.shape) == 2:
                current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        else:
            # Return whole chunk for ALOHA setups
            current_action = self.pred_action_seq
            
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0
        
        return current_action

    def reset(self):
        """Reset model state for new rollout."""
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self.eval()

    def on_train_start(self):
        """Convert model to appropriate dtype on training start."""
        # Move core model components to appropriate device/dtype
        self.to(self.device)
        self.vlm.to(self.device)
        
    def on_validation_start(self):
        """Setup before validation starts."""
        self.eval()

    def on_validation_end(self):
        """Cleanup after validation ends."""
        self.train()

    def print_model_parameters(self):
        """Print model parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params}")
        
        for name, submodule in self.named_modules():
            if '.' not in name or name.count('.') <= 1:
                submodule_params = sum(p.numel() for p in submodule.parameters())
                if submodule_params > 0:
                    print(f"{name} - Total Params: {submodule_params}")
                    
    def print_encoded_texts(self, batch, device):
        """Print encoded text inputs for debugging."""
        text_embeds = self.vlm.get_input_embeddings()(
            batch[self.goal_modalities][self.lang_modalities[0]]['input_ids'].to(self.device)
        ).to(device).squeeze(1)
        
        input_ids = batch[self.goal_modalities][self.lang_modalities[0]]['input_ids'][0].squeeze(0).to(self.device)
        input_ids = input_ids.cpu()
        decoded_text = self.processor.tokenizer.decode(input_ids, skip_special_tokens=False)
        print("Original text:", decoded_text)

        decoded_texts = self.processor.tokenizer.batch_decode(text_embeds.cpu(), skip_special_tokens=True)
        print("Encoded texts:")
        for i, text in enumerate(decoded_texts):
            print(f"Sequence {i+1}: {text}")
    
    def construct_prompts(self, dataset_batch):
        """
        Constructs prompts for Florence-2's encoder to extract task-relevant visual features.
        
        Args:
            dataset_batch: Dictionary containing task information including language instructions
            
        Returns:
            text_prompts: List of formatted prompts for encoder conditioning
        """
        language_instruction = dataset_batch["lang_text"]
        text_prompts = []
        
        for instruction in language_instruction:
            if self.vlm_prompt_style == "default":
                # Original instruction only
                text_prompts.append(self.format_instruction(instruction))
                
            elif self.vlm_prompt_style == "feature_focused":
                # Focus on extracting visual features relevant for manipulation
                prompt = f"<od>{instruction}</od><grounding>identify objects and spatial relationships for robotic manipulation</grounding>"
                text_prompts.append(prompt)
                
            elif self.vlm_prompt_style == "state_oriented":
                # Focus on extracting state-relevant features
                prompt = f"<od>{instruction}</od><referring_expression_segmentation>locate objects and regions for manipulation</referring_expression_segmentation>"
                text_prompts.append(prompt)
                
            else:
                raise ValueError(f"Unknown prompt style: {self.vlm_prompt_style}")
        
        
        return text_prompts
    
    def _get_text_embeddings(self, text, device):
        """Get text embeddings to use with VLM"""
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        return self.vlm.get_input_embeddings()(text_inputs["input_ids"])
    
    def _log_training_metrics(self, llm_loss, token_pred_acc, reconstruct_mse, total_bs):
        """
        Log training metrics
        Args:
            total_loss: Total loss value
            action_loss: Action-specific loss value
            total_bs: Total batch size
        """
        self.log("train/llm_loss", llm_loss, on_step=False, on_epoch=True, 
                sync_dist=True, batch_size=total_bs)
        self.log("train/token_pred_acc", token_pred_acc, on_step=False, on_epoch=True, 
                sync_dist=True, batch_size=total_bs)
        self.log("train/reconstruct_mse", reconstruct_mse, on_step=False, on_epoch=True, 
                sync_dist=True, batch_size=total_bs)       

    def _log_validation_metrics(self, llm_loss, token_pred_acc, reconstruct_mse):
        """
        Log validation metrics
        Args:
            pred_loss: Prediction loss value (scalar)
            val_total_act_loss_pp: Total validation action loss per prediction
        """
        # Log per-modality action loss
        self.log(
            f"val/{self.modality_scope}_llm_loss", 
            llm_loss, 
            sync_dist=True
        )
        
        # Log average action loss across modalities
        # try:
        #     n_modalities = len(self.trainer.datamodule.modalities)
        # except AttributeError:
        #     n_modalities = 1  # Default if modalities not available
            
        self.log(
            "val/token_pred_acc",
            token_pred_acc,
            sync_dist=True
        )

        self.log(
            "val/reconstruct_mse",
            reconstruct_mse,
            sync_dist=True
        )

    # def setup(self, stage):
    #     # not working,ddp
    #     # at least not working at setup stage, all things seems still on cpu, and ddp not fully initialized.
    #     if stage != "fit":
    #         return
    #
    #     log_rank_0("precompute mp normalizer")
    #
    #     local_mp_params = []
    #     # params = []
    #
    #     loader = self.trainer.datamodule.train_dataloader()
    #
    #     for batch in tqdm(loader["lang"], desc=f"Rank_{self.global_rank}, precomputing weight normalizer of MP", unit="batch"):
    #         act_chunks = batch["actions"][..., :self.action_tokenizer.joint_dof]
    #         # !!! make sure the bounds are first set to -1 and 1 !!!
    #         param = self.action_tokenizer.compute_weights(act_chunks)
    #
    #         param = param.to("cpu").numpy()
    #         # params.append(param)
    #         local_mp_params.append(param)
    #
    #     # params = np.concatenate(params, axis=0)
    #     if self.global_rank==0:
    #         params = self.gather_results(local_mp_params)
    #
    #         # params_mean = params.mean(axis=0)
    #         # params_std = params.std(axis=0)
    #         # params_min = params.min(axis=0)
    #         # params_max = params.max(axis=0)
    #         params_min = np.quantile(params, 0.01, 0)
    #         params_max = np.quantile(params, 0.99, 0)
    #
    #         params_min = torch.from_numpy(params_min).to(self.device)
    #         params_max = torch.from_numpy(params_max).to(self.device)
    #
    #         self.action_tokenizer.w_min[:self.action_tokenizer.joint_dof * self.num_basis] = params_min
    #         self.action_tokenizer.w_max[:self.action_tokenizer.joint_dof * self.num_basis] = params_max
    #
    #     if self.trainer.world_size>1 and dist.is_initialized():
    #         dist.broadcast(self.action_tokenizer.w_min, src=0)
    #         dist.broadcast(self.action_tokenizer.w_max, src=0)
    #
    #     log_rank_0(f"mp weight normalizer calculated and set,"
    #                f"mp weight bounds min {self.action_tokenizer.w_min} and max {self.action_tokenizer.w_max}")
    #
    # def gather_results(self, local_list):
    #     # gather list of array into a array and return it to rank 0
    #     local_array = np.concatenate(local_list, axis=0)
    #
    #     if not (dist.is_available() and dist.is_initialized()):
    #         return local_array
    #
    #     world_size = dist.get_world_size()
    #     gathered = [None for _ in range(world_size)]
    #     dist.gather_object(local_array, gathered, dst=0)
    #     gathered = np.concatenate(gathered, axis=0)
    #     return gathered


    def on_fit_start(self,):

        if self.precompute_w_bound:
            self.precompute_mp_normalizer()
            # broadcast main process weight normalizer to other devices
            if self.trainer.world_size>1 and dist.is_initialized():
                dist.broadcast(self.action_tokenizer.w_min, src=0)
                dist.broadcast(self.action_tokenizer.w_max, src=0)
                logger.info(f"mp weight normalizer set in rank {self.global_rank}")
        else:
            logger.info("mp normalizer not precomputed, using default [-1, 1]")

    @rank_zero_only
    def precompute_mp_normalizer(self):
        logger.info("precompute mp normalizer")

        dataloader = self.trainer.datamodule.train_dataloader()

        params = []
        for batch in tqdm(dataloader["lang"], desc=f"Rank_{self.global_rank}, precomputing weight normalizer of MP", unit="batch"):
            act_chunks = batch["actions"][..., :self.action_tokenizer.joint_dof]
            act_chunks = act_chunks.to(self.device)
            # !!! make sure the bounds are first set to -1 and 1 !!!
            param = self.action_tokenizer.compute_weights(act_chunks)
            param = param.to("cpu").numpy()
            params.append(param)
            # Break if we have enough samples
            if len(params) > self.precompute_w_bound_steps:
                logger.info(f"Rank_{self.global_rank}, precomputed enough samples for weight normalizer of MP")
                break
        params = np.concatenate(params, axis=0)

        # params_mean = params.mean(axis=0)
        # params_std = params.std(axis=0)
        # params_min = params.min(axis=0)
        # params_max = params.max(axis=0)
        params_min = np.quantile(params, 0.01, 0)
        params_max = np.quantile(params, 0.99, 0)

        params_min = torch.from_numpy(params_min).to(self.action_tokenizer.w_min.device)
        params_max = torch.from_numpy(params_max).to(self.action_tokenizer.w_max.device)

        self.action_tokenizer.w_min[:self.action_tokenizer.joint_dof * self.num_basis] = params_min
        self.action_tokenizer.w_max[:self.action_tokenizer.joint_dof * self.num_basis] = params_max

        logger.info("mp_normalizer computed and set on rank 0")

