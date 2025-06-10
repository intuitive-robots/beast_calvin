import os
import torch


from beast.models.tokenizers.utils import continuous_to_discrete, discrete_to_continuous
from beast.models.tokenizers.base_tokenizer import TokenizerBase

import numpy as np
import matplotlib.pyplot as plt
import einops


class NaiveTokenizer(TokenizerBase):

    
    def __init__(self, num_dof=1, seq_len=50, vocab_size=256, device="cuda"):

        super().__init__()

        self.device = device

        self.num_dof = num_dof
        self.seq_len = seq_len
        self.num_basis = seq_len
        self.vocab_size = vocab_size
        self.init_pos = False
    
    def update_vlm_vocab_size(self, vlm_vocab_size):
        self.vlm_vocab_size = vlm_vocab_size
    
    def encode(self, trajs, update_bounds=False): 
        flat_trajs = einops.rearrange(trajs, "b t d -> b (t d)")
        tokens = continuous_to_discrete(flat_trajs, num_bins=self.vocab_size, min_val=-1, max_val=1)
        return tokens, None
    
    def tokens_to_llm_tokens(self, tokens):
        if len(tokens.shape) == 3:
            tokens = einops.rearrange(tokens, 'b t d -> b (t d)')
        if self.vlm_vocab_size is None:
            raise ValueError("VLM vocab size is not set.")
        llm_tokens = self.vlm_vocab_size - 1 - tokens
        return llm_tokens
    
    def llm_tokens_to_mp_tokens(self, llm_tokens):
        if self.vlm_vocab_size is None:
            raise ValueError("VLM vocab is not set.")
        tokens = self.vlm_vocab_size - 1 - llm_tokens
        if len(tokens.shape) == 2:
            tokens = einops.rearrange(tokens, 'b (t d) -> b t d', t=self.num_basis, d=self.num_dof)
        return tokens
    
    def decode(self, tokens, **kwargs):
        trajs = discrete_to_continuous(tokens, num_bins=self.vocab_size, min_val=-1, max_val=1)
        return trajs
    
    @torch.no_grad()
    def reconstruct_traj(self, tokens, **kwargs):
        trajs = self.decode(tokens, **kwargs)
        return trajs
    
    def reconstruct_from_llm_tokens(self, llm_tokens, **kwargs):
        tokens = self.llm_tokens_to_mp_tokens(llm_tokens)
        trajs = self.decode(tokens, **kwargs)
        return trajs
    

    def visualize_reconstruction_error_with_llm_tokenizer(self, raw_traj, save_path=None):
            raw_traj = raw_traj.to(torch.float32)
            if len(raw_traj.shape) == 2:
                raw_traj = raw_traj.unsqueeze(0)
            tokens, params_dict = self.encode(raw_traj, update_bounds=True)
            llm_tokens = self.tokens_to_llm_tokens(tokens)
            # reconstruct the trajectory from the llm tokens
            pos = self.reconstruct_from_llm_tokens(llm_tokens)
            pos = pos.detach().cpu().numpy()
            raw_traj = raw_traj.detach().cpu().numpy()
            x_vals = np.linspace(0, 1.0, raw_traj.shape[1])

            batch_size, time_steps, dof = raw_traj.shape
            # Plot both generated and ground truth sine waves
            for sample_idx in range(batch_size):
                fig, axes = plt.subplots(dof, 1, figsize=(8, 2 * dof), sharex=True)

                for i in range(dof):
                    axes[i].plot(x_vals, pos[sample_idx, :, i], marker='o', label='reconstruct', linestyle='-',
                                 color='b')
                    axes[i].plot(x_vals, raw_traj[sample_idx, :, i], marker='*', label='ground_truth', linestyle='--',
                                 color='r')
                    axes[i].set_ylabel(f"DOF {i + 1}")
                    axes[i].grid(True)
                    axes[i].legend(loc="best")

                axes[-1].set_xlabel("Timesteps")
                plt.suptitle(f"Visualization of Sample {sample_idx} in Batch")
                plt.tight_layout(rect=[0, 0, 1, 0.96])

                # Save the figure with the specified naming format
                os.makedirs(save_path, exist_ok=True)
                save_filename = os.path.join(save_path, f"sample_{sample_idx}.png")
                plt.savefig(save_filename, dpi=300, bbox_inches='tight')
                # Close the figure to free memory (important when processing many plots)
                plt.show()
                plt.close(fig) 
