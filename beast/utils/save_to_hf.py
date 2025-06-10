import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from huggingface_hub import HfApi, upload_folder
from safetensors.torch import save_file
from pathlib import Path
import shutil
import json

def save_config(config: DictConfig, save_path: str):
    """Save the model config as a JSON file."""
    # Convert OmegaConf to dict and then to JSON
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Clean up the config to remove unnecessary fields
    cleaned_config = {
        "model_config": {
            "latent_dim": config.latent_dim,
            "obs_enc_dim": config.obs_enc_dim,
            "cond_dim": config.cond_dim,
            "resnet_type": config.resnet_type,
            "model": {
                "inner_model": config.model.inner_model
            },
            "multistep": config.multistep,
            "sampler_type": config.sampler_type,
            "num_sampling_steps": config.num_sampling_steps,
            "sigma_data": config.sigma_data,
            "sigma_min": config.sigma_min,
            "sigma_max": config.sigma_max,
            "noise_scheduler": config.noise_scheduler,
            "sigma_sample_density_type": config.sigma_sample_density_type,
            "act_window_size": config.act_window_size,
            "use_proprio": config.use_proprio
        }
    }
    
    with open(os.path.join(save_path, "config.json"), 'w') as f:
        json.dump(cleaned_config, f, indent=2)

def create_model_card(save_path: str):
    """Create a README.md file with model information."""
    model_card = """# MoDE (Mixture of Diffusion Experts) Model

This is the pretrained MoDE model for robotic manipulation. The model uses a mixture of denoising experts architecture designed for learning robotic manipulation policies.

## Model Description

MoDE is a novel architecture that combines:
- Mixture of Experts Noise-only routing
- Diffusion-based action generation with noise-cond self attention
- Multi-modal input processing (vision + language)

The model can be used for:
- Language-conditioned robotic manipulation

## Usage

```python
from huggingface_hub import snapshot_download
import torch
import hydra
from omegaconf import OmegaConf

# Download the model
model_path = snapshot_download(repo_id="your-username/mode-dp")

# Load config
with open(os.path.join(model_path, "config.json")) as f:
    config = OmegaConf.create(json.load(f))

# Initialize model
model = hydra.utils.instantiate(config.model_config)

# Load weights
state_dict = torch.load(os.path.join(model_path, "model.pt"))
model.load_state_dict(state_dict)
```

## Training

The model was pretrained on a large-scale robotic manipulation dataset. It can be fine-tuned on custom datasets.

## License

[Add your license information here]

## Citation

[Add citation information if available]
"""
    
    with open(os.path.join(save_path, "README.md"), 'w') as f:
        f.write(model_card)

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Create a temporary directory for the model files
    temp_dir = "hf_upload_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    print("Loading model...")
    # Initialize model
    model = hydra.utils.instantiate(cfg)
    
    # Load checkpoint
    ckpt_path = cfg.ckpt_path
    if os.path.isdir(ckpt_path):
        # Find the latest checkpoint in the directory
        ckpt_files = [f for f in os.listdir(ckpt_path) if f.endswith('.ckpt')]
        if not ckpt_files:
            raise ValueError(f"No checkpoint files found in {ckpt_path}")
        ckpt_path = os.path.join(ckpt_path, sorted(ckpt_files)[-1])
    
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Clean up the state dict
    state_dict = checkpoint['state_dict']
    cleaned_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    print("Saving model files...")
    # Save the model weights in both PyTorch and safetensors formats
    torch.save({'state_dict': cleaned_state_dict}, os.path.join(temp_dir, "model.pt"))
    save_file(cleaned_state_dict, os.path.join(temp_dir, "model.safetensors"))
    
    # Save the config
    save_config(cfg, temp_dir)
    
    # Create the model card
    create_model_card(temp_dir)
    
    print("Uploading to Hugging Face Hub...")
    # Initialize Hugging Face API
    api = HfApi()
    
    # Upload to Hugging Face Hub
    # Replace with your desired repository name
    repo_id = "your-username/mode-diffusion"
    
    try:
        api.create_repo(repo_id, exist_ok=True)
        upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload MoDE model"
        )
        print(f"Successfully uploaded model to {repo_id}")
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {e}")
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()