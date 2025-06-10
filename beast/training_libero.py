import logging
from pathlib import Path
import sys
sys.tracebacklimit = None
import os 
import wandb
import hydra
from omegaconf import DictConfig
import torch
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only


# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import beast.models.beast_florence as models_m
from beast.utils.utils import get_git_commit_hash, get_last_checkpoint, initialize_pretrained_weights, print_system_env_info

# Add local repo to path
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def clear_cuda_cache():
    """Clear CUDA cache and garbage collect unused memory."""
    if torch.cuda.is_available():
        # Empty CUDA cache
        torch.cuda.empty_cache()
        # Force garbage collection
        import gc
        gc.collect()
        # Log memory stats
        for i in range(torch.cuda.device_count()):
            memory_stats = torch.cuda.memory_stats(i)
            allocated = memory_stats.get('allocated_bytes.all.current', 0) / (1024**3)
            reserved = memory_stats.get('reserved_bytes.all.current', 0) / (1024**3)
            logger.info(f"GPU {i} Memory: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)

def setup_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    return [hydra.utils.instantiate(cb) for cb in callbacks_cfg.values()]

def setup_logger(cfg: DictConfig, model: LightningModule):
    pathlib_cwd = Path.cwd()
    if "group" in cfg.logger:
        cfg.logger.group = pathlib_cwd.parent.name
        # cfg.logger.name = f"{pathlib_cwd.parent.name}/{pathlib_cwd.name}"
        cfg.logger.name = cfg.logger.id
        cfg.logger.id = cfg.logger.name.replace("/", "_")
    return hydra.utils.instantiate(cfg.logger)

@hydra.main(config_path="../conf", config_name="config_libero")
def train(cfg: DictConfig) -> None:
    try:
        # Setup environment
        os.environ['HYDRA_FULL_ERROR'] = '1'
        # Set memory allocation configuration
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        seed_everything(cfg.seed, workers=True)
        torch.set_float32_matmul_precision('medium')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Clear CUDA cache before initialization
        clear_cuda_cache()
        
        # Initialize components
        log_rank_0(f"\nInitializing training for seed {cfg.seed}")
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        model = hydra.utils.instantiate(cfg.model) if get_last_checkpoint(Path.cwd()) is None else \
               getattr(models_m, cfg.model["_target_"].split(".")[-1]).load_from_checkpoint(get_last_checkpoint(Path.cwd()).as_posix())
        
        if "pretrain_chk" in cfg:
            initialize_pretrained_weights(model, cfg)
            
        # Setup training
        train_logger = setup_logger(cfg, model)
        callbacks = setup_callbacks(cfg.callbacks) + [LearningRateMonitor(logging_interval="step")]
        
        # Set unique working directory for each seed
        work_dir = Path.cwd() / f"seed_{cfg.seed}"
        work_dir.mkdir(exist_ok=True)
        os.chdir(work_dir)
        
        trainer_args = {
            **cfg.trainer,
            "logger": train_logger,
            "callbacks": callbacks,
            "benchmark": False,
            "strategy": "ddp_find_unused_parameters_true",
            "accelerator": "gpu",
            "devices": cfg.trainer.devices,
            "use_distributed_sampler": True,
            "default_root_dir": work_dir,
            "sync_batchnorm": True,
        }
        
        # Log configuration
        log_rank_0(f"Training config for seed {cfg.seed}:\n{cfg}")
        log_rank_0(f"Git commit: {get_git_commit_hash(Path(hydra.utils.to_absolute_path(__file__)))}")
        log_rank_0(print_system_env_info())
                
        # Clear CUDA cache again before training
        clear_cuda_cache()
        
        # Initialize trainer and train
        trainer = Trainer(**trainer_args)
        
        try:
            trainer.fit(model, datamodule=datamodule)
        except Exception as e:
            log_rank_0("\nDetailed Error Information:")
            log_rank_0("=" * 80)
            log_rank_0(f"Error Type: {type(e).__name__}")
            log_rank_0(f"Error Message: {str(e)}")
            log_rank_0("\nFull Traceback:")
            import traceback
            log_rank_0(''.join(traceback.format_tb(e.__traceback__)))
            log_rank_0("\nLocal Variables at Crash Point:")
            tb = e.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            log_rank_0(f"{traceback.extract_tb(tb)}")
            log_rank_0("=" * 80)
            raise e
                
    except Exception as e:
        logger.error(f"\nTraining failed for seed {cfg.seed}:")
        logger.error(f"{'='*80}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        import traceback
        logger.error(traceback.format_exc())
        logger.error(f"{'='*80}")
        raise e
    finally:
        # Clear CUDA cache one final time
        clear_cuda_cache()
        # Clean up
        cleanup_distributed()
        if wandb.run is not None:
            wandb.finish()

def cleanup_distributed():
    """Cleanup distributed training resources"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # Set environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    # Add repo to path
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
    
    try:
        train()
    except Exception as e:
        logger.error(f"\nTraining script failed:")
        logger.error(f"{'='*80}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        import traceback
        logger.error(traceback.format_exc())
        logger.error(f"{'='*80}")
        sys.exit(1)