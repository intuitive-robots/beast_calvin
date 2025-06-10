import contextlib
import logging
from pathlib import Path
from typing import Union
import importlib

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
import pyhash
import torch
from hydra.core.global_hydra import GlobalHydra

from beast.utils.utils import add_text, format_sftp_path

hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)


def load_class(name):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_evaluation_checkpoint(cfg):
    epoch = cfg.epoch_to_load if "epoch_to_load" in cfg else -1
    overwrite_cfg = cfg.overwrite_module_cfg if "overwrite_module_cfg" in cfg else {}
    module_path = str(Path(cfg.module_path).expanduser())
    pl_module = load_pl_module_from_checkpoint(
        module_path,
        epoch=epoch,
        overwrite_cfg=overwrite_cfg,
    ).cuda()
    return pl_module


def get_checkpoint_i_from_dir(dir, i: int = -1):
    ckpt_paths = list(dir.rglob("*.ckpt"))
    if i == -1:
        for ckpt_path in ckpt_paths:
            if ckpt_path.stem == "last":
                return ckpt_path

    # Search for ckpt of epoch i
    for ckpt_path in ckpt_paths:
        split_path = str(ckpt_path).split("_")
        for k, word in enumerate(split_path):
            if word == "epoch":
                if int(split_path[k + 1]) == i:
                    return ckpt_path

    sorted(ckpt_paths, key=lambda f: f.stat().st_mtime)
    return ckpt_paths[i]


def get_config_from_dir(dir):
    dir = Path(dir)
    config_yaml = list(dir.rglob("*hydra/config.yaml"))[0]
    return OmegaConf.load(config_yaml)


def load_pl_module_from_checkpoint(
    filepath: Union[Path, str],
    epoch: int = 1,
    overwrite_cfg: dict = {},
    use_ema_weights: bool = False
):
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if filepath.is_dir():
        filedir = filepath
        ckpt_path = get_checkpoint_i_from_dir(dir=filedir, i=epoch)
    elif filepath.is_file():
        assert filepath.suffix == ".ckpt", "File must have .ckpt extension"
        ckpt_path = filepath
        filedir = filepath.parents[0]
    else:
        raise ValueError(f"not valid file path: {str(filepath)}")
    config = get_config_from_dir(filedir)
    class_name = config.model.pop("_target_")
    if "_recursive_" in config.model:
        del config.model["_recursive_"]
    print(f"class_name {class_name}")
    module_class = load_class(class_name)
    print(f"Loading model from {ckpt_path}")
    load_cfg = {**config.model, **overwrite_cfg}
    model = module_class.load_from_checkpoint(ckpt_path, **load_cfg)
     # Load EMA weights if they exist and the flag is set
    if use_ema_weights:
        checkpoint_data = torch.load(ckpt_path)
        if "ema_weights" in checkpoint_data['callbacks']['EMA']:
            ema_weights_list = checkpoint_data['callbacks']['EMA']['ema_weights']

            # Convert list of tensors to a state_dict format
            ema_weights_dict = {name: ema_weights_list[i] for i, (name, _) in enumerate(model.named_parameters())}

            model.load_state_dict(ema_weights_dict)
            print("Successfully loaded EMA weights from checkpoint!")
        else:
            print("Warning: No EMA weights found in checkpoint!")

    print(f"Finished loading model {ckpt_path}")
    return model



def get_default_model_and_env(train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None, device_id=0):
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    cfg = OmegaConf.load(train_cfg_path)
    lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf/datamodule/datasets")
    # we don't want to use shm dataset for evaluation
    datasets_cfg = hydra.compose("vision_lang.yaml", overrides=["lang_dataset.lang_folder=" + lang_folder])
    # since we don't use the trainer during inference, manually set up data_module
    cfg.datamodule.datasets = datasets_cfg
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["lang"]
    device = torch.device(f"cuda:{device_id}")

    if lang_embeddings is None:
        lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)

    if env is None:
        rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "conf/callbacks/rollout/default.yaml")
        env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)

    checkpoint = format_sftp_path(checkpoint)
    print(f"Loading model from {checkpoint}")

    # new stuff
    epoch = cfg.epoch_to_load if "epoch_to_load" in cfg else -1
    overwrite_cfg = cfg.overwrite_module_cfg if "overwrite_module_cfg" in cfg else {}
    module_path = str(Path(train_folder).expanduser())
    model = load_pl_module_from_checkpoint(
        module_path,
        epoch=epoch,
        overwrite_cfg=overwrite_cfg,
    )
    # model = Hulc.load_from_checkpoint(checkpoint)
    model.freeze()
    if cfg.model.action_decoder.get("load_action_bounds", False):
        model.action_decoder._setup_action_bounds(cfg.datamodule.root_data_dir, None, None, True)
    model = model.cuda(device)
    print("Successfully loaded model.")

    return model, env, data_module, lang_embeddings


def get_default_mode_and_env(train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None, prep_dm_and_deps=True, device_id=0, eval_cfg_overwrite={}):
    # Fix for the path issue - ensure we're working with the directory containing the config
    train_folder_path = Path(train_folder)
    
    # If the train_folder is already pointing to a .yaml file, use its parent directory
    if train_folder_path.suffix == '.yaml':
        train_folder_path = train_folder_path.parent.parent  # Go up two levels from config.yaml
    
    # Now construct the correct path to the config.yaml file
    train_cfg_path = train_folder_path / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    
    print(f"Loading config from: {train_cfg_path}")
    
    def_cfg = OmegaConf.load(train_cfg_path)
    eval_override_cfg = OmegaConf.create(eval_cfg_overwrite)
    cfg = OmegaConf.merge(def_cfg, eval_override_cfg)
    lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
    
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf/datamodule/datasets")
    
    if device_id != 'cpu':
        device = torch.device(f"cuda:{device_id}")
    else:
        device = 'cpu'
    
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    
    if prep_dm_and_deps:
        data_module.prepare_data()
        data_module.setup()
        dataloader = data_module.val_dataloader()
        dataset = dataloader["lang"].dataset

        if lang_embeddings is None:
            lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)

        if env is None:
            rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "conf/callbacks/rollout_lh/calvin.yaml")
            env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=False)

    # Fix for checkpoint path handling
    checkpoint_path = Path(checkpoint).expanduser()
    print(f"Loading model from {checkpoint_path}")
    
    model = load_mode_from_safetensor(
        checkpoint_path,
        overwrite_cfg=eval_cfg_overwrite.get("model", {}),
    )
    
    model.freeze()
    model = model.cuda(device)
    print("Successfully loaded model.")

    return model, env, data_module, lang_embeddings

def load_mode_from_safetensor(
    filepath: Path,
    overwrite_cfg: dict = {},
):
    """Load model from a checkpoint file or directory.
    
    Args:
        filepath: Path to the checkpoint file or directory
        overwrite_cfg: Dict with configuration overrides
        
    Returns:
        Instantiated model
    """
    filepath = Path(filepath)
    
    # Determine if we're dealing with a file or directory
    if filepath.is_file():
        # If it's a checkpoint file, use its parent directory to find the config
        ckpt_path = filepath
        config_dir = filepath.parent
        
        # Try to find config in parent directories
        hydra_dir = None
        current_dir = config_dir
        for _ in range(4):  # Look up to 3 levels up
            if (current_dir / ".hydra").exists():
                hydra_dir = current_dir / ".hydra"
                break
            current_dir = current_dir.parent
        
        if hydra_dir is None:
            # If we can't find .hydra directory, try looking at a common pattern
            # From the filepath, try to find base directory (e.g., calvin_abcd)
            parts = filepath.parts
            try:
                # Look for "best_checkpoints" in the path
                idx = parts.index("best_checkpoints")
                if idx + 1 < len(parts):
                    base_dir = Path(*parts[:idx+2])
                    if (base_dir / ".hydra").exists():
                        hydra_dir = base_dir / ".hydra"
            except ValueError:
                pass
                
        if hydra_dir is None:
            raise ValueError(f"Could not find .hydra directory for checkpoint: {str(filepath)}")
            
        config_path = hydra_dir / "config.yaml"
    elif filepath.is_dir():
        # If it's a directory, look for .hydra/config.yaml
        ckpt_path = filepath
        if (filepath / ".hydra").exists():
            config_path = filepath / ".hydra/config.yaml"
        else:
            raise ValueError(f"Directory does not contain .hydra/config.yaml: {str(filepath)}")
    else:
        raise ValueError(f"Path does not exist: {str(filepath)}")
    
    print(f"Loading config from: {config_path}")
    config = OmegaConf.load(config_path)
    
    print(f"Loading model from {ckpt_path}")
    load_cfg = OmegaConf.create({**OmegaConf.to_object(config.model), **{"optimizer": None}, **overwrite_cfg})
    
    # Remove 'ckpt_path' if it exists in load_cfg to avoid the error
    if 'ckpt_path' in load_cfg:
        del load_cfg['ckpt_path']
    
    # Set the pretrained model path
    load_cfg["load_pretrained"] = True
    load_cfg["pretrained_model_path"] = str(ckpt_path)
    
    # Instantiate the model
    model = hydra.utils.instantiate(load_cfg)

    print(f"Finished loading model {ckpt_path}")
    return model


def join_vis_lang(img, lang_text):
    """Takes as input an image and a language instruction and visualizes them with cv2"""
    img = img[:, :, ::-1].copy()
    img = cv2.resize(img, (500, 500))
    add_text(img, lang_text)
    cv2.imshow("simulation cam", img)
    cv2.waitKey(1)


class LangEmbeddings:
    def __init__(self, val_dataset_path, lang_folder, device=torch.device("cuda:0")):
        embeddings = np.load(Path(val_dataset_path) / lang_folder / "embeddings.npy", allow_pickle=True).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}
        self.device = device

    def get_lang_goal(self, task):
        return {"lang": torch.from_numpy(self.lang_embeddings[task]).to(self.device).squeeze(0).float()}


def imshow_tensor(window, img_tensor, wait=0, resize=True, keypoints=None, text=None):
    img_tensor = img_tensor.squeeze()
    img = np.transpose(img_tensor.cpu().numpy(), (1, 2, 0))
    img = np.clip(((img / 2) + 0.5) * 255, 0, 255).astype(np.uint8)

    if keypoints is not None:
        key_coords = np.clip(keypoints * 200 + 100, 0, 200)
        key_coords = key_coords.reshape(-1, 2)
        cv_kp1 = [cv2.KeyPoint(x=pt[1], y=pt[0], _size=1) for pt in key_coords]
        img = cv2.drawKeypoints(img, cv_kp1, None, color=(255, 0, 0))

    if text is not None:
        add_text(img, text)

    if resize:
        cv2.imshow(window, cv2.resize(img[:, :, ::-1], (500, 500)))
    else:
        cv2.imshow(window, img[:, :, ::-1])
    cv2.waitKey(wait)


def print_task_log(demo_task_counter, live_task_counter, mod):
    print()
    logger.info(f"Modality: {mod}")
    for task in demo_task_counter:
        logger.info(
            f"{task}: SR = {(live_task_counter[task] / demo_task_counter[task]) * 100:.0f}%"
            + f" |  {live_task_counter[task]} of {demo_task_counter[task]}"
        )
    s = sum(demo_task_counter.values())
    success_rate = (sum(live_task_counter.values()) / s if s > 0 else 0) * 100
    logger.info(f"Average Success Rate {mod} = {success_rate:.0f}%")
    logger.info(
        f"Success Rates averaged throughout classes = {np.mean([live_task_counter[task] / demo_task_counter[task] for task in demo_task_counter]) * 100:.0f}%"
    )


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_env_state_for_initial_condition(initial_condition):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    # we want to have a "deterministic" random seed for each initial condition
    seed = hasher(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs