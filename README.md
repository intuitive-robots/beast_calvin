

# BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning

[**Paper**]([https://www.google.com/search?q=https://openreview.net/pdf%3Fid%3DrQCl1sf62w](https://arxiv.org/pdf/2506.06072)) | [**Project Page**](https://intuitive-robots.github.io/beast_website/) | **NeurIPS 2025**

This repository is the official implementation for the paper **BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning** (NeurIPS 2025).

BEAST introduces a novel, highly efficient action representation for imitation learning. By encoding action sequences using B-Splines, it creates a compact, continuous, and expressive tokenization of robot trajectories. Our primary model, **BEAST-F**, models this continuous action-token space using a Rectified Flow model, achieving state-of-the-art performance on challenging long-horizon benchmarks, including CALVIN and LIBERO.

This repository provides all the code necessary to download the datasets, preprocess the data, and reproduce our training and evaluation results.

## Installation

To begin, clone this repository locally

```bash
git clone git@github.com:intuitive-robots/beast_calvin.git
export BEAST_ROOT=$(pwd)/beast_calvin
```

Install requirements
(Note we provided a changed verison of pyhash, given numerous problems we encountered when installing it manually on our slurm cluster)
You can also try to install setup tools using `pip`.

```bash
cd $BEAST_ROOT
conda create -n beast_cal python=3.9
conda activate beast_cal
conda install cmake
cd calvin_env/tacto
pip install -e .
cd ..
pip install -e .
cd ..
cd LIBERO
pip install -r requirements.txt
pip install -e .
pip install numpy~=1.23
cd ..
pip install setuptools==57.5.0
conda install conda-forge::pyhash
cd MP_lite_PyTorch
pip install -e .
pip install addict
cd ..
```

Next we can install the rest of the missing packages

```bash
pip install -r requirements.txt
```

-----

## Download

### CALVIN Dataset

If you want to train on the [CALVIN](https://github.com/mees/calvin) dataset, choose a split with:

```bash
cd $BEAST_ROOT/dataset
sh download_data.sh D | ABCD
```

### LIBERO Dataset

If you want to train on the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) dataset, choose a split with:

```bash
cd $BEAST_ROOT/LIBERO
python benchmark_scripts/download_libero_datasets.py --datasets DATASET_NAME
```

where `DATASET_NAME` is chosen from `[libero_spatial, libero_object, libero_100, libero_goal]`.

## Training

To train the BEAST-F with the 4 GPUS, run:

```
python beast/training_calvin.py 
```

Note that during training the full CALVIN eval or LIBERO rollouts will be called every n\*1k training steps.

For replication of the orginial training results I recommend to use 4 GPUs with a batch\_size of 8 and train them for 40k steps for ABC (ABCD).
See configs for details.

#### Preprocessing with CALVIN

Since BEAST uses action chunking, it needs to load multiple (\~10) `episode_{}.npz` files for each inference. In combination with batching, this results in a large disk bandwidth needed for each iteration (usually \~2000MB/iteration).
This has the potential of significantly reducing your GPU utilization rate during training depending on your hardware.
Therefore, you can use the script `extract_by_key.py` to extract the data into a single file, avoiding opening too many episode files when using the CALVIN dataset.

##### Usage example:

```shell
python preprocess/extract_by_key.py -i /YOUR/PATH/TO/CALVIN/ \
    --in_task all
```

##### Params:

Run this command to see more detailed information:

```shell
python preprocess/extract_by_key.py -h
```

Important params:

  * `--in_root`: `/YOUR/PATH/TO/CALVIN/`, e.g `/data3/geyuan/datasets/CALVIN/`
  * `--extract_key`: A key of `dict(episode_xxx.npz)`, default is **'rel\_actions'**, the saved file name depends on this (i.e `ep_{extract_key}.npy`)

Optional params:

  * `--in_task`: default is **'all'**, meaning all task folders (e.g `task_ABCD_D/`) of CALVIN
  * `--in_split`: default is **'all'**, meaning both `training/` and `validation/`
  * `--out_dir`: optional, default is **'None'**, and will be converted to `{in_root}/{in_task}/{in_split}/extracted/`
  * `--force`: whether to overwrite existing extracted data

-----

## Acknowledgements

This work is only possible because of the code from the following open-source projects and datasets. We thank all authors for their work:

#### CALVIN

Original: [https://github.com/mees/calvin](https://github.com/mees/calvin)
License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)

#### LIBERO

Original: [https://github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
License: [https://github.com/Lifelong-Robot-Learning/LIBERO?tab=MIT-1-ov-file](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=MIT-1-ov-file)

#### Mimictest

[mimictest](https://github.com/EDiRobotics/mimictest)

#### HULC

Original: [https://github.com/lukashermann/hulc](https://github.com/lukashermann/hulc)
License: [MIT](https://github.com/lukashermann/hulc/blob/main/LICENSE)

#### MP\_lite\_PyTorch

Original: [https://github.com/Andrewllab/MP\_lite\_PyTorch](https://github.com/Andrewllab/MP_lite_PyTorch)
License: [GPL](https://github.com/Andrewllab/MP_lite_PyTorch/blob/main/LICENSE)

#### FLOWER

Original: [https://github.com/intuitive-robots/flower\_vla\_calvin](https://github.com/intuitive-robots/flower_vla_calvin)
License: [MIT](https://github.com/intuitive-robots/flower_vla_calvin/blob/main/LICENSE)

-----

## BibTeX

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{
    zhou2025beast,
    title={{BEAST}: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning},
    author={Hongyi Zhou and Weiran Liao and Xi Huang and Yucheng Tang and Fabian Otto and Xiaogang Jia and Xinkai Jiang and Simon Hilber and Ge Li and Qian Wang and {\"O}mer Erdin{\c{c}} Ya{\u{g}}murlu and Nils Blank and Moritz Reuss and Rudolf Lioutikov},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=rQCl1sf62w}
}
```



---
