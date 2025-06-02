[![arXiv](https://img.shields.io/badge/arXiv-2406.04343-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2406.04343)
[![ProjectPage](https://img.shields.io/badge/Project_Page-Flash3D-blue)](https://www.robots.ox.ac.uk/~vgg/research/flash3d/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-yellow)](https://huggingface.co/spaces/szymanowiczs/flash3d) 


# Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image


<p align="center">
  <img src="assets/teaser_video.gif" alt="animated" />
</p>

> [Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image](https://www.robots.ox.ac.uk/~vgg/research/flash3d/)  
> Stanislaw Szymanowicz, Eldar Insafutdinov, Chuanxia Zheng, Dylan Campbell, João F. Henriques, Christian Rupprecht, Andrea Vedaldi  
> 3DV, 2025.
> *[arXiv 2406.04343](https://arxiv.org/pdf/2406.04343.pdf)*  

# News
- [x] `19.07.2024`: Training code and data release

# Setup

## Create a python environment

Flash3D has been trained and tested with the followings software versions:

- Python 3.10
- Pytorch 2.2.2
- CUDA 11.8
- GCC 11.2 (or more recent)

Begin by installing CUDA 11.8 and adding the path containing the `nvcc` compiler to the `PATH` environmental variable.
Then the python environment can be created either via conda:

```sh
conda create -y python=3.10 -n flash3d
conda activate flash3d
```

or using Python's venv module (assuming you already have access to Python 3.10 on your system):

```sh
python3.10 -m venv .venv
. .venv/bin/activate
```

Finally, install the required packages as follows:

```sh
pip install -r requirements-torch.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Download training data

### RealEstate10K dataset

For downloading the RealEstate10K dataset we base our instructions on the [Behind The Scenes](https://github.com/Brummi/BehindTheScenes/tree/main?tab=readme-ov-file#-datasets) scripts.
First you need to download the video sequence metadata including camera poses from https://google.github.io/realestate10k/download.html and unpack it into `data/` such that the folder layout is as follows:

```
data/RealEstate10K/train
data/RealEstate10K/test
```

Finally download the training and test sets of the dataset with the following commands:

```sh
python datasets/download_realestate10k.py -d data/RealEstate10K -o data/RealEstate10K -m train
python datasets/download_realestate10k.py -d data/RealEstate10K -o data/RealEstate10K -m test
```

This step will take several days to complete. Finally, download additional data for the RealEstate10K dataset.
In particular, we provide pre-processed COLMAP cache containing sparse point clouds which are used to estimate the scaling factor for depth predictions.
The last two commands filter the training and testing set from any missing video sequences.

```sh
sh datasets/dowload_realestate10k_colmap.sh
python -m datasets.preprocess_realestate10k -d data/RealEstate10K -s train
python -m datasets.preprocess_realestate10k -d data/RealEstate10K -s test
```

## Download and evaluate the pretrained model

We provide model weights that could be downloaded and evaluated on RealEstate10K test set:

```sh
python -m misc.download_pretrained_models -o exp/re10k_v2
sh evaluate.sh exp/re10k_v2
```

## Training

In order to train the model on RealEstate10K dataset execute this command:
```sh
python train.py \
  +experiment=layered_re10k \
  model.depth.version=v1 \
  train.logging=false 
```

For multiple GPU, we can run with this command:
```sh
sh train.sh
```
You can modify the cluster information in ```configs/hydra/cluster```.


## BibTeX
```
@article{szymanowicz2024flash3d,
      author = {Szymanowicz, Stanislaw and Insafutdinov, Eldar and Zheng, Chuanxia and Campbell, Dylan and Henriques, Joao and Rupprecht, Christian and Vedaldi, Andrea},
      title = {Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image},
      journal = {arxiv},
      year = {2024},
}
```




