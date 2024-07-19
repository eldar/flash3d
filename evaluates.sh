#!/bin/sh

# kitti testing
# python evaluate.py \
# hydra.run.dir=$1 \
# hydra.job.chdir=true \
# +experiment=layered_kitti \
# +dataset.crop_border=true \
# dataset.pad_border_aug=0 \
# model.depth.version=v2 \
# ++eval.save_vis=false

# nyuv2 testing
python evaluate.py \
hydra.run.dir=$1 \
hydra.job.chdir=true \
+experiment=layered_nyuv2 \
+dataset.crop_border=true \
dataset.pad_border_aug=32 \
model.depth.version=v2 \
++eval.save_vis=false

# re10k testing
# python evaluate.py \
# hydra.run.dir=$1 \
# hydra.job.chdir=true \
# +experiment=layered_re10k \
# +dataset.crop_border=true \
# dataset.test_split_path=/users/cxzheng/code/facilitate4d/splits/re10k_mine_filtered/test_files.txt \
# model.depth.version=v2 \
# ++eval.save_vis=false


# python evaluate.py +experiment=kitti_unidepth_extension model.gaussians_per_pixel=2 dataset.pad_border_aug=0 dataset.height=128 dataset.width=384 +dataset.crop_border=true dataset.precise_intrinsics=true
# python evaluate.py +experiment=layered_si_re10k model.gaussians_per_pixel=2 +dataset.crop_border=true
# CUDA_VISIBLE_DEVICES=3 python evaluate.py +experiment=layered_si_nyuv2 model.gaussians_per_pixel=2 +dataset.crop_border=true dataset.pad_border_aug=32
# CUDA_VISIBLE_DEVICES=1 python evaluate.py +experiment=layered_si_re10k model.gaussians_per_pixel=2 dataset.test_split=latentsplat_closer dataset.width=256 +dataset.crop_border=false
# CUDA_VISIBLE_DEVICES=1 python evaluate.py +experiment=layered_si_re10k model.gaussians_per_pixel=2 dataset.test_split=pixelsplat_closer +dataset.crop_border=false dataset.width=256