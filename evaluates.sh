#!/bin/sh

# kitti testing
# python evaluate.py \
# hydra.run.dir=$1 \
# hydra.job.chdir=true \
# +experiment=layered_kitti \
# +dataset.crop_border=true \
# dataset.pad_border_aug=0 \
# model.depth.version=v1 \
# ++eval.save_vis=false

# nyuv2 testing
python evaluate.py \
hydra.run.dir=$1 \
hydra.job.chdir=true \
+experiment=layered_nyuv2 \
+dataset.crop_border=true \
dataset.pad_border_aug=32 \
model.depth.version=v1 \
++eval.save_vis=false

# re10k testing
# python evaluate.py \
# hydra.run.dir=$1 \
# hydra.job.chdir=true \
# +experiment=layered_re10k \
# +dataset.crop_border=true \
# dataset.test_split_path=/users/cxzheng/code/facilitate4d/splits/re10k_mine_filtered/test_files.txt \
# model.depth.version=v1 \
# ++eval.save_vis=false