#!/bin/sh

# re10k testing
python evaluate.py \
    hydra.run.dir=$1 \
    hydra.job.chdir=true \
    +experiment=layered_re10k \
    +dataset.crop_border=true \
    dataset.test_split_path=splits/re10k_mine_filtered/test_files.txt \
    model.depth.version=v1 \
    ++eval.save_vis=false

