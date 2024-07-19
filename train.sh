# python train_fabric.py \
# hydra.run.dir=/work/cxzheng/3D/splatvideo/eldar/test/ \
# +experiment=layered_nyuv2 \
# run.debug=true

python train.py \
  hydra=cluster \
  hydra/launcher=submitit_slurm \
  +hydra.job.tag=fixedgpubug_gaussian2_unidepthv1 \
  +experiment=layered_re10k \
  model.depth.version=v1 \
  train.logging=false \
  -m
