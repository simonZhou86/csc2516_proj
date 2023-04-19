# CSC2516 Course Project

MTSegFormer: A Multitask Learning Approach for Brain Tumor Segmentation using Transformer

## Usage

Local
```shell
python main.py --train --viz_wandb (team name) --c1 0.7 --c2 0.3 --lambda1 0.2 --lambda2 0.2 --cross_att
```
On Compute Canada, fill out train.sh first, then run:
```shell
sbatch train.sh
```

This repo contains all code files for this project for now, will clean up & make it clear later
