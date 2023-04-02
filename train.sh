#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:2 # Request GPU "generic resources"
#SBATCH --cpus-per-task=2 # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --tasks-per-node=2
#SBATCH --mem=126000M # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --job-name=train_trail1
#SBATCH --time=24:00:00
#SBATCH --output=./train_fullds.log
#SBATCH --error=./train_fullds.err
#SBATCH --verbose

module load python
module load scipy-stack
source #YOUR VIRTUAL ENV NAME/bin/activate
cd #YOUR PROJECT PATH

wandb offline
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

srun python3 main.py --train --viz_wandb csc2516_proj --c2 0.2 --lambda1 0.2 --lambda2 0.2 --cross_att --slurm --world_size 2 --init_method tcp://$MASTER_ADDR:3456