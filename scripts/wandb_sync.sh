#!/bin/bash -l

#SBATCH --job-name=TRM_MazeHard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -A plgsoftmaxvit-gpu-gh200
#SBATCH --time=00:10:00
#SBATCH --partition=plgrid-gpu-gh200 
#SBATCH --output=./output/slurm_output/%j.out
#SBATCH --error=./output/slurm_error/%j.out
 

PROJECT_PATH=$SLURM_SUBMIT_DIR
cd $PROJECT_PATH
export LOGLEVEL=INFO

ml ML-bundle/25.04
source venv/bin/activate

# Optionally set WANDB_API_KEY if the file exists and is readable
if [ -r scripts/wandb_api_key.txt ]; then
    export WANDB_API_KEY=$(cat scripts/wandb_api_key.txt)
else
    echo "Warning: scripts/wandb_api_key.txt not found or not readable; skipping WANDB_API_KEY export (WANDB_MODE=$WANDB_MODE)" >&2
fi

wandb sync ./wandb/offline-run-20251010_133908-uajeth58
