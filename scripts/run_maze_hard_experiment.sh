#!/bin/bash -l

#SBATCH --job-name=TRM_MazeHard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH -A plgsoftmaxvit-gpu-gh200
#SBATCH --time=1:00:00
#SBATCH --partition=plgrid-gpu-gh200 
#SBATCH --output=./output/slurm_output/%j.out
#SBATCH --error=./output/slurm_error/%j.out
 

PROJECT_PATH=$SLURM_SUBMIT_DIR
cd $PROJECT_PATH

export WANDB_MODE=offline
export LOGLEVEL=INFO

# Optionally set WANDB_API_KEY if the file exists and is readable
if [ -r scripts/wandb_api_key.txt ]; then
    export WANDB_API_KEY=$(cat scripts/wandb_api_key.txt)
else
    echo "Warning: scripts/wandb_api_key.txt not found or not readable; skipping WANDB_API_KEY export (WANDB_MODE=$WANDB_MODE)" >&2
fi

ml ML-bundle/25.04
source venv/bin/activate

# Create directories if they don't exist
mkdir -p ./output/slurm_gpu_usage

# Monitor GPU usage
while true; do
    nvidia-smi >> ./output/slurm_gpu_usage/gpu_usage_${SLURM_JOB_ID}.txt
    sleep 10
done &



run_name="pretrain_att_maze30x30"
srun torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True