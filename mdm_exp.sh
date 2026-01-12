#!/bin/bash -l

#SBATCH --job-name=mdm1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH -A plgsoftmaxvit-gpu-gh200
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-gh200 
#SBATCH --output=./output/slurm_output/%j.out
#SBATCH --error=./output/slurm_error/%j.err
 

PROJECT_PATH=$SLURM_SUBMIT_DIR
cd $PROJECT_PATH

export WANDB_MODE=online
export LOGLEVEL=INFO
export WANDB_ENTITY=continual-rl

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

run_name="data567_net_4heads_bs128_H2_L2_learned_pos"
project_name="Mdm_dataset_10k_samples-ACT-torch"
hidden_size=512
num_heads=4
global_batch_size=128

# Note: Using python instead of torchrun for single GPU to avoid NCCL issues on GH200
# For multi-GPU, use: torchrun --nproc-per-node N pretrain.py ...
# Disable torch.compile() on GH200 as it causes crashes during compilation
srun torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm_mdm \
data_paths="[data/mdm_dataset567/]" \
evaluators="[]" \
epochs=50000 eval_interval=100 \
global_batch_size=${global_batch_size} \
lr=1e-4 puzzle_emb_lr=0 weight_decay=1.0 puzzle_emb_weight_decay=0 \
arch.L_layers=2 \
arch.hidden_size=${hidden_size} \
arch.num_heads=${num_heads} \
arch.H_cycles=2 arch.L_cycles=2 \
arch.pos_encodings=learned \
+project_name=${project_name} \
+run_name=${run_name} ema=True \
+use_commutative_augmentation=True