#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=plgsoftmaxvit-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=./output/slurm_output/%j.out
#SBATCH --error=./output/slurm_error/%j.out
 

PROJECT_PATH=$SLURM_SUBMIT_DIR
cd $PROJECT_PATH

echo "Current directory: $(pwd)"

# IMPORTANT: load the modules for machine learning tasks and libraries
ml ML-bundle/25.04
 
# create and activate the virtual environment
python -m venv  venv/
source venv/bin/activate
 
# install one of torch versions available at Helios wheel repo
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 # install torch based on your cuda version
pip install -r requirements.txt # install requirements
pip install --no-cache-dir --no-build-isolation adam-atan2 

export WANDB_API_KEY=$(cat scripts/wandb_api_key.txt)
wandb login 

echo "Environment setup completed successfully!"
