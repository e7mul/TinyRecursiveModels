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

nodes_array=( $( scontrol show hostname $SLURM_NODELIST ) )
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
export LOGLEVEL=INFO

ml ML-bundle/25.04
source venv/bin/activate

python dataset/build_maze_dataset.py # 1000 examples, 8 augments
