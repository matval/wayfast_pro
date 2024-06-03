#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --job-name="energy-traversability"
#SBATCH --nodes=1		# This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks-per-node=2	# This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=16
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=4
#SBATCH --threads-per-core=4
#SBATCH --mem-per-cpu=1200
#SBATCH --time=36:00:00

# might need the latest CUDA
module load opence/1.6.1
conda activate energy_traversability

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# run script from above
srun python3 traversability/train.py --cfg_file $1
