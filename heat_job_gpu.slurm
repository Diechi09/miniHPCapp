#!/bin/bash
#SBATCH --job-name=heat_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=heat_gpu.out

module load gcc/9.3.0
module load cuda/11.0
module load openmpi/4.0.3

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun ./heat_gpu
