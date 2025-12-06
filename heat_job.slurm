#!/bin/bash
#SBATCH --job-name=heat_parallel
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=heat_parallel.out

module load gcc/9.3.0
module load openmpi/4.0.3

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mpirun -np $((SLURM_NNODES * SLURM_NTASKS_PER_NODE)) ./heat_parallel
