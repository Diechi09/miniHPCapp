# miniHPCapp

This repository contains reference implementations for the 2D heat equation assignment:

- `heat_serial.c`: Baseline serial solver.
- `heat_parallel.c`: MPI + OpenMP implementation with row-wise domain decomposition and halo exchanges.
- `heat_gpu.c`: OpenACC implementation for GPU acceleration.
- `heat_job.slurm`: Example SLURM script for multi-node CPU runs.
- `heat_job_gpu.slurm`: Example SLURM script for GPU runs.

## Building

```bash
# Serial
gcc -O3 -o heat_serial heat_serial.c -lm

# MPI + OpenMP
mpicc -O3 -fopenmp -o heat_parallel heat_parallel.c -lm

# OpenACC (compiler must support it, e.g., nvc or recent GCC with OpenACC)
mpicc -O3 -fopenmp -fopenacc -o heat_gpu heat_gpu.c -lm -acc
```

## Running on SLURM

Submit the CPU-parallel job:

```bash
sbatch heat_job.slurm
```

Submit the GPU job (on nodes with GPUs available):

```bash
sbatch heat_job_gpu.slurm
```

Both programs print convergence information and elapsed time to their respective output files.
