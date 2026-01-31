#!/bin/bash
#SBATCH -M c6
#SBATCH -A your_account
#SBATCH -J gaea_c6_job
#SBATCH -o %x-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=192

# Load modules (example)
# module load PrgEnv-intel

# Run your application
# srun ./your_executable
