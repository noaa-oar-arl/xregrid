#!/bin/bash
#SBATCH -M c5
#SBATCH -A your_account
#SBATCH -J gaea_c5_job
#SBATCH -o %x-%j.out
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=128

# Load modules (example)
# module load PrgEnv-intel

# Run your application
# srun ./your_executable
