#!/bin/bash
#SBATCH -A your_account
#SBATCH -J hera_job
#SBATCH -o %x-%j.out
#SBATCH -t 01:00:00
#SBATCH -p hera
#SBATCH -N 1
#SBATCH --ntasks-per-node=40

# Load modules (example)
# source $MODULESHOME/init/bash
# module load intel impi

# Run your application
# srun ./your_executable
