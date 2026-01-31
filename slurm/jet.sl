#!/bin/bash
#SBATCH -A your_account
#SBATCH -J jet_job
#SBATCH -o %x-%j.out
#SBATCH -t 01:00:00
# Leave partition empty for random assignment from sjet, vjet, xjet, kjet
# or specify a specific one like -p kjet
#SBATCH -p kjet
#SBATCH -N 1
# Tasks per node depends on the partition:
# sjet/vjet: 16, xjet: 24, kjet: 40
#SBATCH --ntasks-per-node=40

# Load modules (example)
# module load intel mvapich2

# Run your application
# mpiexec -np $SLURM_NTASKS ./your_executable
