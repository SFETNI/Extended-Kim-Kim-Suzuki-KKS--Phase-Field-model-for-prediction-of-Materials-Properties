#!/bin/bash
# Submission script for N
#SBATCH --partition=batch

# Lines beginning with #SBATCH are special commands to configure the job.

### Job Configuration Starts Here #############################################



# Export all current environment variables to the job (Don't change this)
#SBATCH --get-user-env
#SBATCH --output=slurm-job.%j.out   

# Submission script for Nic5
#SBATCH --time=2-00:00:00 # days-hh:mm:ss
#
#SBATCH --nodes=1
#Memory limit per compute node for the  job.  Do not use with mem-per-cpu flag.
#SBATCH --mem=64GB





#SBATCH --mail-user=....
# SBATCH --mail-type=ALL



RUNPATH=$PWD
cd $RUNPATH

# force purge av modules (to avoid possible conflicts)
module --force purge

# load appropriate modules  (NIC5)
#module load releases/2021a
#module load releases/2020a
ml releases/2021b


#module load Python/3.8.2-GCCcore-9.3.0

ml Python/3.9.6-GCCcore-11.2.0


# activate virtual environnement

virtualenv --system-site-packages ~/my_venv_pip
source ~/my_venv_pip/bin/activate



echo "Job start at $(date)" 

### Commands to run your program start here ####################################
python main_RVE_cell.py

echo "Job end at $(date)" 
