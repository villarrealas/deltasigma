#!/bin/bash
#SBATCH --account=hacc
#SBATCH --qos=regular
#SBATCH --constraint=knl
#SBATCH --time=24:00:00
#SBATCH --nodes=8
#SBATCH --tasks-per-node=1
#SBATCH --job-name=deltasigma
#SBATCH --output=M001-%j.out

cd /global/cscratch1/sd/asv13/
source /global/homes/a/asv13/miniconda3/bin/activate chopperds

srun python -u /global/cscratch1/sd/asv13/repos/deltasigma/chopper_ds/run_walker.py 0 '/global/cscratch1/sd/asv13/repos/deltasigma/M001_worklist.json'
