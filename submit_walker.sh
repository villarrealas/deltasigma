#!/bin/bash
#SBATCH --account=hacc
#SBATCH --qos=regular
#SBATCH --constraint=knl
#SBATCH --time=48:00:00
#SBATCH --nodes=8
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=272
#SBATCH --job-name=deltasigma
#SBATCH --output=M001-smol-%j.out

cd /global/cscratch1/sd/asv13/
source /global/homes/a/asv13/miniconda3/bin/activate chopperds

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=68

srun python -u /global/cscratch1/sd/asv13/repos/deltasigma/chopper_ds/run_walker_alltoall.py 0 '/global/cscratch1/sd/asv13/repos/deltasigma/M001_worklist_smol.json'
