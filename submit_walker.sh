#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=27
#SBATCH --tasks-per-node=1
#SBATCH --job-name=paircounts
#SBATCH --output=M002-midsubsample-%j.out

cd /homes/avillarreal/repositories/deltasigma
source /homes/avillarreal/miniconda3/bin/activate chopper_calc

export PATH=/cosmo_tortoise/software/opt/spack/linux-centos7-x86_64/gcc-4.8.5/mpich-3.2.1-wbys3nhdqoqhlv55u2oos7kn7k7ghjbs/bin:$PATH

mpiexec python -u /homes/avillarreal/repositories/deltasigma/chopper_ds/run_walker_alltoall.py 0 /homes/avillarreal/repositories/deltasigma/M010_worklist.json both ${1} ${2}
