#!/bin/bash
#SBATCH --time=128:00:00
#SBATCH --nodes=8
#SBATCH --tasks-per-node=1
#SBATCH --exclude=cg2-p,cg3-p,cg10-p
#SBATCH --job-name=deltasigma
#SBATCH --output=M001-%j.out

cd /homes/avillarreal/scripts/deltasigma
source /homes/avillarreal/miniconda3/bin/activate deltasigma
export PATH=/cosmo_tortoise/software/opt/spack/linux-centos7-x86_64/gcc-4.8.5/mpich-3.2.1-wbys3nhdqoqhlv55u2oos7kn7k7ghjbs/bin:$PATH

mpiexec python -u /homes/avillarreal/scripts/deltasigma/newscript/run_walker.py 0 '/homes/avillarreal/scripts/deltasigma/M001_worklist.json'
