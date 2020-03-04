import pair_counter as pc
import data_handler as dh
import numpy as np
from thechopper import get_buffered_subvolumes
import yaml
import sys
import json
from collections import OrderedDict
from mpi4py import MPI
import psutil
import os
import gc
from datetime import datetime

# This script runs
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    myhost = MPI.Get_processor_name()

    process = psutil.Process(os.getpid())
    # hardcoded here for ease of changing
    seednum = int(sys.argv[1])
    fp_worklist = sys.argv[2]

    # read in some important stuff from the yaml file
    yamlpath = '/homes/avillarreal/scripts/deltasigma/newscript/globalconfig.yaml'
    with open(yamlpath) as fp: config=yaml.safe_load(fp)
    logmass_lowcut = config['analysiscuts']['logmass_lowcut']
    logmass_highcut = config['analysiscuts']['logmass_highcut']
    NX = config['setupinfo']['nx']
    NY = config['setupinfo']['ny']
    NZ = config['setupinfo']['nz']
    RMAX = np.log(config['setupinfo']['rbin_loghigh'])
    LBOX = config['setupinfo']['lbox']

    # so what we need here is a loop. First we read the worklist on everything.
    # worklist of json format: contains both file locations + cosmology information
    with open(fp_worklist) as fp:
        worklist = json.load(fp)
    # Then, we loop over the worklist. Each element is a model AND snap to work on.
    for work in worklist:
        if rank == 0:
            now = datetime.now()
            print('read worklist + yam at {}'.format(now.strftime("%H:%M:%S")), flush=True)
            # rank 0 reads in the data for halos and particles.
            # these outputs are ordered dicts
            halo_data, ptcl_data, params = dh.load_data(work[0], work[1], work[2], work[3], work[4], seednum)
            now = datetime.now()
            print('done with data handler at {}'.format(now.strftime("%H:%M:%S")), flush=True)
        else:
            # every other rank just initializes a blank ordered dict.
            halo_data = OrderedDict()
            ptcl_data = OrderedDict()
            params = []
        # subsample the halo_data here
        if rank == 0:
            # read in mass config here and subsample by halo mass

            mask = (logmass_lowcut < np.log10(halo_data['mass'])) & (np.log10(halo_data['mass']) < logmass_highcut)
            for halo_key in halo_data.keys():
                halo_data[halo_key] = halo_data[halo_key][mask]
            print('read data', flush=True)   
            print('rank 0 reporting used memory: {} GB'.format(process.memory_info().rss/1024./1024./1024.), flush=True) 
            gc.collect()        
            print('rank 0 reporting used memory: {} GB'.format(process.memory_info().rss/1024./1024./1024.), flush=True) 
        # now we do the new chopper broadcasting
        halocats_for_rank, cell_ids_for_rank = get_buffered_subvolumes(
                    comm, halo_data,
                    NX, NY, NZ, LBOX, 0)
        particles_for_rank, __ = get_buffered_subvolumes(comm, ptcl_data,
                    NX, NY, NZ, LBOX, RMAX)
        if rank==0:
            now = datetime.now()
            print('broadcast data at {}.'.format(now.strftime("%H:%M:%S")))
            print('rank 0 reporting used memory: {} GB'.format(process.memory_info().rss/1024./1024./1024.), flush=True) 
            params = comm.bcast(params, root=0)
        else:
            params = None
            params = comm.bcast(params, root=0)
        # and now we'll loop over these
        rank_iter = 0
        del ptcl_data
        del halo_data
        gc.collect()
        if rank==0:
            now = datetime.now()
            print('broadcast halocats at {}.'.format(now.strftime("%H:%M:%S")))
            print('rank 0 reporting used memory: {} GB'.format(process.memory_info().rss/1024./1024./1024.), flush=True) 
        for halocat, particles in zip(halocats_for_rank, particles_for_rank):
            mask = halocat['_inside_subvol'] == True
            # modify halo catalog to only include that inside subvolume
            for halo_key in halocat.keys():
                halocat[halo_key] = halocat[halo_key][mask]
            pc.calculate_delta_sigma(halocat, particles, params, rank, rank_iter)
            rank_iter = rank_iter + 1
            print('rank 0 reporting used memory: {} GB'.format(process.memory_info().rss/1024./1024./1024.), flush=True)
    now = datetime.now()
    print('rank {} completed all work at {}. '.format(rank, now.strftime("%H:%M:%S")))
