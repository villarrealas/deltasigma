import pair_counter as pc
import data_handler as dh
import numpy as np
from thechopper import get_data_for_rank
import yaml
import sys
import json
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
    yamlpath = '/global/cscratch1/sd/asv13/repos/deltasigma/chopper_ds/globalconfig.yaml'
    with open(yamlpath) as fp: config=yaml.safe_load(fp)
    logmass_lowcut = config['analysiscuts']['logmass_lowcut']
    logmass_highcut = config['analysiscuts']['logmass_highcut']
    NX = config['setupinfo']['nx']
    NY = config['setupinfo']['ny']
    NZ = config['setupinfo']['nz']
    RMAX = np.log(config['setupinfo']['rbin_loghigh'])
    LBOX = config['setupinfo']['lbox']
    outfilebase = config['outputinfo']['outfilebase']

    # so what we need here is a loop. First we read the worklist on everything.
    # worklist of json format: contains both file locations + cosmology information
    with open(fp_worklist) as fp:
        worklist = json.load(fp)

    # Then, we loop over the worklist. Each element is a model AND snap to work on.
    for work in worklist:
        # grab a few of these parrarmeters for ease
        boxsize = work[2]
        ptclcube = work[3]
        littleh = work[4] 

        # first every rank erads the list of files and sorts it
        halo_files = sorted(glob.glob(work[0]+'#'))
        particle_files = sorted(glob.glob(work[1]+'#'))
        # determine which rank reads which files
        halofile_split = len(halo_files) / size
        my_low_halofile = rank*halofile_split
        my_high_halofile = (rank+1)*halofile_split
        ptclfile_split = len(particle_files) / size
        my_low_ptclfile = rank*ptclfile_split
        my_high_ptclfile = (rank+1)*ptclfile_split

        # everyone loops over the list of halo files / particle files, but operates only on their own
        # first halos
        master_halo = dict()
        split_counter = 0
        for halo_file in halo_files:
            if (my_high_halofile > split_counter >= my_low_halofile):
                halo_data, ptclmass = dh.load_data_halos(halo_file, littleh)
                mask = (logmass_lowcut < np.log10(halo_data['mass'])) & (np.log10(halo_data['mass']) < logmass_highcut)
                for halo_key in halo_data.keys():
                    halo_data[halo_key] = halo_data[halo_key][mask]
                # update a master dict with all this
                if master_halo.keys():
                    for key in master_halo.keys():
                        master_halo[key] = np.append(master_halo[key], halo_data[key])
                else:
                    for key in halo_data.keys():
                        master_halo[key] = halo_data[key]
            split_counter += 1
        del halo_data
        gc.collect()        

        # now particles
        master_ptcl = dict()
        split_counter = 0
        for ptcl_file in ptcl_files:
            if (my_high_ptclfile > split_counter >= my_low_ptclfile):
                ptcl_data = dh.load_data_ptcls(ptcl_file, littleh)
            if master_ptcl.keys()
                for key in master_ptcl.keys():
                    master_ptcl[key] = np.append(master_ptcl[key], ptcl_data[key])
            else:
                for key in ptcl_data.keys():
                    master_ptcl[key] = ptcl_data[key]
            split_counter+=1
        del ptcl_data
        gc.collect() 

        # we will need the params for the calculation, so let's sort those out now
        # to get downsample factor, we do need to collect number of ptcls
        nptcls_on_rank = len(master_ptcl['x'])
        nptcls_read = np.sum(comm.allgather(nptcls_on_rank))
        downsample_factor = ptclcube**3 / nptcls_read

        # to set outfile name, we do the following:
        outfilestart = outfilebase+'/massenc_{}_{}'.format(
                           work[0].split('/')[-3], work[0].split('/')[-1])   
        params = [ptclmass, downsample_factor, boxsize, outfilestart]

        # now we do the new chopper broadcasting
        halocats_for_rank = get_data_for_rank(comm, master_halo, NX, NY, NZ, LBOX, 0)
        particles_for_rank = get_data_for_rank(comm, master_ptcl, NX, NY, NZ, LBOX, RMAX)

        # clean up duplicates
        del master_halo
        del master_ptcl
        gc.collect()
        rank_iter = 0
        # and now we'll loop over these
        for halo_subvol_id, halo_subvol_data in halocats_for_rank.items():
            halocat = halocats_for_rank[halo_subvol_id]
            mask = halocat['_inside_subvol'] == True
            # modify halo catalog to only include that inside subvolume
            for halo_key in halocat.keys():
                halocat[halo_key] = halocat[halo_key][mask]
            try:
                particles = particles_for_rank[halo_subvol_id]
                pc.calculate_delta_sigma(halocat, particles, params, rank, rank_iter)
                rank_iter = rank_iter + 1
            except KeyError:
                print('no matching particle subvol found')
    print('rank {} completed all work at {}. '.format(rank, now.strftime("%H:%M:%S")))
