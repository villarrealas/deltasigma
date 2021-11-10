import pair_counter_master as pc
import data_handler as dh
import numpy as np
from thechopper.data_chopper import get_data_for_rank
import yaml
import sys
import glob
import json
from mpi4py import MPI
import psutil
import os
import gc
from datetime import datetime
import pickle

def repack_array(input_array, rank):
    '''quick helper function to repackage gathered arrays'''
    if rank == 0:
        input_out = input_array[0]
        input_extend = input_array[1:]
        for input_arr in input_extend:
            input_out = np.append(input_out, input_arr, 0)
    else:
        input_out = None
    return input_out

# This script runs
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    myhost = MPI.Get_processor_name()
    # check how many processors we have
    print('we have {} cpus on {}'.format(psutil.cpu_count(),myhost))
    process = psutil.Process(os.getpid())
    # hardcoded here for ease of changing
    seednum = int(sys.argv[1])
    fp_worklist = sys.argv[2]
    ptcl_flag = sys.argv[3]
    if ptcl_flag != 'halos' and ptcl_flag != 'ptcls' and ptcl_flag != 'both':
        print('error')
        sys.exit()
    logmass_lowcut = float(sys.argv[4])
    logmass_highcut = float(sys.argv[5])
    # read in some important stuff from the yaml file
    yamlpath = '/homes/avillarreal/repositories/deltasigma/chopper_ds/globalconfig.yaml'
    with open(yamlpath) as fp: config=yaml.safe_load(fp)
    NX = config['setupinfo']['nx']
    NY = config['setupinfo']['ny']
    NZ = config['setupinfo']['nz']
    RMAX = 10**config['setupinfo']['rbin_loghigh']
    LBOX = config['setupinfo']['lbox']
    outfilebase = config['outputinfo']['outfilebase']

    ts_format = "%H:%M:%S"

    # so what we need here is a loop. First we read the worklist on everything.
    # worklist of json format: contains both file locations + cosmology information
    with open(fp_worklist) as fp:
        worklist = json.load(fp)

    # Then, we loop over the worklist. Each element is a model AND snap to work on.
    for work in worklist:
        comm.Barrier()
        outfilestart = outfilebase+'/outputs_{}'.format(
                           work[0].split('/')[6])   
        # grab a few of these parrarmeters for ease
        boxsize = work[2]
        ptclcube = work[3]
        littleh = work[4] 

        # first every rank erads the list of files and sorts it
        halo_files = sorted(glob.glob(work[0]+'#*'))
        particle_files = sorted(glob.glob(work[1]+'#*'))

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

        now = datetime.now()
        print('rank {} reading halos at {}.'.format(rank, now.strftime(ts_format)))
        for halo_file in halo_files:
            if (my_high_halofile > split_counter >= my_low_halofile):
                halo_data, ptclmass = dh.load_data_halos(halo_file, littleh)
                mask = (logmass_lowcut < np.log10(halo_data['mass'])) & (np.log10(halo_data['mass']) <= logmass_highcut)
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
        # let's count and compare halos read in for safety
        num_halos_on_rank = len(master_halo['x'])
        num_halos_across = comm.allgather(num_halos_on_rank)
        now = datetime.now()
        print('rank {} reading particles at {}'.format(rank, now.strftime(ts_format)))
        # now particles
        if ptcl_flag == 'halos':
            master_ptcl = None
        else:
            master_ptcl = dict()
            split_counter = 0
            for ptcl_file in particle_files:
                if (my_high_ptclfile > split_counter >= my_low_ptclfile):
                    ptcl_data = dh.load_data_ptcls(ptcl_file, littleh)
                    if master_ptcl.keys():
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

        if ptcl_flag != 'halos':
            nptcls_on_rank = len(master_ptcl['x'])
            nptcls_read = np.sum(comm.allgather(nptcls_on_rank))
            downsample_factor = ptclcube**3 / nptcls_read
        else:
            downsample_factor = 0

        params = [ptclmass, downsample_factor, boxsize, outfilestart]
        now = datetime.now()
        print('rank {} chopping halos at {}'.format(rank, now.strftime(ts_format)))
        # now we do the new chopper broadcasting
        halocats_for_rank = get_data_for_rank(comm, master_halo, NX, NY, NZ, LBOX, RMAX)
        # and count halos after chop for comparison purposes
        num_halos_after_chop_for_rank = 0
        for halo_subvol_id, halo_subvol_data in halocats_for_rank.items():
            halocat = halocats_for_rank[halo_subvol_id]
        if ptcl_flag == 'halos':
            particles_for_rank = None
        else:
            particles_for_rank = get_data_for_rank(comm, master_ptcl, NX, NY, NZ, LBOX, RMAX)
            now = datetime.now()
        # clean up duplicates
        del master_halo
        del master_ptcl
        gc.collect()
        rank_iter = 0
        # and now we'll loop over these
        now = datetime.now()
        #np.save('halocat_{}.npy'.format(rank), halocats_for_rank)
        print('rank {} starting evaluations at {}'.format(rank, now.strftime(ts_format)))
        for halo_subvol_id, halo_subvol_data in halocats_for_rank.items():
            halocat = halocats_for_rank[halo_subvol_id]
            np.save('halocat_{}.npy'.format(rank), halocat)
            try:
                if ptcl_flag != 'ptcls':
                    hh_pairs_dd, hh_pairs_dd_hc, \
                        hh_pairs_dd_lc, len_halos, len_halos_subv = \
                        pc.calculate_halohalo(halocat, params, rank, rank_iter)

                if ptcl_flag != 'halos':
                    now = datetime.now()
                    print('rank {} starting ptcl evaluations at {}'.format(rank, now.strftime(ts_format)))
                    particles = particles_for_rank[halo_subvol_id]
                    pp_pairs_dd, hp_pairs_dd, \
                        len_ptcls, len_ptcls_subv, \
                        halos_subvol_mass, halos_subvol_cnfw, deltasigma = \
                        pc.calculate_haloptcl(halocat, particles, params, rank, rank_iter)

                rank_iter = rank_iter + 1
                now = datetime.now()
                print('evaluations for {} done at {}'.format(rank, now.strftime(ts_format)))
            except KeyError:
                print('no work found')
            gc.collect()
    now = datetime.now()
    # small change here: we're now 
    print('rank {} completed all pair counts at {}. '.format(rank, now.strftime(ts_format)))
    # now we'll need to gather all of those stuff.
    comm.Barrier()
    if ptcl_flag != 'ptcls':
        #hh_pairs_dd = np.sum(comm.gather(hh_pairs_dd, root=0),axis=0)
        #hh_pairs_dd_hc = np.sum(comm.gather(hh_pairs_dd_hc, root=0),axis=0)
        #hh_pairs_dd_lc = np.sum(comm.gather(hh_pairs_dd_lc, root=0),axis=0)
        #len_halos = np.sum(comm.gather(len_halos, root=0))
        #len_halos_subv = np.sum(comm.gather(len_halos_subv, root=0))

        # attempting to preserve more information for downstream error calculation
        hh_pairs_dd = comm.gather(hh_pairs_dd, root=0)
        hh_pairs_dd_hc = comm.gather(hh_pairs_dd_hc, root=0)
        hh_pairs_dd_lc = comm.gather(hh_pairs_dd_lc, root=0)
        len_halos = comm.gather(len_halos, root=0)
        len_halos_subv = comm.gather(len_halos_subv, root=0)
        if rank == 0:
            halo_array = np.array([hh_pairs_dd, hh_pairs_dd_hc, \
                hh_pairs_dd_lc, \
                len_halos, len_halos_subv], dtype=object)
            np.save('{}_{}to{}_halos.npy'.format(outfilestart, logmass_lowcut, logmass_highcut), halo_array)

    if ptcl_flag != 'halos':
        #halos_subvol_mass = repack_array(comm.gather(halos_subvol_mass, root=0), rank)
        #halos_subvol_cnfw = repack_array(comm.gather(halos_subvol_cnfw, root=0), rank)
        #pp_pairs_dd = np.sum(comm.gather(pp_pairs_dd, root=0),axis=0)
        #hp_pairs_dd = repack_array(comm.gather(hp_pairs_dd, root=0), rank)
        #deltasigma = repack_array(comm.gather(deltasigma, root=0), rank)
        #len_ptcls = np.sum(comm.gather(len_ptcls, root=0))
        #len_ptcls_subv = np.sum(comm.gather(len_ptcls_subv, root=0))
        # same jam for the ptcl section
        halos_subvol_mass = comm.gather(halos_subvol_mass, root=0)
        halos_subvol_cnfw = comm.gather(halos_subvol_cnfw, root=0)
        pp_pairs_dd = comm.gather(pp_pairs_dd, root=0)
        hp_pairs_dd = comm.gather(hp_pairs_dd, root=0)
        deltasigma = comm.gather(deltasigma, root=0)
        len_ptcls = comm.gather(len_ptcls, root=0)
        len_ptcls_subv = comm.gather(len_ptcls_subv, root=0)

        if rank == 0:
            ptcl_array = np.array([pp_pairs_dd, hp_pairs_dd, \
                len_ptcls, len_ptcls_subv, \
                halos_subvol_mass, halos_subvol_cnfw, deltasigma], dtype=object)
            np.save('{}_{}to{}_ptcls.npy'.format(outfilestart, logmass_lowcut, logmass_highcut), ptcl_array)
