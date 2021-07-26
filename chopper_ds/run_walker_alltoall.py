import pair_counter_tpcf as pc
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

    # read in some important stuff from the yaml file
    yamlpath = '/homes/avillarreal/repositories/deltasigma/chopper_ds/globalconfig.yaml'
    with open(yamlpath) as fp: config=yaml.safe_load(fp)
    logmass_lowcut = config['analysiscuts']['logmass_lowcut']
    logmass_highcut = config['analysiscuts']['logmass_highcut']
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
        #print('rank {} synced for {}'.format(rank, work[0]))
        outfilestart = outfilebase+'/tpcf_{}_{}'.format(
                           work[0].split('/')[-3], work[0].split('/')[-1])   
        look_for_outputs = glob.glob(outfilestart+'*')
        if look_for_outputs:
            if rank == 0:
                print('found existing outputs')
            continue
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

        print(halo_files)
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
        # let's count and compare halos read in for safety
        num_halos_on_rank = len(master_halo['x'])
        num_halos_across = comm.allgather(num_halos_on_rank)
        now = datetime.now()
        print('rank {} reading particles at {}'.format(rank, now.strftime(ts_format)))
        # now particles
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
        nptcls_on_rank = len(master_ptcl['x'])
        nptcls_read = np.sum(comm.allgather(nptcls_on_rank))
        downsample_factor = ptclcube**3 / nptcls_read

        # to set outfile name, we do the following:
        outfilestart = outfilebase+'/massenc_{}_{}'.format(
                           work[0].split('/')[-3], work[0].split('/')[-1])   
        params = [ptclmass, downsample_factor, boxsize, outfilestart]
        now = datetime.now()
        print('rank {} chopping halos at {}'.format(rank, now.strftime(ts_format)))
        # now we do the new chopper broadcasting
        halocats_for_rank = get_data_for_rank(comm, master_halo, NX, NY, NZ, LBOX, RMAX)
        # and count halos after chop for comparison purposes
        num_halos_after_chop_for_rank = 0
        for halo_subvol_id, halo_subvol_data in halocats_for_rank.items():
            halocat = halocats_for_rank[halo_subvol_id]
        now = datetime.now()
        #print('rank {} chopping {} particles at {}'.format(rank, len(master_ptcl['x']), now.strftime(ts_format)))
        if rank == 0:
            np.save('particles_before.npy', master_ptcl)
        particles_for_rank = get_data_for_rank(comm, master_ptcl, NX, NY, NZ, LBOX, RMAX)
        if rank == 0:
            np.save('particles_after.npy', particles_for_rank)
        now = datetime.now()
        #print('rank {} chopped {} particles at {}'.format(rank, (particles_for_rank), now.strftime(ts_format)))
        print('rank {} done with precalc at {}'.format(rank, now.strftime(ts_format)))
        # clean up duplicates
        del master_halo
        del master_ptcl
        gc.collect()
        rank_iter = 0
        # and now we'll loop over these
        now = datetime.now()
        print('rank {} starting evaluations at {}'.format(rank, now.strftime(ts_format)))
        #print('rank {} reporting memory use at {} GB at start of evaluations'.format(rank, process.memory_info().rss/1024./1024./1024.))
        for halo_subvol_id, halo_subvol_data in halocats_for_rank.items():
            halocat = halocats_for_rank[halo_subvol_id]
            try:
                particles = particles_for_rank[halo_subvol_id]
                # small change here. We're now collecting arrays.
                hh_stuff, pp_stuff, hp_stuff, ph_stuff = pc.calculate_delta_sigma(halocat, particles, params, rank, rank_iter)
                rank_iter = rank_iter + 1
                now = datetime.now()
            except KeyError:
                print('no matching particle subvol found')
            gc.collect()
    now = datetime.now()
    # small change here: we're now 
    print('rank {} completed all pair counts at {}. '.format(rank, now.strftime(ts_format)))
    # now we'll need to gather all of those stuff.
    comm.Barrier()
    hh_gather = np.array(comm.gather(hh_stuff, root=0))
    pp_gather = np.array(comm.gather(pp_stuff, root=0))
    ph_gather = np.array(comm.gather(ph_stuff, root=0))
    hp_gather = np.array(comm.gather(hp_stuff, root=0))

    if rank == 0:
        # first we gather all the halo statistics and calculate xi
        hh_gather = np.array(hh_gather)
        hh_gather_dd = np.sum(np.array(hh_gather[:,0]),axis=0)
        hh_gather_dr = np.sum(np.array(hh_gather[:,1]),axis=0)
        hh_gather_rr = np.sum(np.array(hh_gather[:,2]),axis=0)

        xi_hh = (hh_gather_dd - 2*hh_gather_dr + hh_gather_rr) / hh_gather_rr

        # ditto for the cross correlation

        hp_gather = np.array(hp_gather)
        hp_gather_dd = np.sum(np.array(hp_gather[:,0]),axis=0)
        hp_gather_dr = np.sum(np.array(hp_gather[:,1]),axis=0)
        hp_gather_rd = np.sum(np.array(hp_gather[:,2]),axis=0)
        hp_gather_rr = np.sum(np.array(hp_gather[:,3]),axis=0)

        xi_hp = (hp_gather_dd - hp_gather_dr - hp_gather_rd + hp_gather_rr)/hp_gather_rr

        ph_gather = np.array(ph_gather)
        ph_gather_dd = np.sum(np.array(ph_gather[:,0]),axis=0)
        ph_gather_dr = np.sum(np.array(ph_gather[:,1]),axis=0)
        ph_gather_rd = np.sum(np.array(ph_gather[:,2]),axis=0)
        ph_gather_rr = np.sum(np.array(ph_gather[:,3]),axis=0)

        xi_ph = (ph_gather_dd - ph_gather_dr - ph_gather_rd + ph_gather_rr)/ph_gather_rr
 
        # and the matter autocorrelation

        pp_gather = np.array(pp_gather)
        pp_gather_dd = np.sum(np.array(pp_gather[:,0]),axis=0)
        pp_gather_dr = np.sum(np.array(pp_gather[:,1]),axis=0)
        pp_gather_rr = np.sum(np.array(pp_gather[:,2]),axis=0)

        xi_pp = (pp_gather_dd - 2*pp_gather_dr + pp_gather_rr)/pp_gather_rr

        print('halo-halo tpcf is {}'.format(xi_hh))
        print('halo-matter tpcf is {}'.format(xi_hp))
        print('matter-halo tpcf is {}'.format(xi_ph))
        print('diff between the two {}'.format(xi_hp - xi_ph))
        print('matter-matter tpcf is {}'.format(xi_pp))

        np.save('{}.npy'.format(outfilestart), np.array((xi_hh, xi_hp, xi_ph, xi_pp)))
