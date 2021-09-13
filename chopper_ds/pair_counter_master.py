import yaml
import numpy as np
import psutil
import os
import gc
from halotools.mock_observables.pair_counters import npairs_per_object_3d, npairs_3d
from halotools.mock_observables import mean_delta_sigma
from functools import partial

def calculate_halohalo(halocat, params, rank, rank_iter):
    """
    Description here.
    """
    # read yaml to configure binning information
    process = psutil.Process(os.getpid())
    yamlpath = '/homes/avillarreal/repositories/deltasigma/chopper_ds/globalconfig.yaml'
    with open(yamlpath) as fp: config=yaml.safe_load(fp)
    rbins_loglow = config['setupinfo']['rbin_loglow']
    rbins_loghigh = config['setupinfo']['rbin_loghigh']
    n_rbins = config['setupinfo']['n_rbins']
    rbins = np.logspace(rbins_loglow, rbins_loghigh, n_rbins+1)

    # also grab numcores
    numcores = config['computeinfo']['numcores']

    # set max length for buffering
    rp_max = np.max(rbins)
   
    # pull out useful stuff from data ahead of freeing
    particlemass = params[0]
    downsample_factor = params[1]
    lbox = params[2]
    outfilebase = params[3]

    in_subvol_halo = halocat['_inside_subvol']
    halos = np.vstack((halocat['x'], halocat['y'], halocat['z'])).T
    halos_mass = halocat['mass']
    halos_cnfw = halocat['cnfw']
    halos_subvol = np.vstack((halocat['x'][in_subvol_halo], halocat['y'][in_subvol_halo],
                          halocat['z'][in_subvol_halo])).T
    halos_subvol_mass = halocat['mass'][in_subvol_halo]
    halos_subvol_cnfw = halocat['cnfw'][in_subvol_halo]

    # generate random points of different number density
    nrand_multiply = 3

    len_halos = len(halocat['x'])
    len_halos_r = len_halos*nrand_multiply

    x_halos_r = np.random.uniform( np.min(halocat['x']), np.max(halocat['x']), len_halos_r)
    y_halos_r = np.random.uniform( np.min(halocat['y']), np.max(halocat['y']), len_halos_r)
    z_halos_r = np.random.uniform( np.min(halocat['z']), np.max(halocat['z']), len_halos_r)
    halos_r = np.vstack((x_halos_r, y_halos_r, z_halos_r)).T
    del x_halos_r, y_halos_r, z_halos_r
    gc.collect()

    len_halos_subv = len(halocat['x'][in_subvol_halo])
    len_halos_subv_r = len_halos_subv * nrand_multiply

    x_halos_r = np.random.uniform( np.min(halocat['x'][in_subvol_halo]),
				np.max(halocat['x'][in_subvol_halo]), len_halos_subv_r)
    y_halos_r = np.random.uniform( np.min(halocat['y'][in_subvol_halo]),
				np.max(halocat['y'][in_subvol_halo]), len_halos_subv_r)
    z_halos_r = np.random.uniform( np.min(halocat['z'][in_subvol_halo]),
				np.max(halocat['z'][in_subvol_halo]), len_halos_subv_r)
    halos_subvol_r = np.vstack((x_halos_r, y_halos_r, z_halos_r)).T
    del x_halos_r, y_halos_r, z_halos_r
    gc.collect()

    # calculate pair counts for each pairing of objects / randoms needed for
    # correlation functions
    upper_lim = np.max(halos[:,0])+10 # upper limit for boosted box

    #first halos
    hh_pairs_dd = np.diff(npairs_per_object_3d(halos_subvol, halos, rbins, period=None))
#        (len_halos_subv * (len_halos - 1)) 
    hh_pairs_dr = np.diff(npairs_per_object_3d(halos_subvol, halos_r, rbins, period=None))
#        (len_halos_subv * (len_halos_r))
    hh_pairs_rr = np.diff(npairs_per_object_3d(halos_subvol_r, halos_r, rbins, period=None))
#        (len_halos_subv_r * (len_halos_r - 1))
    if rank == 0:
        np.save('hh_pairs_rank.npy', [hh_pairs_dd,hh_pairs_dr,hh_pairs_rr, len_halos, len_halos_subv,
            len_halos_r, len_halos_subv_r])
    # we will also probably want these results for low concnetration and high concentration samples
    halos_subvol_hc = halos_subvol[halos_subvol_cnfw >= np.percentile(halos_subvol_cnfw,80)]
    halos_hc = halos[halos_cnfw >= np.percentile(halos_subvol_cnfw,80)]
    hh_pairs_dd_hc = np.diff(npairs_per_object_3d(halos_subvol_hc, halos_hc, rbins, period=None))
    hh_pairs_dr_hc = np.diff(npairs_per_object_3d(halos_subvol_hc, halos_r, rbins, period=None))
    hh_pairs_rr_hc = np.diff(npairs_per_object_3d(halos_subvol_r, halos_r, rbins, period=None))

    halos_subvol_lc = halos_subvol[halos_subvol_cnfw < np.percentile(halos_subvol_cnfw,20)]
    halos_lc = halos[halos_cnfw < np.percentile(halos_subvol_cnfw,20)]
    temp = np.percentile(halos_subvol_cnfw,20)

    hh_pairs_dd_lc = np.diff(npairs_per_object_3d(halos_subvol_lc, halos_lc, rbins, period=None))
    hh_pairs_dr_lc = np.diff(npairs_per_object_3d(halos_subvol_lc, halos_r, rbins, period=None))
    hh_pairs_rr_lc = np.diff(npairs_per_object_3d(halos_subvol_r, halos_r, rbins, period=None))

    return hh_pairs_dd, hh_pairs_dr, hh_pairs_rr, hh_pairs_dd_hc, hh_pairs_dr_hc, hh_pairs_rr_hc, \
        hh_pairs_dd_lc, hh_pairs_dr_lc, hh_pairs_rr_lc, len_halos, len_halos_subv, \
        len_halos_r, len_halos_subv_r

def calculate_haloptcl(halocat, particles, params, rank, rank_iter):
    """
    Description here.
    """
    # read yaml to configure binning information
    process = psutil.Process(os.getpid())
    yamlpath = '/homes/avillarreal/repositories/deltasigma/chopper_ds/globalconfig.yaml'
    with open(yamlpath) as fp: config=yaml.safe_load(fp)
    rbins_loglow = config['setupinfo']['rbin_loglow']
    rbins_loghigh = config['setupinfo']['rbin_loghigh']
    n_rbins = config['setupinfo']['n_rbins']
    rbins = np.logspace(rbins_loglow, rbins_loghigh, n_rbins+1)

    # also grab numcores
    numcores = config['computeinfo']['numcores']
    # set max length for buffering
    rp_max = np.max(rbins)
   
    # pull out useful stuff from data ahead of freeing
    particlemass = params[0]
    downsample_factor = params[1]
    lbox = params[2]
    outfilebase = params[3]

    in_subvol_halo = halocat['_inside_subvol']
    halos = np.vstack((halocat['x'], halocat['y'], halocat['z'])).T
    halos_subvol = np.vstack((halocat['x'][in_subvol_halo], halocat['y'][in_subvol_halo],
                          halocat['z'][in_subvol_halo])).T
    halos_subvol_mass = halocat['mass'][in_subvol_halo]
    halos_subvol_cnfw = halocat['cnfw'][in_subvol_halo]

    in_subvol_ptcl = particles['_inside_subvol']
    ptcls = np.vstack((particles['x'], particles['y'], particles['z'])).T
    ptcls_subvol = np.vstack((particles['x'][in_subvol_ptcl], particles['y'][in_subvol_ptcl],
                          particles['z'][in_subvol_ptcl])).T

    # generate random points of different number density
    nrand_multiply = 3

    len_halos = len(halocat['x'])
    len_halos_r = len_halos*nrand_multiply

    x_halos_r = np.random.uniform( np.min(halocat['x']), np.max(halocat['x']), len_halos_r)
    y_halos_r = np.random.uniform( np.min(halocat['y']), np.max(halocat['y']), len_halos_r)
    z_halos_r = np.random.uniform( np.min(halocat['z']), np.max(halocat['z']), len_halos_r)
    halos_r = np.vstack((x_halos_r, y_halos_r, z_halos_r)).T
    del x_halos_r, y_halos_r, z_halos_r
    gc.collect()

    len_halos_subv = len(halocat['x'][in_subvol_halo])
    len_halos_subv_r = len_halos_subv * nrand_multiply

    x_halos_r = np.random.uniform( np.min(halocat['x'][in_subvol_halo]),
				np.max(halocat['x'][in_subvol_halo]), len_halos_subv_r)
    y_halos_r = np.random.uniform( np.min(halocat['y'][in_subvol_halo]),
				np.max(halocat['y'][in_subvol_halo]), len_halos_subv_r)
    z_halos_r = np.random.uniform( np.min(halocat['z'][in_subvol_halo]),
				np.max(halocat['z'][in_subvol_halo]), len_halos_subv_r)
    halos_subvol_r = np.vstack((x_halos_r, y_halos_r, z_halos_r)).T
    del x_halos_r, y_halos_r, z_halos_r
    gc.collect()

    len_ptcls = len(particles['x'])
    len_ptcls_r = len_ptcls * nrand_multiply
    x_ptcls_r = np.random.uniform( np.min(particles['x']), np.max(particles['x']), len_ptcls_r)
    y_ptcls_r = np.random.uniform( np.min(particles['y']), np.max(particles['y']), len_ptcls_r)
    z_ptcls_r = np.random.uniform( np.min(particles['z']), np.max(particles['z']), len_ptcls_r)
    ptcls_r = np.vstack((x_ptcls_r, y_ptcls_r, z_ptcls_r)).T
    del x_ptcls_r, y_ptcls_r, z_ptcls_r
    gc.collect()

    len_ptcls_subv = len(particles['x'][in_subvol_ptcl])
    len_ptcls_subv_r = len_ptcls_subv*nrand_multiply
    x_ptcls_r = np.random.uniform( np.min(particles['x'][in_subvol_ptcl]),
				np.max(particles['x'][in_subvol_ptcl]), len_ptcls_subv_r)
    y_ptcls_r = np.random.uniform( np.min(particles['y'][in_subvol_ptcl]),
				np.max(particles['y'][in_subvol_ptcl]), len_ptcls_subv_r)
    z_ptcls_r = np.random.uniform( np.min(particles['z'][in_subvol_ptcl]),
				np.max(particles['z'][in_subvol_ptcl]), len_ptcls_subv_r)
    ptcls_subvol_r = np.vstack((x_ptcls_r, y_ptcls_r, z_ptcls_r)).T
    del x_ptcls_r, y_ptcls_r, z_ptcls_r
    gc.collect()

    # calculate pair counts for each pairing of objects / randoms needed for
    # correlation functions
    upper_lim = np.max(ptcls[:,0])+5 # upper limit for boosted box

    # and then particles!
    pp_pairs_dd = np.diff(npairs_3d(ptcls_subvol, ptcls, rbins, period=None))
    pp_pairs_dr = np.diff(npairs_3d(ptcls_subvol, ptcls_r, rbins, period=None))
    pp_pairs_rr = np.diff(npairs_3d(ptcls_subvol_r, ptcls_r, rbins, period=None))

    # and we'll also need some cross correlation functions
    hp_pairs_dd = np.diff(npairs_per_object_3d(halos_subvol, ptcls, rbins, period=None))
    hp_pairs_dr = np.diff(npairs_per_object_3d(halos_subvol, ptcls_r, rbins, period=None))
    hp_pairs_rd = np.diff(npairs_per_object_3d(halos_subvol_r, ptcls, rbins, period=None))
    hp_pairs_rr = np.diff(npairs_per_object_3d(halos_subvol_r, ptcls_r, rbins, period=None))

    if rank ==0:
        print('starting delta sigma calcs')
    # and finally let us do delta sigma per object
    delta_sigma = mean_delta_sigma(halos_subvol, ptcls, particlemass*downsample_factor,
        rbins, period=None, num_threads=numcores, per_object=True)
    # and let's do some quick moving around...
    halos_subvol = np.vstack((halos_subvol[:,0], halos_subvol[:,2], halos_subvol[:,1])).T
    ptcls = np.vstack((ptcls[:,0], ptcls[:,2], ptcls[:,1])).T
    
    if rank == 0:
        print('first ds done')
    delta_sigma = delta_sigma + mean_delta_sigma(halos_subvol, ptcls, particlemass*downsample_factor,
        rbins, period=None, num_threads=numcores, per_object=True)
    
    if rank == 0:
        print('second ds done')
    halos_subvol = np.vstack((halos_subvol[:,1], halos_subvol[:,2], halos_subvol[:,0])).T
    ptcls = np.vstack((ptcls[:,1], ptcls[:,2], ptcls[:,0])).T
    
    delta_sigma = (delta_sigma + mean_delta_sigma(halos_subvol, ptcls, particlemass*downsample_factor,
        rbins, period=None, num_threads=numcores, per_object=True))/3
    return pp_pairs_dd, pp_pairs_dr, pp_pairs_rr, hp_pairs_dd, hp_pairs_dr, hp_pairs_rd, \
        hp_pairs_rr, len_ptcls, len_ptcls_subv, len_ptcls_r, len_ptcls_subv_r, \
        halos_subvol_mass, halos_subvol_cnfw, delta_sigma
