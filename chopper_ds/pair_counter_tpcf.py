import yaml
import numpy as np
import psutil
import os
import gc
from halotools.mock_observables import npairs_3d
from functools import partial

def calculate_delta_sigma(halocat, particles, params, rank, rank_iter):
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

    # also grab ncores
    ncores = config['computeinfo']['numcores']

    # set max length for buffering
    rp_max = np.max(rbins)
   
    # pull out useful stuff from data ahead of freeing
    particlemass = params[0]
    downsample_factor = params[1]
    lbox = params[2]
    outfilebase = params[3]

    #if rank == 0:
         #np.save('particles_received.npy', particles)
 
    # boost all positions by amount to avoid edge calculations
    # this avoids a double counting error

    in_subvol_halo = halocat['_inside_subvol']
    halos = np.vstack((halocat['x']+rp_max, halocat['y']+rp_max, halocat['z']+rp_max)).T
    halos_subvol = np.vstack((halocat['x'][in_subvol_halo]+rp_max, halocat['y'][in_subvol_halo]+rp_max,
                          halocat['z'][in_subvol_halo]+rp_max)).T

    in_subvol_ptcl = particles['_inside_subvol']
    ptcls = np.vstack((particles['x']+rp_max, particles['y']+rp_max, particles['z']+rp_max)).T
    ptcls_subvol = np.vstack((particles['x'][in_subvol_ptcl]+rp_max, particles['y'][in_subvol_ptcl]+rp_max,
                          particles['z'][in_subvol_ptcl]+rp_max)).T

    # generate random points of different number density
    nrand_multiply = 3

    len_halos = len(halocat['x'])
    len_halos_r = len_halos*nrand_multiply

    x_halos_r = np.random.uniform( np.min(halocat['x'])+rp_max, np.max(halocat['x'])+rp_max, len_halos_r)
    y_halos_r = np.random.uniform( np.min(halocat['y'])+rp_max, np.max(halocat['y'])+rp_max, len_halos_r)
    z_halos_r = np.random.uniform( np.min(halocat['z'])+rp_max, np.max(halocat['z'])+rp_max, len_halos_r)
    halos_r = np.vstack((x_halos_r, y_halos_r, z_halos_r)).T
    del x_halos_r, y_halos_r, z_halos_r
    gc.collect()

    len_halos_subv = len(halocat['x'][in_subvol_halo])
    len_halos_subv_r = len_halos_subv * nrand_multiply

    x_halos_r = np.random.uniform( np.min(halocat['x'][in_subvol_halo])+rp_max,
				np.max(halocat['x'][in_subvol_halo])+rp_max, len_halos_subv_r)
    y_halos_r = np.random.uniform( np.min(halocat['y'][in_subvol_halo])+rp_max,
				np.max(halocat['y'][in_subvol_halo])+rp_max, len_halos_subv_r)
    z_halos_r = np.random.uniform( np.min(halocat['z'][in_subvol_halo])+rp_max,
				np.max(halocat['z'][in_subvol_halo])+rp_max, len_halos_subv_r)
    halos_subvol_r = np.vstack((x_halos_r, y_halos_r, z_halos_r)).T
    del x_halos_r, y_halos_r, z_halos_r
    gc.collect()

    len_ptcls = len(particles['x'])
    len_ptcls_r = len_ptcls * nrand_multiply
    x_ptcls_r = np.random.uniform( np.min(particles['x'])+rp_max, np.max(particles['x'])+rp_max, len_ptcls_r)
    y_ptcls_r = np.random.uniform( np.min(particles['y'])+rp_max, np.max(particles['y'])+rp_max, len_ptcls_r)
    z_ptcls_r = np.random.uniform( np.min(particles['z'])+rp_max, np.max(particles['z'])+rp_max, len_ptcls_r)
    ptcls_r = np.vstack((x_ptcls_r, y_ptcls_r, z_ptcls_r)).T
    del x_ptcls_r, y_ptcls_r, z_ptcls_r
    gc.collect()

    len_ptcls_subv = len(particles['x'][in_subvol_ptcl])
    len_ptcls_subv_r = len_ptcls_subv*nrand_multiply
    x_ptcls_r = np.random.uniform( np.min(particles['x'][in_subvol_ptcl])+rp_max,
				np.max(particles['x'][in_subvol_ptcl])+rp_max, len_ptcls_subv_r)
    y_ptcls_r = np.random.uniform( np.min(particles['y'][in_subvol_ptcl])+rp_max,
				np.max(particles['y'][in_subvol_ptcl])+rp_max, len_ptcls_subv_r)
    z_ptcls_r = np.random.uniform( np.min(particles['z'][in_subvol_ptcl])+rp_max,
				np.max(particles['z'][in_subvol_ptcl])+rp_max, len_ptcls_subv_r)
    ptcls_subvol_r = np.vstack((x_ptcls_r, y_ptcls_r, z_ptcls_r)).T
    del x_ptcls_r, y_ptcls_r, z_ptcls_r
    gc.collect()

    # calculate pair counts for each pairing of objects / randoms needed for
    # correlation functions
    upper_lim = np.max(ptcls[:,0])+5 # upper limit for boosted box

    # first halos!
    hh_pairs_dd = np.diff(npairs_3d(halos_subvol, halos, rbins, period=upper_lim)) / \
        (len_halos_subv * (len_halos - 1))
    hh_pairs_dr = np.diff(npairs_3d(halos_subvol, halos_r, rbins, period=upper_lim)) / \
        (len_halos_subv * len_halos_r)
    hh_pairs_rr = np.diff(npairs_3d(halos_subvol_r, halos_r, rbins, period=upper_lim)) / \
        (len_halos_subv_r * (len_halos_r - 1))

    # and then particles!
    pp_pairs_dd = np.diff(npairs_3d(ptcls_subvol, ptcls, rbins, period=upper_lim)) / \
        (len_ptcls_subv * (len_ptcls - 1))
    pp_pairs_dr = np.diff(npairs_3d(ptcls_subvol, ptcls_r, rbins, period=upper_lim)) / \
        (len_ptcls_subv * len_ptcls_r)
    pp_pairs_rr = np.diff(npairs_3d(ptcls_subvol_r, ptcls_r, rbins, period=upper_lim)) / \
        (len_ptcls_subv_r * (len_ptcls_r - 1))

    # and we'll also need some cross correlation functions
    hp_pairs_dd = np.diff(npairs_3d(halos_subvol, ptcls, rbins, period=upper_lim)) / \
        (len_halos_subv * (len_ptcls - 1))
    hp_pairs_dr = np.diff(npairs_3d(halos_subvol, ptcls_r, rbins, period=upper_lim)) / \
        (len_halos_subv * len_ptcls_r)
    hp_pairs_rd = np.diff(npairs_3d(halos_subvol_r, ptcls, rbins, period=upper_lim)) / \
        (len_halos_subv_r * len_ptcls)
    hp_pairs_rr = np.diff(npairs_3d(halos_subvol_r, ptcls_r, rbins, period=upper_lim)) / \
        (len_halos_subv_r * (len_ptcls_r - 1))

    # and the cross check
    ph_pairs_dd = np.diff(npairs_3d(ptcls_subvol, halos, rbins, period=upper_lim)) / \
        (len_ptcls_subv * (len_halos - 1))
    ph_pairs_dr = np.diff(npairs_3d(ptcls_subvol, halos_r, rbins, period=upper_lim)) / \
        (len_ptcls_subv * len_halos_r)
    ph_pairs_rd = np.diff(npairs_3d(ptcls_subvol_r, halos, rbins, period=upper_lim)) / \
        (len_ptcls_subv_r * len_halos)
    ph_pairs_rr = np.diff(npairs_3d(ptcls_subvol_r, halos_r, rbins, period=upper_lim)) / \
        (len_ptcls_subv_r * (len_halos_r -1))

    # calculate auto correlation functions
    xi_hh_ls = (hh_pairs_dd - 2*hh_pairs_dr + hh_pairs_rr) / hh_pairs_rr
    xi_pp_ls = (pp_pairs_dd - 2*pp_pairs_dr + pp_pairs_rr) / pp_pairs_rr

    # calculate cross correlation function
    xi_hp_ls = (hp_pairs_dd - hp_pairs_dr - hp_pairs_rd + hp_pairs_rr) / hp_pairs_rr
    xi_ph_ls = (ph_pairs_dd - ph_pairs_dr - ph_pairs_rd + ph_pairs_rr) / ph_pairs_rr
    gc.collect()

    # save outputs as a numpy array
    rbins_mids = (rbins[1:]+rbins[:-1])/2
    augmentedarray = np.array((rbins_mids, xi_hh_ls, xi_pp_ls, xi_hp_ls, xi_ph_ls))
    np.save('output_{}_{}.npy'.format(rank, rank_iter), augmentedarray)

    # but what if I do this more correctly now?
    xi_hh_ls_components = np.array((hh_pairs_dd, hh_pairs_dr, hh_pairs_rr))
    xi_pp_ls_components = np.array((pp_pairs_dd, pp_pairs_dr, pp_pairs_rr))
    xi_hp_ls_components = np.array((hp_pairs_dd, hp_pairs_dr, hp_pairs_rd, hp_pairs_rr))
    xi_ph_ls_components = np.array((ph_pairs_dd, ph_pairs_dr, ph_pairs_rd, ph_pairs_rr))

    return xi_hh_ls_components, xi_pp_ls_components, xi_hp_ls_components, xi_ph_ls_components
