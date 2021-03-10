import yaml
import numpy as np
import psutil
import os
import gc
from halotools.mock_observables import tpcf
from functools import partial

def calculate_delta_sigma(halocat, particles, params, rank, rank_iter):
    """
    Description here.
    """
    # read yaml to configure binning information
    process = psutil.Process(os.getpid())
    yamlpath = '/global/cscratch1/sd/asv13/repos/deltasigma/chopper_ds/globalconfig.yaml'
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

    # quick fix for tpcf which moves it into the box in a kludgy fashion...
    if np.max(halocat['x']) > lbox/2:
        halocat['x'] = halocat['x'] - lbox/2
    if np.max(halocat['y']) > lbox/2:
        halocat['y'] = halocat['y'] - lbox/2
    if np.max(halocat['z']) > lbox/2:
        halocat['z'] = halocat['z'] - lbox/2
    if np.max(particles['x']) > lbox/2:
        particles['x'] = particles['x'] - lbox/2
    if np.max(particles['y']) > lbox/2:
        particles['y'] = particles['y'] - lbox/2
    if np.max(particles['z']) > lbox/2:
        particles['z'] = particles['z'] - lbox/2

    # and here we are going to use halotools to do a per object mean delta sigma
    # calculation
    halos = np.vstack((halocat['x'], halocat['y'], halocat['z'])).T.astype('float32')
    ptcls = np.vstack((np.mod(particles['x'],lbox), 
                       np.mod(particles['y'],lbox),
                       np.mod(particles['z'],lbox))).T.astype('float32')

    test_small = np.sum([np.sum([x<0 for x in ptcl]) for ptcl in ptcls])
    if(test_small > 0):
        print('too small: ', np.min(ptcls))
    test_big = np.sum([np.sum([x>lbox  for x in ptcl]) for ptcl in ptcls])
    if(test_big > 0):
        print('too big: ', np.max(ptcls))

    print('reporting {} GB used before first ds.'.format(process.memory_info().rss/1024./1024./1024.))
    result = tpcf(halos, rbins, sample2=ptcls, period=lbox/2, do_cross=False, num_threads=ncores).astype('float32')
    print('reporting {} GB used after first ds.'.format(process.memory_info().rss/1024./1024./1024.))
    gc.collect()

    result = result.tolist()
    augmentedlist = []
    for i in range(0,len(result)):
        item=[]
        item.append(halocat['mass'][i])
        item.append(halocat['cnfw'][i])
        item.append(halocat['x'][i])
        item.append(halocat['y'][i])
        item.append(halocat['z'][i])
        for bincount in result[i]:
            item.append(bincount)
        augmentedlist.append(item)
    outfilepath = outfilebase + '_{}_{}.txt'.format(rank, rank_iter) 
    with open(outfilepath, 'w') as fp:
        np.savetxt(fp, np.array(augmentedlist))
    return