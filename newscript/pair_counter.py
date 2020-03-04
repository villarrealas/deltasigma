import yaml
import numpy as np
import psutil
from halotools.mock_observables import mean_delta_sigma
from functools import partial

def calculate_delta_sigma(halocat, particles, params, rank, rank_iter):
    """
    Description here.
    """
    # read yaml to configure binning information
    yamlpath = '/homes/avillarreal/scripts/deltasigma/globalconfig.yaml'
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

    # and here we are going to use halotools to do a per object mean delta sigma
    # calculation
    halos = np.vstack((halocat['x'], halocat['y'], halocat['z'])).T
    ptcls = np.vstack((np.mod(particles['x'],lbox), 
                       np.mod(particles['y'],lbox),
                       np.mod(particles['z'],lbox))).T

    test_small = np.sum([np.sum([x<0 for x in ptcl]) for ptcl in ptcls])
    if(test_small > 0):
        print('too small: ', np.min(ptcls))
    test_big = np.sum([np.sum([x>lbox  for x in ptcl]) for ptcl in ptcls])
    if(test_big > 0):
        print('too big: ', np.max(ptcls))

    result = mean_delta_sigma(halos, ptcls, particlemass, downsample_factor,
                              rbins, period=None, num_threads=ncores,
                              per_object=True) 
    if(rank==0):
        print(result)
    # repeat two more times to average
    halos = np.vstack((halocat['x'], halocat['z'], halocat['y'])).T
    ptcls = np.vstack((np.mod(particles['x'],lbox), 
                       np.mod(particles['z'],lbox),
                       np.mod(particles['y'],lbox))).T
    result = result + mean_delta_sigma(halos, ptcls, particlemass, downsample_factor,
                                       rbins, period=lbox, num_threads=ncores,
                                       per_object=True)

    halos = np.vstack((halocat['z'], halocat['y'], halocat['x'])).T
    ptcls = np.vstack((np.mod(particles['z'],lbox), 
                       np.mod(particles['y'],lbox),
                       np.mod(particles['x'],lbox))).T
    result = result + mean_delta_sigma(halos, ptcls, particlemass, downsample_factor,
                                       rbins, period=lbox, num_threads=ncores,
                                       per_object=True)

    # and average
    result = result / 3.

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
