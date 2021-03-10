import sys
import yaml
import numpy as np
gio_path = '/global/cscratch1/sd/asv13/repos/genericio/python'
_ = sys.path.insert(0, gio_path)
import genericio
import gc
import psutil
import os

def load_data_halos(halopath, littleh):
    halocat = dict()
    halocat['x'] = (genericio.read(halopath, 'sod_halo_mean_x')/littleh).astype('float32')
    halocat['y'] = (genericio.read(halopath, 'sod_halo_mean_y')/littleh).astype('float32')
    halocat['z'] = (genericio.read(halopath, 'sod_halo_mean_z')/littleh).astype('float32')
    nptcl_halo = genericio.read(halopath, 'sod_halo_count')
    halocat['mass'] = genericio.read(halopath, 'sod_halo_mass')/littleh
    halocat['cnfw'] = genericio.read(halopath, 'sod_halo_cdelta').astype('float32')
    particlemass = np.average(halocat['mass'] / nptcl_halo)
    del nptcl_halo
    gc.collect()
    return halocat, particlemass

def load_data_ptcls(ptclpath, littleh):
    particles = dict()
    particles['x'] = (genericio.read(ptclpath, 'x')[0]/littleh).astype('float32')
    gc.collect()
    particles['y'] = (genericio.read(ptclpath, 'y')[0]/littleh).astype('float32')
    gc.collect()
    particles['z'] = (genericio.read(ptclpath, 'z')[0]/littleh).astype('float32')
    gc.collect()
    return particles

def load_data(halopath, ptclpath, boxsize, ptclcube, littleh, seednum):
    nptcl_full = ptclcube**3

    print('reading particles', flush=True)
    # read particles into OrderedDict
    process = psutil.Process(os.getpid())
    print('rank 0 reporting used memory: {} GB'.format(process.memory_info().rss/1024./1024./1024.), flush=True) 
    particles = OrderedDict()
    particles['x'] = (genericio.read(ptclpath, 'x')[0]/littleh).astype('float32')
    gc.collect()
    particles['y'] = (genericio.read(ptclpath, 'y')[0]/littleh).astype('float32')
    gc.collect()
    particles['z'] = (genericio.read(ptclpath, 'z')[0]/littleh).astype('float32')
    gc.collect()
    print('rank 0 reporting used memory: {} GB'.format(process.memory_info().rss/1024./1024./1024.), flush=True) 
    nptcl = len(particles['x'])
    gc.collect()
    print('nptcl is {}'.format(nptcl), flush=True)
    downsample_factor = nptcl_full / nptcl
    print('read particles. Moving onto halos', flush=True)

    # read halos into ordered dict
    halocat = OrderedDict()
    halocat['x'] = (genericio.read(halopath, 'sod_halo_mean_x')/littleh).astype('float32')
    halocat['y'] = (genericio.read(halopath, 'sod_halo_mean_y')/littleh).astype('float32')
    halocat['z'] = (genericio.read(halopath, 'sod_halo_mean_z')/littleh).astype('float32')
    nptcl_halo = genericio.read(halopath, 'sod_halo_count')
    halocat['mass'] = genericio.read(halopath, 'sod_halo_mass')/littleh
    halocat['cnfw'] = genericio.read(halopath, 'sod_halo_cdelta').astype('float32')
    particlemass = np.average(halocat['mass'] / nptcl_halo)
    del nptcl_halo
    print('read halos', flush=True)

    # fix halo boundaries
    #halocat['x'] = np.mod(halocat['x'], boxsize)
    #halocat['y'] = np.mod(halocat['y'], boxsize)
    #halocat['z'] = np.mod(halocat['z'], boxsize)

    # fix particle boundaries too
    #particles['x'] = np.mod(particles['x'], boxsize)
    #particles['y'] = np.mod(particles['y'], boxsize)
    #particles['z'] = np.mod(particles['z'], boxsize)

    gc.collect()

    yamlpath = '/global/cscratch1/sd/asv13/repos/deltasigma/chopper_ds/globalconfig.yaml'
    with open(yamlpath) as fp: config=yaml.safe_load(fp)
    outfilebase = config['outputinfo']['outfilebase']
    outfilestart = outfilebase+'/npairs_{}_{}_seed{}'.format(halopath.split('/')[-3],
                                                                        halopath.split('/')[-1],
                                                                        seednum)
    # now we return all of this as two ordered dicts and a parameter set.
    return halocat, particles, [particlemass, downsample_factor, boxsize, outfilestart]
