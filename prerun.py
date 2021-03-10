import json
import os
import glob

model = 'M010'
h0 =  0.7833
worklist = [ ['/global/project/projectdirs/hacc/MiraTitan/{}/'.format(model), 2100, 3200, h0] ]
add_to_path_halos = 'Halos/'
add_to_path_particles = 'Particles/'
newworklist = []

dir_struct = 0 # handles different structures for how halos are sorted

for workitem in worklist:
    basepath = workitem[0]
    path_to_halos = os.path.join(basepath, add_to_path_halos)
    path_to_particles = os.path.join(basepath, add_to_path_particles)
    if dir_struct == 0:
        step_paths = glob.glob(path_to_halos+'STEP*/')
    if dir_struct == 1:
        halo_files = glob.glob(path_to_halos+'*sodproperties')
        step_paths = ['STEP'+halo_file.split('.sodproperties')[0].split('-')[-1]+'/' for halo_file in halo_files]

    for step_path in step_paths:
        if dir_struct == 0:
            step_num = step_path.split('STEP')[-1].split('/')[0]
            halobase = os.path.commonprefix(glob.glob(step_path+'*.sodproperties'))
        if dir_struct == 1:
            step_num = step_path.split('STEP')[-1].split('/')[0]
            halobase = os.path.commonprefix(glob.glob(path_to_halos+'*'+step_num+'*.sodproperties'))
        stepchoice = os.path.basename(os.path.normpath(step_path))
        updated_particle_path = os.path.join(path_to_particles,stepchoice)
        particlebase = os.path.commonprefix(glob.glob(updated_particle_path+'/*.mpicosmo*'))
        if int(step_num) > 1:
            newworklist.append([halobase, particlebase, workitem[1], workitem[2], workitem[3]])
with open('{}_worklist.json'.format(model), 'w') as fp:
    json.dump(newworklist, fp)
