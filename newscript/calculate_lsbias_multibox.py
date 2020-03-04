import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from scipy.stats import pearsonr
from halotools.mock_observables import delta_sigma_from_precomputed_pairs
from colossus.cosmology import cosmology
from colossus.lss import peaks

def z_step(step):
    """Takes in a step number and returns the redshift of the simulation.
    This assumes 500 steps from z=200 to z=0, with spacing in scale factor.
    """
    a_in = 1./(1.+200)
    a_fin = 1./(1.)
    delta = (a_fin-a_in)/500.
    a = a_in + delta*(step+1.)
    z = 1./a -1
    return z

def calculate_ls_bias(data, highp, lowcutoff, highcutoff, lbox, norm_method, cosmo, params, redshift):
    """Calculate the large scale bias using the enclosed masses.
    highp is the upper percentile for which we are determining the large scale bias
    lowcutoff + highcutoff are cutoffs in the appropriate normalization space
    lbox = depth of box used in the calculation
    norm_method decides how to normalize masses
    """
    mass_temp = data[:,0]
    if norm_method == 'None':
        highcutoff = 10**highcutoff
        lowcutoff = 10**lowcutoff
    elif norm_method == 'Mstar':
        highcutoff = 10**highcutoff
        lowcutoff = 10**lowcutoff
        mstar = (peaks.nonLinearMass(redshift)/params['H0']/100.)
        mass_temp /= mstar
        print(mass_temp[0:20])
    elif norm_method == 'peak_height':
        highcutoff = 10**highcutoff
        lowcutoff = 10**lowcutoff
        mass_temp = peaks.peakHeight(mass_temp*params['H0']/100., redshift)
        print(mass_temp[0:20])
    else:
        print('how did this even happen?')
        return
    mass_mask = ((mass_temp <= highcutoff) & (mass_temp > lowcutoff))
    pos_sub = np.vstack( (data[:,2][mass_mask], data[:,3][mass_mask], data[:,4][mass_mask]) ).T
    deltasigma_sub = np.array(data[:,5:][mass_mask])
    conc_sub = data[:,1][mass_mask]
    highp_mask = (conc_sub >= np.percentile(conc_sub, highp))
    nbins = 20
    rbins = np.logspace(1.4,2,nbins+1)
    rbins_alt = np.logspace(1.4,2,2)

    pos_high = pos_sub[highp_mask]
    deltasigma_high =  deltasigma_sub[highp_mask]
    meands_all = np.array( [np.mean(deltasigma_sub[:,i]) for i in range(nbins)])
    meands_high = np.array([np.mean(deltasigma_high[:,i]) for i in range(nbins)]) 
    ls_bias = meands_high / meands_all
    bootstraps = []
    for i in range(200):
        resamp = np.random.randint(0,len(deltasigma_high),len(deltasigma_high))
        meands_temp = np.array([np.mean((deltasigma_sub[resamp])[:,j]) for j in range(nbins)])
        bootstraps.append(meands_temp/meands_all)
    bootstraps = np.array(bootstraps)
    print(bootstraps[0,:])
    error_low = [np.percentile(bootstraps[i,:], 16) for i in range(nbins)]
    error_high = [np.percentile(bootstraps[i,:], 84) for i in range(nbins)]
    return ls_bias, error_low, error_high

file_path = '/cosmo/scratch/avillarreal/deltasigma/testdata_20bins_27nodes'

# Leaving out M001 for now due to breaking Colossus
list_models = ['M001']#, 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009']#, 'M010']
list_params = [{'flat': True, 'H0':61.67, 'Om0': 0.38704, 'Ob0': 0.05945, 'sigma8': 0.8778, 'ns': 0.9611, 'de_model':'w0wa', 'w0': -0.7000, 'wa':0.6722},
#               {'flat': True, 'H0':75.00, 'Om0': 0.24107, 'Ob0': 0.04139, 'sigma8': 0.8556, 'ns': 1.0500, 'de_model':'w0wa', 'w0': -1.0330, 'wa': 0.91110},
#               {'flat': True, 'H0':71.67, 'Om0': 0.30176, 'Ob0': 0.04271, 'sigma8': 0.9000, 'ns': 0.8944, 'de_model':'w0wa', 'w0': -1.1000, 'wa':-0.28330},
#               {'flat': True, 'H0':58.33, 'Om0': 0.36416, 'Ob0': 0.06710, 'sigma8': 0.7889, 'ns': 0.8722, 'de_model':'w0wa', 'w0': -1.1670, 'wa': 1.15000},
#               {'flat': True, 'H0':85.00, 'Om0': 0.19834, 'Ob0': 0.03253, 'sigma8': 0.7667, 'ns': 0.9833, 'de_model':'w0wa', 'w0': -1.2330, 'wa':-0.04445},
#               {'flat': True, 'H0':55.00, 'Om0': 0.43537, 'Ob0': 0.07107, 'sigma8': 0.8333, 'ns': 0.9167, 'de_model':'w0wa', 'w0': -0.7667, 'wa': 0.19440},
#               {'flat': True, 'H0':81.67, 'Om0': 0.22654, 'Ob0': 0.03324, 'sigma8': 0.8111, 'ns': 1.0280, 'de_model':'w0wa', 'w0': -0.8333, 'wa':-1.00000},
#               {'flat': True, 'H0':68.33, 'Om0': 0.25701, 'Ob0': 0.04939, 'sigma8': 0.7000, 'ns': 1.0060, 'de_model':'w0wa', 'w0': -0.9000, 'wa': 0.43330},
#               {'flat': True, 'H0':65.00, 'Om0': 0.32994, 'Ob0': 0.05141, 'sigma8': 0.7444, 'ns': 0.8500, 'de_model':'w0wa', 'w0': -0.9667, 'wa':-0.76110},
#               {'flat': True, 'H0':78.33, 'Om0': 0.20829, 'Ob0': 0.03649, 'sigma8': 0.7222, 'ns': 0.9389, 'de_model':'w0wa', 'w0': -1.3000, 'wa':-0.52220}
             ]

files_models = []
for model in list_models:
    files_models.append(glob.glob(os.path.join(file_path, model, 'massenc_*.txt'))) 

print(files_models) 
#uniquesteps = list(set([item.split('M001_')[-1].split('_')[0] for item in files_models[0]]))
uniquesteps = ['STEP499',]
lowcutoff = float(sys.argv[1])
highcutoff = float(sys.argv[2])
highp = int(sys.argv[3])
norm_method = str(sys.argv[4])

for step in uniquesteps:
    outfile = open('{}_{}_{}to{}_lsbias_compare_models_27nodes.txt'.format(step, norm_method, lowcutoff, highcutoff), 'w')
    for model, params in zip(list_models, list_params):
        print('analyzing {} on {}!'.format(model, step))
        stepfiles = glob.glob(os.path.join( file_path, model, '*{}*.txt'.format(step)))
        data = []
        for stepfile in stepfiles:
            data.extend(np.loadtxt(stepfile))
        data = np.array(data)
        # we now need an initial calculation of mstar for rescaling purposes
        if norm_method == 'None':
            redshift = None
            cosmo = None
        elif norm_method == 'Mstar':
            redshift = z_step(int(step[-3:]))
            cosmo = cosmology.setCosmology(model, params)
        elif norm_method == 'peak_height':
            redshift = z_step(int(step[-3:]))
            cosmo = cosmology.setCosmology(model, params)
        else:
            print('Please choose between valid norm_method strings.\n'
                   'None (standard logmass cutoffs),\n'
                   'Mstar (cutoffs by M/Mstar), \n'
                   'or peak_height (cutoffs by peak height).')
            quit()
            
        ls_bias, err_low, err_high = calculate_ls_bias(data, highp, lowcutoff, highcutoff, 2100, norm_method, cosmo, params, redshift)
        outfile.write(model)
        outfile.write(' {} {} {} {}'.format(highp, ls_bias, err_low, err_high))
        outfile.write('\n')
    outfile.close()
