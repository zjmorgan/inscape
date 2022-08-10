from mantid.simpleapi import *
import numpy as np

#import itertools

import sys, os, re

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

import parameters

from peak import PeakDictionary

from mantid.kernel import V3D

_, filename, *ub_twin_list = sys.argv

CreateSampleWorkspace(OutputWorkspace='sample')

dictionary = parameters.load_input_file(filename)

a = dictionary['a']
b = dictionary['b']
c = dictionary['c']
alpha = dictionary['alpha']
beta = dictionary['beta']
gamma = dictionary['gamma']

min_d = dictionary.get('minimum-d-spacing')

if min_d is None:
    min_d = 0.7

adaptive_scale = dictionary.get('adaptive-scale')
scale_factor = dictionary.get('scale-factor')

if scale_factor is None:
    scale_factor = 1

if dictionary.get('chemical-formula') is not None:
    chemical_formula = ''.join([' '+item if item.isalpha() else item for item in re.findall(r'[A-Za-z]+|\d+', dictionary['chemical-formula'])]).lstrip(' ')
else:
    chemical_formula = None

z_parameter = dictionary['z-parameter']
sample_mass = dictionary['sample-mass']
vanadium_mass = dictionary.get('vanadium-mass')

if vanadium_mass is None:
    vanadium_mass = 0

facility, instrument = parameters.set_instrument(dictionary['instrument'])
ipts = dictionary['ipts']

working_directory = '/{}/{}/IPTS-{}/shared/'.format(facility,instrument,ipts)
shared_directory = '/{}/{}/shared/'.format(facility,instrument)
    
if dictionary['ub-file'] is not None:
    ub_file = os.path.join(working_directory, dictionary['ub-file'])
    if '*' in ub_file:
        ub_file = [ub_file.replace('*', str(run)) for run in run_nos]
else:
    ub_file = None

directory = os.path.dirname(os.path.abspath(filename))
outname = dictionary['name']

outdir = os.path.join(directory, outname)

mod_vector_1 = dictionary['modulation-vector-1']
mod_vector_2 = dictionary['modulation-vector-2']
mod_vector_3 = dictionary['modulation-vector-3']
max_order = dictionary['max-order']
cross_terms = dictionary['cross-terms']

if not all([a,b,c,alpha,beta,gamma]):
    if ub_file is not None:
        if type(ub_file) is list:
            LoadIsawUB(InputWorkspace='sample', Filename=ub_file[0])
        else:
            LoadIsawUB(InputWorkspace='sample', Filename=ub_file)
        a = mtd['sample'].sample().getOrientedLattice().a()
        b = mtd['sample'].sample().getOrientedLattice().b()
        c = mtd['sample'].sample().getOrientedLattice().c()
        alpha = mtd['sample'].sample().getOrientedLattice().alpha()
        beta = mtd['sample'].sample().getOrientedLattice().beta()
        gamma = mtd['sample'].sample().getOrientedLattice().gamma()

scale_constant = 1e+4

scale = np.loadtxt(os.path.join(outdir, 'scale.txt'))

peak_dictionary = PeakDictionary(a, b, c, alpha, beta, gamma)
peak_dictionary.set_satellite_info(mod_vector_1, mod_vector_2, mod_vector_3, max_order)
peak_dictionary.set_material_info(chemical_formula, z_parameter, 0)
peak_dictionary.set_scale_constant(scale_constant)
peak_dictionary.load(os.path.join(directory, outname+'.pkl'))
peak_dictionary.apply_spherical_correction(0)
peak_dictionary.clear_peaks()
peak_dictionary.repopulate_workspaces()
    
LoadIsawUB(InputWorkspace='cws', Filename=os.path.join(directory, outname+'_cal.mat'))

peak_dictionary.recalculate_hkl()
 
ub_twin_list = [os.path.join(directory, ub_twin) for ub_twin in ub_twin_list]

for i, ub_twin in enumerate(ub_twin_list):
    CloneWorkspace(InputWorkspace='iws', OutputWorkspace='iws{}'.format(i))
    LoadIsawUB(InputWorkspace='iws{}'.format(i), Filename=ub_twin)
    IndexPeaks(PeaksWorkspace='iws{}'.format(i), Tolerance=0.12)

for pn in range(mtd['iws'].getNumberPeaks()-1,-1,-1):

    pk = mtd['iws'].getPeak(pn)

    h, k, l = pk.getIntHKL()
    m, n, p = pk.getIntMNP()

    for i, ub_twin in enumerate(ub_twin_list):
        pk_twin = mtd['iws{}'.format(i)].getPeak(pn)
        H, K, L = pk_twin.getHKL()
        if not np.allclose([H,K,L],0):
            h, k, l = 0, 0, 0
            m, n, p = 0, 0, 0

    dh, dk, dl = (m*np.array(mod_vector_1)+n*np.array(mod_vector_2)+p*np.array(mod_vector_3)).astype(float)

    pk.setHKL(h+dh,k+dk,l+dl)
    pk.setIntHKL(V3D(h,k,l))
    pk.setIntMNP(V3D(m,n,p))

for pn in range(mtd['iws'].getNumberPeaks()-1,-1,-1):

    pk = mtd['iws'].getPeak(pn)

    h, k, l = pk.getIntHKL()
    m, n, p = pk.getIntMNP()

    if (np.array([h,k,l,m,n,p]) == 0).all():
        mtd['iws'].removePeak(pn)
 
peak_dictionary.save_hkl(os.path.join(directory, outname+'_twin_prune.int'), adaptive_scale=False, scale=scale)