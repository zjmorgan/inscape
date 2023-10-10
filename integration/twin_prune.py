from mantid.simpleapi import *
import numpy as np

#import itertools

import sys, os, re

sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python310.zip')
sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python3.10')
sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python3.10/lib-dynload')
sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python3.10/site-packages')

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

import parameters

from peak import PeakDictionary, PeakFitPrune

from mantid.kernel import V3D

import scipy.spatial

_, filename, *ub_twin_list = sys.argv

#filename, *ub_twin_list = '/SNS/CORELLI/IPTS-28114/shared/integration/Cr5Te8_170K_Hex_0914_int.inp', '/SNS/CORELLI/IPTS-28114/shared/scripts/Cr5Te8_170K_Hex_224116_224235_D2.mat'

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
dbgdir = os.path.join(outdir, 'debug')

mod_vector_1 = dictionary['modulation-vector-1']
mod_vector_2 = dictionary['modulation-vector-2']
mod_vector_3 = dictionary['modulation-vector-3']
max_order = dictionary['max-order']
cross_terms = dictionary['cross-terms']

reflection_condition = dictionary.get('reflection-condition')
centering = dictionary.get('centering')

group = dictionary.get('group')

if np.any([key in ['P', 'Primitive'] for key in [centering, reflection_condition]]):
    reflection_condition = 'Primitive'    
    centering = 'P'
elif np.any([key in ['F', 'All-face centred'] for key in [centering, reflection_condition]]):
    reflection_condition = 'All-face centred'
    centering = 'F'
elif np.any([key in ['I', 'Body centred'] for key in [centering, reflection_condition]]):
    reflection_condition = 'Body centred'
    centering = 'I'
elif np.any([key in ['A', 'A-face centred'] for key in [centering, reflection_condition]]):
    reflection_condition = 'A-face centred'
    centering = 'A'
elif np.any([key in ['B', 'B-face centred'] for key in [centering, reflection_condition]]):
    reflection_condition = 'B-face centred'
    centering = 'B'
elif np.any([key in ['C', 'C-face centred'] for key in [centering, reflection_condition]]):
    reflection_condition = 'C-face centred'
    centering = 'C'
elif np.any([key in ['R', 'Robv', 'Rhombohedrally centred, obverse'] for key in [centering, reflection_condition]]):
    reflection_condition = 'Rhombohedrally centred, obverse'
    centering = 'R'
elif np.any([key in ['Rrev', 'Rhombohedrally centred, reverse'] for key in [centering, reflection_condition]]):
    reflection_condition = 'Rhombohedrally centred, reverse'
    centering = 'R'
elif np.any([key in ['H', 'Hexagonally centred, reverse'] for key in [centering, reflection_condition]]):
    reflection_condition = 'Hexagonally centred, reverse'
    centering = 'H'

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
peak_dictionary.load(os.path.join(outdir, outname+'.pkl'))
peak_dictionary.apply_spherical_correction(0)
peak_dictionary.clear_peaks()
peak_dictionary.repopulate_workspaces()

LoadIsawUB(InputWorkspace='cws', Filename=os.path.join(outdir, outname+'_cal.mat'))

peak_dictionary.save_calibration(os.path.join(outdir, outname+'_cal.nxs'))
peak_dictionary.recalculate_hkl(fname=os.path.join(outdir, 'indexing.txt'))

for pn in range(mtd['iws'].getNumberPeaks()-1,-1,-1):

    pk = mtd['cws'].getPeak(pn)

    h, k, l = pk.getIntHKL()
    m, n, p = pk.getIntMNP()

    if (np.array([h,k,l,m,n,p]) == 0).all():
        mtd['iws'].removePeak(pn)
        mtd['cws'].removePeak(pn)

d = mtd['cws'].column(6)

min_d_spacing = np.min(d)*0.95
max_d_spacing = np.max(d)*1.05

ub_twin_list = [os.path.join(directory, ub_twin) for ub_twin in ub_twin_list]

# result = tree.query_ball_point([2, -2, 10], 0.3)
# print(result)

CloneWorkspace(InputWorkspace='iws', OutputWorkspace='ref')
ClearUB(Workspace='ref')
LoadIsawUB(InputWorkspace='ref', Filename=ub_file)

UB = mtd['ref'].sample().getOrientedLattice().getUB()

for i, ub_twin in enumerate(ub_twin_list):
    CloneWorkspace(InputWorkspace='ref', OutputWorkspace='iws{}'.format(i))
    ClearUB(Workspace='iws{}'.format(i))
    LoadIsawUB(InputWorkspace='iws{}'.format(i), Filename=ub_twin)
    #IndexPeaks(PeaksWorkspace='iws{}'.format(i), Tolerance=0.12)
    PredictPeaks(InputWorkspace='iws{}'.format(i),
                 MinDSpacing=min_d_spacing,
                 MaxDSpacing=max_d_spacing,
                 OutputType='LeanElasticPeak',
                 CalculateWavelength=False,
                 ReflectionCondition=reflection_condition if reflection_condition is not None else 'Primitive',
                 OutputWorkspace='iws{}'.format(i))

    Qx, Qy, Qz = np.array(mtd['iws{}'.format(i)].column(12)).T
    points = np.c_[Qx, Qy, Qz]

    tree = scipy.spatial.cKDTree(points)

    for pn in range(mtd['iws'].getNumberPeaks()-1,-1,-1):

        pk = mtd['iws'].getPeak(pn)

        h, k, l = pk.getIntHKL()
        m, n, p = pk.getIntMNP()

        dh, dk, dl = m*np.array(mod_vector_1)+n*np.array(mod_vector_2)+p*np.array(mod_vector_3)       
        Q = 2*np.pi*np.dot(UB,[h+dh,k+dk,l+dl])
        n = Q/np.linalg.norm(Q)

        results = tree.query_ball_point(Q, 0.35)

        for result in results:
            if np.abs(np.dot(n, points[result]-Q)) < 0.2:
                mtd['iws'].removePeak(pn)

peak_dictionary.save_hkl(os.path.join(outdir, outname+'_twin.hkl'), adaptive_scale=False, scale=scale)
peak_dictionary.save_reflections(os.path.join(outdir, outname+'_twin.hkl'), adaptive_scale=False, scale=scale)

if max_order == 0:
    peak_prune = PeakFitPrune(os.path.join(outdir, outname+'_twin_norm.hkl'))
    peak_prune.fit_peaks()
    peak_prune.write_intensity()