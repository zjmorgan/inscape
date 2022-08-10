from mantid.simpleapi import *
import numpy as np

#import itertools

import sys, os, re

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

import parameters

from peak import PeakDictionary, PeakStatistics

from mantid.geometry import PointGroupFactory, SpaceGroupFactory

filename = sys.argv[1]

CreateSampleWorkspace(OutputWorkspace='sample')

dictionary = parameters.load_input_file(filename)

a = dictionary['a']
b = dictionary['b']
c = dictionary['c']
alpha = dictionary['alpha']
beta = dictionary['beta']
gamma = dictionary['gamma']

adaptive_scale = dictionary.get('adaptive-scale')
scale_factor = dictionary.get('scale-factor')

if scale_factor is None:
    scale_factor = 1

group = dictionary['group']

pgs = [pg.replace(' ', '') for pg in PointGroupFactory.getAllPointGroupSymbols()]
sgs = [sg.replace(' ', '') for sg in SpaceGroupFactory.getAllSpaceGroupSymbols()]

sg = None
pg = None

if type(group) is int:
    sg = SpaceGroupFactory.subscribedSpaceGroupSymbols(group)[0]
elif group in pgs:
    pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index(group)]).getPointGroup().getHMSymbol()
elif group in sgs:
    sg = SpaceGroupFactory.createSpaceGroup(SpaceGroupFactory.getAllSpaceGroupSymbols()[sgs.index(group)]).getHMSymbol()

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

LoadIsawUB(InputWorkspace='cws', Filename=os.path.join(directory, outname+'_cal.mat'))

peak_dictionary.recalculate_hkl()
peak_dictionary.save_hkl(os.path.join(directory, outname+'.int'), adaptive_scale=False, scale=scale)
peak_dictionary.save_reflections(os.path.join(directory, outname+'.hkl'), adaptive_scale=False, scale=scale)

if sg is not None:
    peak_statistics = PeakStatistics(os.path.join(directory, outname+'.int'), sg)
    peak_statistics.prune_outliers()
    peak_statistics.write_statisics()
    peak_statistics.write_intensity()

absorption_file = os.path.join(outdir, 'absorption.txt')

if chemical_formula is not None and z_parameter > 0 and sample_mass > 0:
    peak_dictionary.set_material_info(chemical_formula, z_parameter, sample_mass)
    peak_dictionary.apply_spherical_correction(vanadium_mass, fname=absorption_file)
    peak_dictionary.recalculate_hkl()
    peak_dictionary.save_hkl(os.path.join(directory, outname+'_w_abs.int'), adaptive_scale=False, scale=scale)
    peak_dictionary.save_reflections(os.path.join(directory, outname+'_w_abs.hkl'), adaptive_scale=False, scale=scale)

    if sg is not None:
        peak_statistics = PeakStatistics(os.path.join(directory, outname+'_w_abs.int'), sg)
        peak_statistics.prune_outliers()
        peak_statistics.write_statisics()
        peak_statistics.write_intensity()

peak_dictionary.save(os.path.join(directory, outname+'.pkl'))