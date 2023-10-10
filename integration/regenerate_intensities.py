from mantid.simpleapi import *
import numpy as np

#import itertools

import sys, os, re

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

import parameters

from peak import PeakDictionary, PeakStatistics, PeakFitPrune

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

min_I_sig = dictionary.get('minimum-signal-to-noise-ratio')
if min_I_sig is None:
    min_I_sig = 3

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

if sg is None:
    sg = 'P 1'

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
scale_file = os.path.join(outdir, 'scale.txt')

adaptive_scale = False

if scale_factor is None and os.path.exists(scale_file):
    scale = np.loadtxt(scale_file)
elif scale_factor is None:
    scale = None
    adaptive_scale = True
else:
    scale = scale_factor

cif_file = dictionary.get('cif-file')

peak_dictionary = PeakDictionary(a, b, c, alpha, beta, gamma)

if cif_file is not None:
    peak_dictionary.load_cif(os.path.join(working_directory, cif_file))

peak_dictionary.set_satellite_info(mod_vector_1, mod_vector_2, mod_vector_3, max_order)
peak_dictionary.set_material_info(chemical_formula, z_parameter, 0)
peak_dictionary.set_scale_constant(scale_constant)
peak_dictionary.load(os.path.join(outdir, outname+'.pkl'))
peak_dictionary.apply_spherical_correction(0)

LoadIsawUB(InputWorkspace='cws', Filename=os.path.join(outdir, outname+'_cal.mat'))

peak_dictionary.save_calibration(os.path.join(outdir, outname+'_cal.nxs'))
peak_dictionary.recalculate_hkl(fname=os.path.join(outdir, 'indexing.txt'))
scale = peak_dictionary.save_hkl(os.path.join(outdir, outname+'.hkl'), min_sig_noise_ratio=min_I_sig, adaptive_scale=adaptive_scale, scale=scale)
peak_dictionary.save_reflections(os.path.join(outdir, outname+'.hkl'), min_sig_noise_ratio=min_I_sig, adaptive_scale=False, scale=scale)

if max_order == 0:
    peak_prune = PeakFitPrune(os.path.join(outdir, outname+'_norm.hkl'), sg)
    peak_prune.fit_peaks()
    peak_prune.write_intensity()

if sg is not None:
    peak_statistics = PeakStatistics(os.path.join(outdir, outname+'.int'), sg)
    peak_statistics.prune_outliers()
    peak_statistics.write_statisics()
    peak_statistics.write_intensity()

absorption_file = os.path.join(outdir, 'absorption.txt')

if chemical_formula is not None and z_parameter > 0 and sample_mass > 0:
    peak_dictionary.set_material_info(chemical_formula, z_parameter, sample_mass)
    peak_dictionary.apply_spherical_correction(vanadium_mass, fname=absorption_file)
    peak_dictionary.recalculate_hkl(fname=os.path.join(outdir, 'indexing_w_abs.txt'))
    peak_dictionary.save_hkl(os.path.join(outdir, outname+'_w_abs.hkl'), min_sig_noise_ratio=min_I_sig, adaptive_scale=False, scale=scale)
    peak_dictionary.save_reflections(os.path.join(outdir, outname+'_w_abs.hkl'), min_sig_noise_ratio=min_I_sig, adaptive_scale=False, scale=scale)

    if max_order == 0:
        peak_prune = PeakFitPrune(os.path.join(outdir, outname+'_w_abs_norm.hkl'), sg)
        peak_prune.fit_peaks()
        peak_prune.write_intensity()

    if sg is not None:
        peak_statistics = PeakStatistics(os.path.join(outdir, outname+'_w_abs.hkl'), sg)
        peak_statistics.prune_outliers()
        peak_statistics.write_statisics()
        peak_statistics.write_intensity()

peak_dictionary.save(os.path.join(outdir, outname+'.pkl'))

def wobble_scale(theta, wl, mu, alpha, a, b, c, e):

    t = np.deg2rad(theta-mu)

    beta = np.deg2rad(alpha)

    d = c*np.sqrt(1-e**2)

    x = c*np.cos(t)*np.cos(beta)-d*np.sin(t)*np.sin(beta)

    f = np.exp(-(x-b)**2/(1+a*wl)**2)

    return 1/f

wobble_file = os.path.join(outdir, 'wobble.txt')

if os.path.exists(wobble_file):

    with open(wobble_file, 'r') as f:

        mu = float(f.readline().rstrip().strip('deg').split(':')[1])
        alpha = float(f.readline().rstrip().strip('deg').split(':')[1])
        a = float(f.readline().rstrip().split(':')[1])
        b = float(f.readline().rstrip().split(':')[1])
        c = float(f.readline().rstrip().split(':')[1])
        e = float(f.readline().rstrip().split(':')[1])

    for key in peak_dictionary.peak_dict.keys():

        peaks = peak_dictionary.peak_dict.get(key)

        h, k, l, m, n, p = key

        for peak in peaks:

            if peak.get_merged_intensity() > 0:

                intens = peak.get_intensity()
                scales = peak.get_data_scale().copy()
                omegas = peak.get_omega_angles()
                lamdas = peak.get_wavelengths()

                peak.set_data_scale(scales*wobble_scale(omegas, lamdas, mu, alpha, a, b, c, e))

    peak_dictionary.save_hkl(os.path.join(outdir, outname+'_w_pre.hkl'), min_sig_noise_ratio=min_I_sig, adaptive_scale=False, scale=scale)
    peak_dictionary.save_reflections(os.path.join(outdir, outname+'_w_pre.hkl'), min_sig_noise_ratio=min_I_sig, adaptive_scale=False, scale=scale)

    if max_order == 0:
        peak_prune = PeakFitPrune(os.path.join(outdir, outname+'_w_pre_norm.hkl'), sg)
        peak_prune.fit_peaks()
        peak_prune.write_intensity()

    if sg is not None:
        peak_statistics = PeakStatistics(os.path.join(outdir, outname+'_w_pre.hkl'), sg)
        peak_statistics.prune_outliers()
        peak_statistics.write_statisics()
        peak_statistics.write_intensity()

peak_dictionary.save(os.path.join(outdir, outname+'_corr.pkl'))