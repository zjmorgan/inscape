from mantid.simpleapi import *
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from itertools import cycle

import sys, os, re

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

import parameters

from peak import PeakDictionary, PeakStatistics

from mantid.geometry import PointGroupFactory, SpaceGroupFactory, CrystalStructure

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

cif_file = dictionary.get('cif-file')

peak_dictionary = PeakDictionary(a, b, c, alpha, beta, gamma)

if cif_file is not None:
    peak_dictionary.load_cif(os.path.join(working_directory, cif_file))
        
peak_dictionary.set_satellite_info(mod_vector_1, mod_vector_2, mod_vector_3, max_order)
peak_dictionary.set_material_info(chemical_formula, z_parameter, sample_mass)
peak_dictionary.set_scale_constant(scale_constant)
peak_dictionary.load(os.path.join(directory, outname+'.pkl'))

models = ['primary', 'secondary, gaussian', 'secondary, lorentzian',
          'secondary, gaussian type I', 'secondary, gaussian type II', 
          'secondary, lorentzian type I', 'secondary, lorentzian type II']

rs, gs, Us, scales, chi_sqs = [], [], [], [], []

ext_file = open(os.path.join(outdir, 'extinction.txt'), 'w')

for model in models:

    r, g, scale, U, chi_sq = peak_dictionary.fit_extinction(model)

    ext_file.write('model: {}\n'.format(model))
    if r > 0:
        ext_file.write('crystallite size r: {:.4f} micron\n'.format(r/10000))
    if g > 0:
        ext_file.write('crystallite parameter g: {:.4f} \n'.format(g))
        if 'gaussian' in model:
            sig = np.rad2deg(1/(2*np.sqrt(np.pi)*g))
            ext_file.write('crystallite misorientation: {:.4f} deg \n'.format(sig))
        if 'lorentzian' in model:
            eta = np.rad2deg(1/(2*np.pi*g))
            ext_file.write('crystallite misorientation: {:.4f} deg \n'.format(eta))

    ext_file.write('Uiso: {:.4e} \n'.format(U))
    ext_file.write('scale: {:.4e} \n'.format(scale))
    ext_file.write('chi^2: {:.4e} \n\n'.format(chi_sq))

    rs.append(r)
    gs.append(g)
    Us.append(U)
    scales.append(scale)
    chi_sqs.append(chi_sq)

i = np.argmin(chi_sqs)
model = models[i]

r, g = rs[i], gs[i]

message = ''
lamda = 1 # Ang

if 'secondary' in model and 'type' not in model:
    if g/(r/lamda) < 0.001:
        model += ' type I'
        message = 'r >> lambda g'
    elif (r/lamda)/g < 0.001:
        model += ' type II'
        message = 'r << lambda g'

ext_file.write('model: {} {}'.format(model,message))
ext_file.close()

i = models.index(model)

r, g, s, U = rs[i], gs[i], scales[i], Us[i]

uc = peak_dictionary.cs.getUnitCell()

a, b, c, alpha, beta, gamma = uc.a(), uc.b(), uc.c(), uc.alpha(), uc.beta(), uc.gamma()

constants = '{} {} {} {} {} {}'.format(a,b,c,alpha,beta,gamma)

scatterers = peak_dictionary.cs.getScatterers()

atoms = []
for j, scatterer in enumerate(scatterers):
    elm, x, y, z, occ, _ = scatterer.split(' ')
    atoms.append(' '.join([elm,x,y,z,occ,str(U)]))

atoms = '; '.join(atoms)

peak_dictionary.cs = CrystalStructure(constants, peak_dictionary.hm, atoms)

X, Y, I, E, HKL, d_spacing = peak_dictionary.extinction_curves(r, g, s, model)

marker = ['o', 's', '<']
markers = cycle(marker)

fig, ax = plt.subplots(1, 1, num=2)

for j, (x, y, i, e, hkl, d) in enumerate(zip(X, Y, I, E, HKL, d_spacing)):

    mark = next(markers)
    sort = np.argsort(x)

    ax.errorbar(x[sort], i[sort], yerr=e[sort], linestyle='none', marker=mark, color='C{}'.format(j%9), label='({},{},{})'.format(*hkl))
    ax.plot(x[sort], y[sort], linestyle='--', color='k', zorder=100)

ax.legend()
ax.set_yscale('linear')
ax.set_xlabel(r'$x$') #
ax.set_ylabel(r'$I$ [arb. unit]')

xlim = ax.get_xlim()
ylim = ax.get_ylim()

marker = ['o', 's', '<']
markers = cycle(marker)

with PdfPages(os.path.join(outdir, 'extinction.pdf')) as pdf:

    for j, (x, y, i, e, hkl, d) in enumerate(zip(X, Y, I, E, HKL, d_spacing)):

        fig, ax = plt.subplots(1, 1, num=3)

        mark = next(markers)
        sort = np.argsort(x)

        ax.errorbar(x[sort], i[sort], yerr=e[sort], linestyle='none', marker=mark, color='C{}'.format(j%9), label='({},{},{})'.format(*hkl))
        ax.plot(x[sort], y[sort], linestyle='--', color='k', zorder=100)
        ax.set_title('d = {:.4} \u212B'.format(d))

        ax.legend()
        ax.set_yscale('linear')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r'$x$') #
        ax.set_ylabel(r'$I$ [arb. unit]')

        pdf.savefig()
        plt.close()

peak_dictionary.apply_extinction_correction(r, g, s, model=model)
peak_dictionary.save_hkl(os.path.join(directory, outname+'_w_ext.int'))

peak_statistics = PeakStatistics(os.path.join(directory, outname+'_w_abs.int'), peak_dictionary.hm)
peak_statistics.prune_outliers()
peak_statistics.write_statisics()
peak_statistics.write_intensity()

peak_statistics = PeakStatistics(os.path.join(directory, outname+'_w_ext.int'), peak_dictionary.hm)
peak_statistics.prune_outliers()
peak_statistics.write_statisics()
peak_statistics.write_intensity()