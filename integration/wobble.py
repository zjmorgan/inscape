# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python38.zip')
sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python3.8')
sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python3.8/lib-dynload')
sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python3.8/site-packages')

from matplotlib.backends.backend_pdf import PdfPages
from itertools import cycle

import sys, os, re

directory = '/SNS/software/scd/reduction/inscape_dev/integration/'
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

import parameters

from peak import PeakDictionary, PeakStatistics

from mantid.geometry import PointGroupFactory, SpaceGroupFactory, CrystalStructure

from mantid.kernel import V3D

from scipy.optimize import minimize, least_squares, curve_fit

import copy

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

if sg is not None:
    pg = PointGroupFactory.createPointGroupFromSpaceGroup(SpaceGroupFactory.createSpaceGroup(sg))

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

scale = np.loadtxt(os.path.join(outdir, 'scale.txt'))

cif_file = dictionary.get('cif-file')

peak_dictionary = PeakDictionary(a, b, c, alpha, beta, gamma)

if cif_file is not None:
    peak_dictionary.load_cif(os.path.join(working_directory, cif_file))

peak_dictionary.set_satellite_info(mod_vector_1, mod_vector_2, mod_vector_3, max_order)
peak_dictionary.set_material_info(chemical_formula, z_parameter, sample_mass)
peak_dictionary.set_scale_constant(scale_constant)

peak_dictionary.load(os.path.join(outdir, outname+'.pkl'))

keys = list(peak_dictionary.peak_dict.keys())

pg = PointGroupFactory.createPointGroup('-1')

symmetry = { }

for key in keys:

    h, k, l, m, n, p = key

    equivalents = [[h,k,l,m,n,p],[-h,-k,-l,-m,-n,-p]]

    symm_key = (int(h),int(k),int(l),int(m),int(n),int(p))

    if symmetry.get(symm_key) is None:

        for equivalent in equivalents:

            h, k, l, m, n, p = equivalent 

            equi_key = (int(h),int(k),int(l),int(m),int(n),int(p))

            if peak_dictionary.peak_dict.get(equi_key) is not None:

                if symmetry.get(symm_key) is not None:

                    pair_list = symmetry[symm_key]
                    pair_list.append(equi_key)
                    symmetry[symm_key] = pair_list

                else:

                    symmetry[symm_key] = [equi_key]

                keys.remove(equi_key)

keys = list(symmetry.keys())

for key in keys:

    equivalents = symmetry[key]

    if len(equivalents) <= 1:

        symmetry.pop(key)

plt.close('all')

values = []
angles = []

vals = []
angs = []

symmetry_pairs = []

for key in symmetry.keys():

    equivalents = symmetry[key]

    value = []
    angle = []

    for equivalent in equivalents:

        h, k, l, m, n, p = equivalent

        peaks = peak_dictionary.peak_dict[equivalent]

        for ind, peak in enumerate(peaks):

            if peak.is_peak_integrated():

                pk_vol_fract = peak.get_merged_peak_volume_fraction()

                I = peak.get_intensity()
                sig = peak.get_intensity_error()

                rot = peak.get_omega_angles()

                merge = peak.get_merged_intensity()
                error = peak.get_merged_intensity_error()

                mask = np.isfinite(I) & (I > 3*sig)

                if len(I[mask]) > 0 and merge > 3*error and pk_vol_fract > 0.85:

                    average = np.rad2deg(np.angle(np.sum(np.exp(1j*np.deg2rad(rot[mask])))))

                    value.append(merge)
                    angle.append(average)

    if len(value) >= 2:

        value = (np.array(value)/np.mean(value)).tolist()

        values += value
        angles += angle

        vals.append(value)
        angs.append(angle)

        symmetry_pairs.append(equivalents)

values = np.array(values)
angles = np.array(angles)

def scale(theta, wl, mu, alpha, a, b, c, e):

    t = np.deg2rad(theta-mu)

    beta = np.deg2rad(alpha)

    d = c*np.sqrt(1-e**2)

    x = c*np.cos(t)*np.cos(beta)-d*np.sin(t)*np.sin(beta)

    f = np.exp(-(x-b)**2/(1+a*wl)**2)

    return 1/f

def residual(x, ref_dict, keys):

    mu, alpha, a, b, c, e = x

    print((6*'{:.4f} ').format(*x))

    diff = []

    for equivalents in keys:

        z, sig, key_list = [], [], []

        for equivalent in equivalents:

            peaks = ref_dict[equivalent]

            for peak in peaks:

                sig0 = peak.get_merged_intensity_error(fit_contrib=False)

                if sig0 > 0:

                    w = peak.get_wavelengths()
                    t = peak.get_omega_angles()

                    s = scale(t, w, mu, alpha, a, b, c, e)

                    scales = peak.get_data_scale().copy()

                    peak.set_data_scale(scales*s)

                    z0 = peak.get_merged_intensity()

                    peak.set_data_scale(scales)

                    z.append(z0)
                    sig.append(sig0)

                    key_list.append(equivalent)

        z, sig = np.array(z), np.array(sig)

        z0 = np.mean(z)
        sig0 = np.array(sig) # np.sqrt(np.sum(np.square(sig))+np.var(z))

        diff += ((z-z0)/sig0).tolist()
        diff += ((1/z-1/z0)*sig0).tolist()

    return diff

def init(theta, mu, k):

    return np.exp(k*np.cos(np.deg2rad(theta-mu)))

const = 1.5

popt, pcov = curve_fit(init, angles, values, (0,0.1), bounds=([-180,0],[180,np.inf]), loss='soft_l1', verbose=2)
mu, k_const = popt

sort = np.argsort(angles)

fig, ax = plt.subplots()
ax.scatter(angles, values, color='C0')
ax.plot(angles[sort], init(angles[sort], mu, k_const), '-', color='C1')
ax.plot(angles[sort], init(angles[sort], mu, k_const)/const, '-', color='C2')
ax.plot(angles[sort], init(angles[sort], mu, k_const)*const, '-', color='C2')
ax.grid(True)
ax.set_ylim(0.1,10)
ax.set_yscale('log')
ax.set_xlabel('Goniometer angle')
ax.set_ylabel('Ratio')
ax.minorticks_on()
fig.savefig(os.path.join(outdir, 'wobble_uncorrected.pdf'))

mask = (init(angles, mu, k_const)/const < values) & (values < init(angles, mu, k_const)*const)

popt, pcov = curve_fit(init, angles[mask], values[mask], (0,0.1), bounds=([-180,0],[180,np.inf]), verbose=2)
mu, k_const = popt

sort = np.argsort(angles)

fig, ax = plt.subplots()
ax.scatter(angles[mask], values[mask], color='C0')
ax.plot(angles[sort], init(angles[sort], mu, k_const), '-', color='C1')
ax.plot(angles[sort], init(angles[sort], mu, k_const)/const, '-', color='C2')
ax.plot(angles[sort], init(angles[sort], mu, k_const)*const, '-', color='C2')
ax.grid(True)
ax.set_ylim(0.1,10)
ax.set_yscale('log')
ax.set_xlabel('Goniometer angle')
ax.set_ylabel('Ratio')
ax.minorticks_on()
fig.savefig(os.path.join(outdir, 'wobble_uncorrected_prune.pdf'))

data = []
for i, (pairs, ang, val) in enumerate(zip(symmetry_pairs,angs,vals)):
    mask = (init(ang, mu, k_const)/const < val) & (val < init(ang, mu, k_const)*const)
    if mask.all():
        data.append(pairs)

ref_dict = copy.deepcopy(peak_dictionary.peak_dict)

alpha = 0
a = 0
b = k_const/2
c = k_const/2
e = 0

x0 = (mu, alpha, a, b, c, e)
args = (ref_dict, data)
bounds = ([-180, -180, 0, 0, 0, 0], [180, 180, np.inf, np.inf, np.inf, 1])

sol = least_squares(residual, x0, args=args, bounds=bounds, loss='soft_l1', verbose=2) #, method='trust-constr'
mu, alpha, a, b, c, e = sol.x

with open(os.path.join(outdir, 'wobble.txt'), 'w') as f:

    f.write('goniometer offset: {:.4f} deg \n'.format(mu))
    f.write('sample offset: {:.4f} deg \n'.format(alpha))
    f.write('wavelength sensitivity: {:.4f} \n'.format(a))
    f.write('offcentering mean parameter: {:.4f} \n'.format(b))
    f.write('offcentering effective radius: {:.4f} \n'.format(c))
    f.write('eccentricity: {:.4f} \n'.format(e))

# print(sol.x)
# print(mu, alpha, a, b, c, e)
# 
# print(4*b*c-k_const)

ratios = []
angles = []
wavelengths = []

for key in peak_dictionary.peak_dict.keys():

    peaks = peak_dictionary.peak_dict.get(key)

    h, k, l, m, n, p = key

    for peak in peaks:

        if peak.get_merged_intensity() > 0:

            intens = peak.get_intensity()
            scales = peak.get_data_scale().copy()
            omegas = peak.get_omega_angles()
            lamdas = peak.get_wavelengths()

            peak.set_data_scale(scales*scale(omegas, lamdas, mu, alpha, a, b, c, e))

            corr = peak.get_intensity()

            ratios.append(corr/intens)
            angles.append(omegas)
            wavelengths.append(lamdas)

fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
fig2, ax2 = plt.subplots()

x = np.linspace(-180,180,360)
y = np.deg2rad(x)

for i in range(len(angles)):
    s1 = ax1.scatter(np.deg2rad(angles[i]), 1/ratios[i], c=wavelengths[i], s=1, vmin=0.8, vmax=2.5)
    s2 = ax2.scatter(angles[i], 1/ratios[i], c=wavelengths[i], s=1, vmin=0.8, vmax=2.5)

cb1 = fig1.colorbar(s1, ax=ax1, orientation='vertical')
cb2 = fig2.colorbar(s2, ax=ax2, orientation='vertical')

cb1.ax.set_ylabel('Wavelength [ang.]')
cb2.ax.set_ylabel('Wavelength [ang.]')

ax1.grid(True)
ax2.grid(True)

ax1.set_xlabel('Goniometer angle')
ax2.set_xlabel('Goniometer angle')

ax1.set_ylabel('Ratio')
ax2.set_ylabel('Ratio')

ax1.set_ylim(0,1)
ax2.set_ylim(0,1)

ax1.minorticks_on()
ax2.minorticks_on()

# ax1.legend()
# ax2.legend()

fig1.savefig(os.path.join(outdir, 'wobble_polar.pdf'))
fig2.savefig(os.path.join(outdir, 'wobble_angle.pdf'))

values = []
angles = []

for key in symmetry.keys():

    equivalents = symmetry[key]

    value = []
    angle = []

    for equivalent in equivalents:

        h, k, l, m, n, p = equivalent

        peaks = peak_dictionary.peak_dict[equivalent]

        for ind, peak in enumerate(peaks):

            I = peak.get_intensity()
            rot = peak.get_omega_angles()
            merge = peak.get_merged_intensity()

            mask = np.isfinite(I) & (I > 0)

            if len(I[mask]) > 0 and merge > 0:

                average = np.rad2deg(np.angle(np.sum(np.exp(1j*np.deg2rad(rot[mask])))))

                value.append(merge)
                angle.append(average)

    if len(value) >= 2:

        value = np.array(value)/np.mean(value)

        values += value.tolist()
        angles += angle

values = np.array(values)
angles = np.array(angles)

const = 1.5

popt, pcov = curve_fit(init, angles, values, (0,0.1), bounds=([-180,0],[180,np.inf]), loss='soft_l1', verbose=2)
mu, k_const = popt

sort = np.argsort(angles)

fig, ax = plt.subplots()
ax.scatter(angles, values, color='C0')
ax.plot(angles[sort], init(angles[sort], mu, k_const), '-', color='C1')
ax.plot(angles[sort], init(angles[sort], mu, k_const)/const, '-', color='C2')
ax.plot(angles[sort], init(angles[sort], mu, k_const)*const, '-', color='C2')
ax.grid(True)
ax.set_ylim(0.1,10)
ax.set_yscale('log')
ax.set_xlabel('Goniometer angle')
ax.set_ylabel('Ratio')
ax.minorticks_on()
fig.savefig(os.path.join(outdir, 'wobble_corrected.pdf'))

mask = (init(angles, mu, k_const)/const < values) & (values < init(angles, mu, k_const)*const)

popt, pcov = curve_fit(init, angles[mask], values[mask], (0,0.1), bounds=([-180,0],[180,np.inf]), verbose=2)
mu, k = popt

sort = np.argsort(angles)

fig, ax = plt.subplots()
ax.scatter(angles[mask], values[mask], color='C0')
ax.plot(angles[sort], init(angles[sort], mu, k_const), '-', color='C1')
ax.plot(angles[sort], init(angles[sort], mu, k_const)/const, '-', color='C2')
ax.plot(angles[sort], init(angles[sort], mu, k_const)*const, '-', color='C2')
ax.grid(True)
ax.set_ylim(0.1,10)
ax.set_yscale('log')
ax.set_xlabel('Goniometer angle')
ax.set_ylabel('Ratio')
ax.minorticks_on()
fig.savefig(os.path.join(outdir, 'wobble_corrected_prune.pdf'))

# for key in peak_dictionary.peak_dict.keys():
#     h, k, l, m, n, p = key
#     pair = (-h,-k,-l,-m,-n,-p)
#     peaks = peak_dictionary.peak_dict[key]
#     # if peak_dictionary.peak_dict.get(pair) is None:
#     #     for peak in peaks:
#     #         scale = peak.get_data_scale().copy()
#     #         peak.set_data_scale(0*scale)
#     # else:
#     metric = 0
#     pairs = peak_dictionary.peak_dict[pair]
#     for peak in pairs:
#         metric += peak.get_merged_intensity()
#     metric /= len(pairs)
#     for peak in peaks:
#         rot = peak.get_omega_angles()
#         value = peak.get_merged_intensity()
#         angle = np.rad2deg(np.angle(np.sum(np.exp(1j*np.deg2rad(rot)))))
#         if metric*init(angle, mu, k_const)/const > value or value > metric*init(angle, mu, k_const)*const:
#             scale = peak.get_data_scale().copy()
#             peak.set_data_scale(0*scale)

LoadIsawUB(InputWorkspace='cws', Filename=os.path.join(outdir, outname+'_cal.mat'))

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

LoadIsawUB(InputWorkspace='cws', Filename=os.path.join(outdir, outname+'_cal.mat'))

peak_dictionary.save_calibration(os.path.join(outdir, outname+'_cal.nxs'))
peak_dictionary.recalculate_hkl(fname=os.path.join(outdir, 'indexing.txt'))
scale = peak_dictionary.save_hkl(os.path.join(outdir, outname+'_w_pre.hkl'), adaptive_scale=adaptive_scale, scale=scale)
peak_dictionary.save_reflections(os.path.join(outdir, outname+'_w_pre.hkl'), adaptive_scale=False, scale=scale)
peak_dictionary.save(os.path.join(outdir, outname+'_corr.pkl'))