# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python310.zip')
sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python3.10')
sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python3.10/lib-dynload')
sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python3.10/site-packages')

from matplotlib.backends.backend_pdf import PdfPages
from itertools import cycle

import sys, os, re

directory = '/SNS/software/scd/reduction/inscape_dev/integration/'
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

import parameters

from peak import PeakDictionary, PeakStatistics, PeakFitPrune

from mantid.geometry import PointGroupFactory, SpaceGroupFactory, CrystalStructure

from mantid.kernel import V3D

from scipy.optimize import minimize, least_squares, curve_fit
#from lmfit import minimize, Parameters

import copy

_, filename, *ub_twin_list = sys.argv

#filename = '/SNS/CORELLI/IPTS-31203/shared/test_integration_Sr4Ru3O10_SS2/Sr4Ru3O10_HHL_150K_0T_fix_int.inp'
#filename = '/SNS/CORELLI/IPTS-31203/shared/final_integration_Sr4Ru3O10_SS2/Sr4Ru3O10_HHL_150K_0T_fix_int.inp'

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

if sg is None:
    sg = 'P 1'

if dictionary.get('chemical-formula') is not None:
    chemical_formula = ''.join([' '+item if item.isalpha() else item for item in re.findall(r'[\(\)A-Za-z]+|[\d?\d.\d]+', dictionary['chemical-formula'])]).lstrip(' ')
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
peak_dictionary.apply_spherical_correction(vanadium_mass, os.path.join(outdir, 'absorption.txt'))

#peak_dictionary.apply_ellipsoidal_correction(vanadium_mass, ratios=[4,1,2], omega=np.deg2rad(0), fname=os.path.join(outdir, 'absorption.txt'))

keys = list(peak_dictionary.peak_dict.keys())

#pg = PointGroupFactory.createPointGroup('-1')

symmetry = { }

for key in keys:

    h, k, l, m, n, p = key

    if m**2+n**2+p**2 == 0:

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

            if peak.is_peak_integrated() and len(peak.get_individual_bin_size()) > 0:

                pk_vol_fract = peak.get_merged_peak_volume_fraction()

                I = peak.get_intensity()
                sig = peak.get_intensity_error()

                rot = peak.get_omega_angles()

                #merge = peak.get_merged_intensity()
                #error = peak.get_merged_intensity_error()

                intens = peak.get_individual_intensity()
                error = peak.get_individual_intensity_error()

                mask = np.isfinite(I) & (I > 3*sig) & (intens > 3*error)

                if len(I[mask]) > 0 and pk_vol_fract > 0.85:

                    average = np.rad2deg(np.angle(np.sum(np.exp(1j*np.deg2rad(rot[mask])))))

                    ave_intens = np.mean(intens[mask])
                    ave_error = np.sqrt(np.mean(error[mask]**2))

                    value.append(ave_intens)
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

def scale(theta, wl, mu, alpha, beta, omega, a, bx, by, c, e):

    t = np.deg2rad(theta-mu)

    x0 = c*np.cos(t)
    y0 = np.zeros_like(x0)
    z0 = c*np.sqrt(1-e**2)*np.sin(t)

    ux = np.cos(np.deg2rad(alpha))*np.sin(np.deg2rad(beta))
    uy = np.sin(np.deg2rad(alpha))*np.sin(np.deg2rad(beta))
    uz = np.cos(np.deg2rad(beta))

    gamma = np.deg2rad(omega)

    U = np.array([[np.cos(gamma)+ux**2*(1-np.cos(gamma)), ux*uy*(1-np.cos(gamma))-uz*np.sin(gamma), ux*uz*(1-np.cos(gamma))+uy*np.sin(gamma)],
                  [uy*ux*(1-np.cos(gamma))+uz*np.sin(gamma), np.cos(gamma)+uy**2*(1-np.cos(gamma)), uy*uz*(1-np.cos(gamma))-ux*np.sin(gamma)],
                  [uz*ux*(1-np.cos(gamma))-uy*np.sin(gamma), uz*uy*(1-np.cos(gamma))+ux*np.sin(gamma), np.cos(gamma)+uz**2*(1-np.cos(gamma))]])

    x, y, z = np.einsum('ij,jk->ik', U, [x0,y0,z0])

    f = np.exp(-0.5*((x-bx)**2+(y-by)**2)/(1+a*wl)**2)

    return 1/f

def residual(x, ref_dict, keys):

#     mu = params['mu']
#     alpha = params['alpha']
# 
#     a = params['a']
#     b = params['b']
#     c = params['c']
#     e = params['e']
# 
#     x = mu.value, alpha.value, a.value, b.value, c.value, e.value

    mu, alpha, beta, omega, a, bx, by, c, e = x

    #print((6*'{:.4f} ').format(*x))

    diff = []

    for equivalents in keys:

        y, z, sig = [], [], []

        peaks_1 = ref_dict[equivalents[0]]
        peaks_2 = ref_dict[equivalents[1]]

        for peak_1 in peaks_1:

            if peak_1.is_peak_integrated() and len(peak_1.get_individual_bin_size()) > 0:
                #peak_1.individual_integrate()
                #peak_1.prune_peaks()

                lamda_1 = peak_1.get_wavelengths().copy()
                omega_1 = peak_1.get_omega_angles().copy()
                scales_1 = peak_1.get_data_scale().copy()

                intens_1 = peak_1.get_individual_intensity().copy()
                sig_intens_1 = peak_1.get_individual_intensity_error().copy()

                fit_intens_1 = peak_1.get_individual_fitted_intensity().copy()
                fit_sig_intens_1 = peak_1.get_individual_fitted_intensity_error().copy()

                pk_vol_fract_1 = peak_1.get_individual_peak_volume_fraction().copy()

                mask_1 = (intens_1 > 3*sig_intens_1) & (fit_intens_1 > 3*fit_sig_intens_1) & (pk_vol_fract_1 > 0.85)

                lamda_1 = lamda_1[mask_1]
                omega_1 = omega_1[mask_1]
                scales_1 = scales_1[mask_1]

                intens_1 = intens_1[mask_1]
                sig_intens_1 = sig_intens_1[mask_1]
                indices_1 = np.arange(lamda_1.size) 

                if len(indices_1) > 0:

                    for peak_2 in peaks_2:

                        if peak_2.is_peak_integrated() and len(peak_2.get_individual_bin_size()) > 0:
                            #peak_2.individual_integrate()
                            #peak_2.prune_peaks()

                            lamda_2 = peak_2.get_wavelengths().copy()
                            omega_2 = peak_2.get_omega_angles().copy()
                            scales_2 = peak_2.get_data_scale().copy()

                            intens_2 = peak_2.get_individual_intensity().copy()
                            sig_intens_2 = peak_2.get_individual_intensity_error().copy()

                            fit_intens_2 = peak_2.get_individual_fitted_intensity().copy()
                            fit_sig_intens_2 = peak_2.get_individual_fitted_intensity_error().copy()

                            pk_vol_fract_2 = peak_2.get_individual_peak_volume_fraction().copy()

                            mask_2 = (intens_2 > 3*sig_intens_2) & (fit_intens_2 > 3*fit_sig_intens_2) & (pk_vol_fract_2 > 0.85)

                            lamda_2 = lamda_2[mask_2]
                            omega_2 = omega_2[mask_2]
                            scales_2 = scales_2[mask_2]

                            intens_2 = intens_2[mask_2]
                            sig_intens_2 = sig_intens_2[mask_2]
                            indices_2 = np.arange(lamda_2.size)

                            if len(indices_2) > 0:

                                angle_1 = np.mod(omega_1, 180)
                                angle_2 = np.mod(omega_2, 180)

                                inds = np.isclose(angle_1, angle_2[:,np.newaxis], atol=1e-1)
                                #print(angle_1.round(2))
                                #print(angle_2.round(2))
                                #print(inds)

                                i1, i2 = np.meshgrid(indices_1, indices_2, indexing='xy')

                                i1_inds, i2_inds = np.unique(i1[inds]), np.unique(i2[inds])

                                i1_inds = i1_inds[np.argsort(angle_1[i1_inds])]
                                i2_inds = i2_inds[np.argsort(angle_2[i2_inds])]

                                if len(i1_inds) > 0 and len(i2_inds) > 0 and len(intens_1) > 0 and len(intens_2) > 0 and len(i1_inds) == len(i2_inds) and len(intens_1) == len(intens_2) and len(intens_1) > i1_inds.max():

                                    s1 = scale(omega_1, lamda_1, mu, alpha, beta, omega, a, bx, by, c, e)
                                    s2 = scale(omega_2, lamda_2, mu, alpha, beta, omega, a, bx, by, c, e)

                                    #sig_intens_1 = peak_1.get_partial_merged_intensity_error(i1_inds, False)
                                    #sig_intens_2 = peak_2.get_partial_merged_intensity_error(i2_inds, False)

                                    #sig_intens_1 = peak_1.get_individual_intensity_error()
                                    #sig_intens_2 = peak_2.get_individual_intensity_error()

                                    #sig_intens_1 = np.sqrt(np.mean(sig_intens_1[i1_inds]**2))
                                    #sig_intens_2 = np.sqrt(np.mean(sig_intens_2[i2_inds]**2))

                                    #peak_1.set_data_scale(scales_1*s1)
                                    #peak_2.set_data_scale(scales_2*s2)

                                    #intens_1 = peak_1.get_partial_merged_intensity(i1_inds)
                                    #intens_2 = peak_2.get_partial_merged_intensity(i2_inds)

                                    #intens_1 = peak_1.get_individual_intensity()
                                    #intens_2 = peak_2.get_individual_intensity()

                                    #intens_1 = np.mean(intens_1[i1_inds])
                                    #intens_2 = np.mean(intens_2[i2_inds])

                                    intens_1 = intens_1[i1_inds]
                                    intens_2 = intens_2[i2_inds]

                                    sig_intens_1 = sig_intens_1[i1_inds]
                                    sig_intens_2 = sig_intens_2[i2_inds]

                                    s1 = s1[i1_inds]
                                    s2 = s2[i2_inds]

                                    if len(intens_1) > 0 and len(intens_2) > 0:

                                        #ave_intens_1 = np.sum(intens_1/sig_intens_1**2)/np.sum(1/sig_intens_2**2)
                                        #ave_intens_2 = np.sum(intens_2/sig_intens_2**2)/np.sum(1/sig_intens_2**2)

                                        #ave_sig_intens_1 = 1/np.sqrt(np.sum(1/sig_intens_1**2))
                                        #ave_sig_intens_2 = 1/np.sqrt(np.sum(1/sig_intens_2**2))

                                        #peak_1.set_data_scale(scales_1)
                                        #peak_2.set_data_scale(scales_2)

                                        #z = np.array([ave_intens_1,ave_intens_2])
                                        #sig = np.array([ave_sig_intens_1,ave_sig_intens_2])

                                        #z0 = np.sum(z/sig**2)/np.sum(1/sig**2)
                                        #sig0 = 1/np.sqrt(np.sum(1/sig**2))

                                        #x0 = intens_1/intens_2
                                        #y0 = intens_2/intens_1
                                        #x = s2/s1
                                        #y = s1/s2
                                        #sig = np.sqrt(sig_intens_1**2/intens_1**2+sig_intens_2**2/intens_2**2)

                                        #diff += (np.abs(x0-x)/sig).tolist() #np.linalg.norm
                                        #diff += (np.abs(y0-y)/sig).tolist() #np.linalg.norm

                                        diff += (intens_1*s1-intens_2*s2).tolist()
                                        #diff += (1/intens_1/s1-1/intens_2/s2).tolist()

                                        #z0 = np.diff(z)
                                        #diff += ((intens_1-z0)/sig_intens_1).tolist()
                                        #diff += ((intens_2-z0)/sig_intens_2).tolist()

                                        #inv_z0 = np.diff(1/z)
                                        #diff += ((1/intens_1-1/z0)*sig_intens_1).tolist()
                                        #diff += ((1/intens_2-1/z0)*sig_intens_2).tolist()

                                        z += (intens_1*s1).tolist()
                                        z += (intens_2*s2).tolist()
                                        sig += (sig_intens_1).tolist()
                                        sig += (sig_intens_2).tolist()

        z, sig = np.array(z), np.array(sig)
        #z0 = np.sum(z/sig**2)/np.sum(1/sig**2)
        #sig0 = 1/np.sqrt(np.sum(1/sig**2))

        #y0 = np.mean(y)
        z0 = np.mean(z)

        #diff += ((y-y0)).tolist()
        #diff += ((z-z0)).tolist()
        #diff += ((1/z-1/z0)*sig0).tolist()

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

alpha = 90
beta = 90
omega = 0
a = 0
bx = k_const/2
by = 0
c = k_const/2
d = 0.1
e = 0.0

x0 = (mu, alpha, beta, omega, a, bx, by, c, e)
args = (ref_dict, data)
bounds = ([-180, -180, 0, -180, 0, 0, 0, 0, 0], [180, 180, 180, 180, np.inf, np.inf, np.inf, np.inf, 1])

sol = least_squares(residual, x0, args=args, bounds=np.array(bounds), method='trf', verbose=2) #, method='trust-constr', loss='soft_l1'
mu, alpha, beta, omega, a, bx, by, c, e = sol.x

# params = Parameters()
# params.add('mu', value=mu, min=-180, max=180)
# params.add('alpha', value=alpha, min=-180, max=180)
# params.add('a', value=a, min=0, max=np.inf)
# params.add('b', value=b, min=0, max=np.inf)
# params.add('c', value=c, min=0, max=np.inf)
# params.add('e', value=e, min=0, max=1)
# 
# out = minimize(residual, params, args=(ref_dict, data))
# 
# mu = out.params['mu'].value
# alpha = out.params['alpha'].value
# 
# a = out.params['a'].value
# b = out.params['b'].value
# c = out.params['c'].value
# e = out.params['e'].value

with open(os.path.join(outdir, 'wobble.txt'), 'w') as f:

    f.write('goniometer offset: {:.4f} deg \n'.format(mu))
    f.write('sample offset: {:.4f} deg \n'.format(omega))
    f.write('rotation axis azimuth: {:.4f} deg \n'.format(alpha))
    f.write('rotation axis polar: {:.4f} deg \n'.format(beta))
    f.write('wavelength sensitivity: {:.4f} \n'.format(a))
    f.write('off-centering mean x parameter: {:.4f} \n'.format(bx))
    f.write('off-centering mean y parameter: {:.4f} \n'.format(by))
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

        if peak.is_peak_integrated() and len(peak.get_individual_bin_size()) > 0:

            intens = peak.get_individual_intensity()
            scales = peak.get_data_scale().copy()
            omegas = peak.get_omega_angles()
            lamdas = peak.get_wavelengths()

            peak.set_data_scale(scales*scale(omegas, lamdas, mu, alpha, beta, omega, a, bx, by, c, e))

            corr = peak.get_individual_intensity()

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

            if peak.is_peak_integrated() and len(peak.get_individual_bin_size()) > 0:

                I = peak.get_intensity()
                rot = peak.get_omega_angles()

                intens = peak.get_individual_intensity()
                error = peak.get_individual_intensity_error()

                mask = np.isfinite(I) & (intens > 3*error)

                if len(I[mask]) > 0:

                    average = np.rad2deg(np.angle(np.sum(np.exp(1j*np.deg2rad(rot[mask])))))

                    ave_intens = np.mean(intens[mask])
                    ave_error = np.sqrt(np.mean(error[mask]**2))

                    value.append(ave_intens)
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

for pn in range(mtd['iws'].getNumberPeaks()-1,-1,-1):

    pk = mtd['cws'].getPeak(pn)

    h, k, l = pk.getIntHKL()
    m, n, p = pk.getIntMNP()

    if (np.array([h,k,l,m,n,p]) == 0).all():
        mtd['iws'].removePeak(pn)
        mtd['cws'].removePeak(pn)

ub_twin_list = [os.path.abspath(os.path.join(directory, ub_twin)) for ub_twin in ub_twin_list]

for i, ub_twin in enumerate(ub_twin_list):
    CloneWorkspace(InputWorkspace='cws', OutputWorkspace='cws{}'.format(i))
    LoadIsawUB(InputWorkspace='cws{}'.format(i), Filename=ub_twin)
    IndexPeaks(PeaksWorkspace='cws{}'.format(i), Tolerance=0.12)

for pn in range(mtd['iws'].getNumberPeaks()-1,-1,-1):

    pk = mtd['iws'].getPeak(pn)

    h, k, l = pk.getIntHKL()
    m, n, p = pk.getIntMNP()

    for i, ub_twin in enumerate(ub_twin_list):
        pk_twin = mtd['cws{}'.format(i)].getPeak(pn)
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

# peak_dictionary.save_calibration(os.path.join(outdir, outname+'_cal.nxs'))
# peak_dictionary.recalculate_hkl(fname=os.path.join(outdir, 'indexing.txt'))
scale = peak_dictionary.save_hkl(os.path.join(outdir, outname+'_w_pre.hkl'), adaptive_scale=adaptive_scale, scale=scale)
peak_dictionary.save_reflections(os.path.join(outdir, outname+'_w_pre.hkl'), adaptive_scale=False, scale=scale)
peak_dictionary.save(os.path.join(outdir, outname+'_corr.pkl'))

if max_order == 0:
    peak_prune = PeakFitPrune(os.path.join(outdir, outname+'_w_pre_norm.hkl'), sg)
    peak_prune.fit_peaks()
    peak_prune.write_intensity()

if sg is not None:
    peak_statistics = PeakStatistics(os.path.join(outdir, outname+'_w_pre.hkl'), sg)
    peak_statistics.prune_outliers()
    peak_statistics.write_statisics()
    peak_statistics.write_intensity() 