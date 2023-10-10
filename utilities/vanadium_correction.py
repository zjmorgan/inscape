# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

import os
import sys

import scipy.optimize

#filename = sys.argv[1] #'/SNS/CORELLI/IPTS-23019/shared/Vanadium/vanadium.config'
#filename = sys.argv[1]  #'/SNS/TOPAZ/IPTS-31189/shared/YAG/calibration/vanadium.config'

#filename = '/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/2022B_0725_CCR_5x7.inp'
filename = '/SNS/CORELLI/shared/Vanadium/2023B_0813_CCR_5x8/2023B_0813_CCR_5x8.inp'

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

import parameters

dictionary = parameters.load_input_file(filename)

run_nos = dictionary['runs']
bkg_nos = dictionary['background-runs']

facility, instrument = parameters.set_instrument(dictionary['instrument'])

ipts = dictionary['ipts']
bkg_ipts = dictionary['background-ipts']
bkg_scale = dictionary['background-scale']
beam_power = dictionary.get('minimum-beam-power')

directory = os.path.dirname(os.path.abspath(filename))
cycle = dictionary['cycle']

calibration_directory = '/SNS/{}/shared/calibration/'.format(instrument)
if '/' in cycle:
    cycle = cycle.split('/')[0]
    vanadium_directory = os.path.join(directory,cycle)
    background_directory = os.path.join(directory,cycle)
else:
    vanadium_directory = '/SNS/{}/shared/Vanadium/{}'.format(instrument,cycle)
    background_directory = '/SNS/{}/shared/Background/{}'.format(instrument,cycle)

if not os.path.exists(vanadium_directory):
    os.mkdir(vanadium_directory)

parameters.output_input_file(filename, vanadium_directory, cycle)

if dictionary.get('tube-file') is not None:
    tube_calibration = os.path.join(calibration_directory, dictionary['tube-file'])
else:
    tube_calibration = None

if dictionary.get('detector-file') is not None:
    detector_calibration = os.path.join(calibration_directory, dictionary['detector-file'])
else:
    detector_calibration = None

banks_to_mask = dictionary.get('banks-to-mask')
pixels_to_mask = dictionary.get('rows-to-mask')
tubes_to_mask = dictionary.get('columns-to-mask')
bank_tube_to_mask = dictionary.get('banks-columns-to-mask')
bank_tube_pixel_to_mask = dictionary.get('banks-columns-rows-to-mask')

if type(banks_to_mask) is int:
    banks_to_mask = [banks_to_mask]

k_min, k_max = dictionary['k-range']

append_name = str(k_min).replace('.','p')+'-'+str(k_max).replace('.','p')

if dictionary.get('tof-range') is not None:
    tof_min, tof_max = dictionary['tof-range']
else:
    tof_min, tof_max = None, None

sample_mass = dictionary.get('mass')
if sample_mass is None:
    sample_mass = 0

file_directory = '/SNS/{}/IPTS-{}/nexus/'
file_name = '{}_{}.nxs.h5'

instrument_filename = dictionary.get('filename')

if instrument_filename is not None:
    LoadEmptyInstrument(Filename=instrument_filename,
                        OutputWorkspace=instrument)
else:
    LoadEmptyInstrument(InstrumentName=instrument,
                        OutputWorkspace=instrument)

ExtractMonitors(InputWorkspace=instrument, DetectorWorkspace=instrument, MonitorWorkspace='monitors')

CreateGroupingWorkspace(InputWorkspace=instrument,
                        GroupDetectorsBy='bank',
                        OutputWorkspace='group')

SaveDetectorsGrouping(InputWorkspace='group',
                      OutputFile=os.path.join(vanadium_directory, 'grouping.xml'))

if banks_to_mask is not None:
    banks_to_mask = ','.join([str(banks[0])+'-'+str(banks[-1]) if type(banks) is list else str(banks) for banks in banks_to_mask])
    MaskBTP(Workspace=instrument, Bank=banks_to_mask)

if pixels_to_mask is not None:
    pixels_to_mask = ','.join([str(pixels[0])+'-'+str(pixels[-1]) if type(pixels) is list else str(pixels) for pixels in pixels_to_mask])
    MaskBTP(Workspace=instrument, Pixel=pixels_to_mask)

if tubes_to_mask is not None:
    tubes_to_mask = ','.join([str(tubes[0])+'-'+str(tubes[-1]) if type(tubes) is list else str(tubes) for tubes in tubes_to_mask])
    MaskBTP(Workspace=instrument, Tube=tubes_to_mask)

if bank_tube_to_mask is not None:
    bank_tube_to_mask = [[str(banks[0])+'-'+str(banks[-1]) if type(banks) is list else str(banks),\
                          str(tubes[0])+'-'+str(tubes[-1]) if type(tubes) is list else str(tubes)] for banks, tubes in bank_tube_to_mask]
    for bank_to_mask, tube_to_mask in bank_tube_to_mask:
        MaskBTP(Workspace=instrument, Bank=bank_to_mask, Tube=tube_to_mask)

if bank_tube_pixel_to_mask is not None:
    bank_tube_pixel_to_mask = [[str(banks[0])+'-'+str(banks[-1]) if type(banks) is list else str(banks),\
                                str(tubes[0])+'-'+str(tubes[-1]) if type(tubes) is list else str(tubes),\
                                str(pixels[0])+'-'+str(pixels[-1]) if type(pixels) is list else str(pixels)] for banks, tubes, pixels in bank_tube_pixel_to_mask]
    for bank_to_mask, tube_to_mask, pixel_to_mask in bank_tube_pixel_to_mask:
        if '-' in bank_to_mask:
            b_to_mask = bank_to_mask.split('-')
            for b in range(int(b_to_mask[0]),int(b_to_mask[1])+1):
                MaskBTP(Workspace=instrument, Bank=b, Tube=tube_to_mask, Pixel=pixel_to_mask)
        else:
            MaskBTP(Workspace=instrument, Bank=int(float(bank_to_mask)), Tube=tube_to_mask, Pixel=pixel_to_mask)

ExtractMask(InputWorkspace=instrument,
            OutputWorkspace='mask')

SaveMask(InputWorkspace='mask',
         OutputFile=os.path.join(vanadium_directory, 'mask.xml'))

if type(run_nos) is list:
    files_to_load = [os.path.join(file_directory.format(instrument,ipts), file_name.format(instrument,run)) for run in run_nos]
else:
    files_to_load = [os.path.join(file_directory.format(instrument,ipts), file_name.format(instrument,run_nos))]

if type(bkg_nos) is list:
    bkg_files_to_load = '+'.join([os.path.join(file_directory.format(instrument,bkg_ipts), file_name.format(instrument,run)) for run in bkg_nos])
else:
    bkg_files_to_load = os.path.join(file_directory.format(instrument,bkg_ipts), file_name.format(instrument,bkg_nos))

rebin_param = '{},{},{}'.format(k_min,k_max,k_max)

if tube_calibration is not None:

    LoadNexus(Filename=os.path.join(calibration_directory, tube_calibration),
              OutputWorkspace='tube_table')

gd_runs = []
no_evnts = []

for i, file_to_load in enumerate(files_to_load):

    Load(Filename=file_to_load,
         LoadMonitors=False,
         OutputWorkspace='ws')

    if instrument == 'CORELLI':

        gon_axis = 'BL9:Mot:Sample:Axis3'
        possible_axes = ['BL9:Mot:Sample:Axis1', 'BL9:Mot:Sample:Axis2', 'BL9:Mot:Sample:Axis3', 
                         'BL9:Mot:Sample:Axis1.RBV', 'BL9:Mot:Sample:Axis2.RBV', 'BL9:Mot:Sample:Axis3.RBV'] #.RBV

        for possible_axis in possible_axes:
            if mtd['ws'].run().hasProperty(possible_axis):
                angle = np.mean(mtd['ws'].run().getProperty(possible_axis).value)
                if not np.isclose(angle,0):
                    gon_axis = possible_axis

        SetGoniometer(Workspace='ws', Axis0='{},0,1,0,1'.format(gon_axis))

    else:

        SetGoniometer(Workspace='ws', Goniometers='Universal') 

    add_to = True

    if beam_power is not None:

        FilterBadPulses(InputWorkspace='ws', LowerCutoff=95, OutputWorkspace='ws')

        if beam_power > mtd['ws'].run().getProperty('BeamPower').timeAverageValue()*0.975:
            add_to = False

    if add_to:

        gd_runs.append(mtd['ws'].run().getGoniometer().getEulerAngles('YZY')[0])

        MaskDetectors(Workspace='ws', MaskedWorkspace='mask')
        NormaliseByCurrent(InputWorkspace='ws', OutputWorkspace='ws_norm')

        no_evnts.append(mtd['ws_norm'].getNumberEvents())

        if i == 0:
            CloneWorkspace(InputWorkspace='ws', OutputWorkspace='van')
        else:
            Plus(LHSWorkspace='van', RHSWorkspace='ws', OutputWorkspace='van')

        GroupDetectors(InputWorkspace='ws_norm',
                       CopyGroupingFromWorkspace='group',
                       OutputWorkspace='ws_{}'.format(i))

np.savetxt(os.path.join(vanadium_directory, 'normalized_counts.txt'), np.column_stack([gd_runs,no_evnts]))

GroupWorkspaces(GlobExpression='ws_*', OutputWorkspace='ws')

ConvertUnits(InputWorkspace='ws', OutputWorkspace='ws', Target='Momentum')
Rebin(InputWorkspace='ws', OutputWorkspace='ws', Params='{},{},{}'.format(k_min,0.1,k_max))

counts = []
angles = []

for i, file_to_load in enumerate(files_to_load):

    ws = 'ws_{}'.format(i)

    k_bins = mtd[ws].readX(0)
    k = 0.5*(k_bins[1:]+k_bins[:-1])
    wl = 2*np.pi/k

    omega, chi, phi = mtd[ws].run().getGoniometer(0).getEulerAngles()

    count = mtd[ws].extractY()
    mask = count.any(axis=1)
    banks = np.arange(mask.size)+1
    banks = banks[mask]
    count = count[mask]

    counts.append(count)
    angles.append(omega)

counts = np.array(counts)
angles = np.array(angles)

sort = np.argsort(angles)

counts = counts[sort]
angles = angles[sort]

min_inds = []
max_inds = []

with PdfPages(os.path.join(vanadium_directory, 'spectrum.pdf')) as pdf:
    for i, b in enumerate(banks):
        fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 0.5]})
        ax[0].pcolormesh(angles, k, counts[:,i,:].T, linewidth=0, rasterized=True)
        ax[0].set_xlabel('Goniometer angle [deg.]')
        ax[0].set_ylabel('Momentum [inv. ang.]')
        ax[0].set_title('Bank #{}'.format(b))
        ax[0].minorticks_on()
        ax[1].plot(counts[:,i,:].mean(axis=0), k, label='Ave')
        ind_max, ind_peak = np.unravel_index(np.argmax(counts[:,i,:]), counts.shape[0::2])
        ind_min = (ind_max+counts.shape[0] // 2) % counts.shape[0]
        min_inds.append(ind_min)
        max_inds.append(ind_max)
        ax[1].plot(counts[ind_min,i,:], k, label='Min')
        ax[1].plot(counts[ind_max,i,:], k, label='Max')
        ax[1].legend()
        ax[1].set_xlabel('Counts')
        ax[1].minorticks_on()
        pdf.savefig()
        plt.close()

def scale_function(theta, wl, mu, alpha, beta, omega, a, bx, by, c, e):

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

def model(x, angles, k):

    theta, wl = np.meshgrid(angles, 2*np.pi/k, indexing='ij')

    scale = scale_function(theta.flatten(), wl.flatten(), *x).reshape(angles.size, k.size)

    return np.einsum('ik,ijk->ijk', scale, counts)

def residual(x, angles, k, counts):

    scale_counts = model(x, angles, k)

    return (scale_counts-scale_counts.mean(axis=0)).flatten()

mu = 0
alpha = 90
beta = 90
omega = 0
a = 0
bx = 1
by = 0
c = 1
d = 0.1
e = 0.0

x0 = (mu, alpha, beta, omega, a, bx, by, c, e)
args = (angles, k, counts)
bounds = ([-180, -180, 0, -180, 0, 0, 0, 0, 0], [180, 180, 180, 180, np.inf, np.inf, np.inf, np.inf, 1])

sol = scipy.optimize.least_squares(residual, x0, args=args, method='lm', verbose=2) #, bounds=np.array(bounds), method='trust-constr', loss='soft_l1'

scale_counts = model(sol.x, angles, k)

with PdfPages(os.path.join(vanadium_directory, 'spectrum_corrected.pdf')) as pdf:
    for i, b in enumerate(banks):
        fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 0.5]})
        ax[0].pcolormesh(angles, k, scale_counts[:,i,:].T, linewidth=0, rasterized=True)
        ax[0].set_xlabel('Goniometer angle [deg.]')
        ax[0].set_ylabel('Momentum [inv. ang.]')
        ax[0].set_title('Bank #{}'.format(b))
        ax[0].minorticks_on()
        ax[1].plot(scale_counts[:,i,:].mean(axis=0), k, label='Ave')
        ax[1].plot(scale_counts[min_inds[i],i,:], k, label='Min')
        ax[1].plot(scale_counts[max_inds[i],i,:], k, label='Max')
        ax[1].legend()
        ax[1].set_xlabel('Counts')
        ax[1].minorticks_on()
        pdf.savefig()
        plt.close()

(mu, alpha, beta, omega, a, bx, by, c, e) = sol.x

for i, file_to_load in enumerate(files_to_load):

    Load(Filename=file_to_load,
         LoadMonitors=False,
         OutputWorkspace='ws_corr')

    if instrument == 'CORELLI':

        gon_axis = 'BL9:Mot:Sample:Axis3'
        possible_axes = ['BL9:Mot:Sample:Axis1', 'BL9:Mot:Sample:Axis2', 'BL9:Mot:Sample:Axis3', 
                         'BL9:Mot:Sample:Axis1.RBV', 'BL9:Mot:Sample:Axis2.RBV', 'BL9:Mot:Sample:Axis3.RBV'] #.RBV

        for possible_axis in possible_axes:
            if mtd['ws_corr'].run().hasProperty(possible_axis):
                angle = np.mean(mtd['ws_corr'].run().getProperty(possible_axis).value)
                if not np.isclose(angle,0):
                    gon_axis = possible_axis

        SetGoniometer(Workspace='ws_corr', Axis0='{},0,1,0,1'.format(gon_axis))

    else:

        SetGoniometer(Workspace='ws_corr', Goniometers='Universal') 

    theta, _, _ = mtd['ws_corr'].run().getGoniometer(0).getEulerAngles()

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

    x, y, z = np.einsum('ij,j->i', U, [x0,y0,z0])

    const = 0.5*(x-bx)**2+(y-by)**2

    # inv_f = np.exp(0.5*((x-bx)**2+(y-by)**2)/(1+a*wl)**2)

    FilterBadPulses(InputWorkspace='ws_corr', LowerCutoff=95, OutputWorkspace='ws_corr')

    ConvertUnits(InputWorkspace='ws_corr', OutputWorkspace='ws_corr', Target='Momentum')

    Rebin(InputWorkspace='ws_corr', OutputWorkspace='ws_corr', Params='{},{},{}'.format(k_min,0.1,k_max))

    CropWorkspace(InputWorkspace='ws_corr', OutputWorkspace='ws_corr', XMin=k_min, XMax=k_max)

    ConvertUnits(InputWorkspace='ws_corr', OutputWorkspace='ws_corr', Target='Wavelength')

    CreateSingleValuedWorkspace(DataValue=0, OutputWorkspace='val')

    Multiply(LHSWorkspace='ws_corr', RHSWorkspace='val', OutputWorkspace='scale')

    CreateSingleValuedWorkspace(DataValue=const, OutputWorkspace='val')

    ConvertUnits(InputWorkspace='scale', OutputWorkspace='scale', Target='Momentum')

    Rebin(InputWorkspace='scale', OutputWorkspace='scale', Params='{},{},{}'.format(k_min,0.1,k_max), PreserveEvents=False)

    ConvertUnits(InputWorkspace='scale', OutputWorkspace='scale', Target='Wavelength')

    Plus(LHSWorkspace='scale', RHSWorkspace='val', OutputWorkspace='scale')

    PolynomialCorrection(InputWorkspace='scale', Coefficients=[1, 2*a, a**2], Operation='Divide', OutputWorkspace='scale')

    Exponential(InputWorkspace='scale', OutputWorkspace='scale')

    Multiply(LHSWorkspace='ws_corr', RHSWorkspace='scale', OutputWorkspace='ws_corr')

    if i == 0:
        CloneWorkspace(InputWorkspace='ws_corr', OutputWorkspace='van')
    else:
        Plus(LHSWorkspace='van', RHSWorkspace='ws_corr', OutputWorkspace='van')

    GroupDetectors(InputWorkspace='ws_corr',
                   CopyGroupingFromWorkspace='group',
                   OutputWorkspace='ws_{}'.format(i))

GroupWorkspaces(GlobExpression='ws_*', OutputWorkspace='ws')

ConvertUnits(InputWorkspace='ws', OutputWorkspace='ws', Target='Momentum')
Rebin(InputWorkspace='ws', OutputWorkspace='ws', Params='{},{},{}'.format(k_min,0.1,k_max))

counts = []
angles = []

for i, file_to_load in enumerate(files_to_load):

    ws = 'ws_{}'.format(i)

    wl_bins = mtd[ws].readX(0)
    wl = 0.5*(wl_bins[1:]+wl_bins[:-1])

    omega, chi, phi = mtd[ws].run().getGoniometer(0).getEulerAngles()

    count = mtd[ws].extractY()
    mask = count.any(axis=1)
    banks = np.arange(mask.size)+1
    banks = banks[mask]
    count = count[mask]

    counts.append(count)
    angles.append(omega)

counts = np.array(counts)
angles = np.array(angles)

sort = np.argsort(angles)

counts = counts[sort]
angles = angles[sort]

with PdfPages(os.path.join(vanadium_directory, 'spectrum_scaled.pdf')) as pdf:
    for i, b in enumerate(banks):
        fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 0.5]})
        ax[0].pcolormesh(angles, wl, counts[:,i,:].T, linewidth=0, rasterized=True)
        ax[0].set_xlabel('Goniometer angle [deg.]')
        ax[0].set_ylabel('Wavelength [ang.]')
        ax[0].set_title('Bank #{}'.format(b))
        ax[0].minorticks_on()
        ax[1].plot(counts[:,i,:].mean(axis=0), wl, label='Ave')
        ind_max, ind_peak = np.unravel_index(np.argmax(counts[:,i,:]), counts.shape[0::2])
        ind_min = (ind_max+counts.shape[0] // 2) % counts.shape[0]
        ax[1].plot(counts[ind_min,i,:], wl, label='Min')
        ax[1].plot(counts[ind_max,i,:], wl, label='Max')
        ax[1].legend()
        ax[1].set_xlabel('Counts')
        ax[1].minorticks_on()
        pdf.savefig()
        plt.close()

ConvertUnits(InputWorkspace='van',
             OutputWorkspace='van',
             Target='Momentum')

AddSampleLog(Workspace='van', 
             LogName='vanadium-mass', 
             LogText=str(sample_mass),
             LogType='Number Series')

NormaliseByCurrent(InputWorkspace='van',
                   OutputWorkspace='van')

if tof_min is not None and tof_max is not None:

    Load(Filename=bkg_files_to_load,
         OutputWorkspace='bkg',
         FilterByTofMin=tof_min,
         FilterByTofMax=tof_max)

else:

    Load(Filename=bkg_files_to_load,
         OutputWorkspace='bkg')

if instrument_filename is not None:
    LoadInstrument(Workspace='bkg', Filename=instrument_filename, RewriteSpectraMap=True)

NormaliseByCurrent(InputWorkspace='bkg',
                   OutputWorkspace='normbkg')

ConvertUnits(InputWorkspace='normbkg',
             OutputWorkspace='normbkg',
             Target='Momentum')

mtd['normbkg'] *= bkg_scale

Minus(LHSWorkspace='van', RHSWorkspace='normbkg', OutputWorkspace='van')

mtd['normbkg'] /= bkg_scale

if tube_calibration is not None:

    ApplyCalibration(Workspace='van', CalibrationTable='tube_table')

if detector_calibration is not None:

    ext = os.path.splitext(detector_calibration)[1]
    if ext == '.xml':
        LoadParameterFile(Workspace='van',
                          Filename=os.path.join(calibration_directory, detector_calibration))
    else:
        LoadIsawDetCal(InputWorkspace='van',
                       Filename=os.path.join(calibration_directory, detector_calibration))

MaskDetectors(Workspace='van', MaskedWorkspace='mask')

if instrument == 'TOPAZ':
    AnvredCorrection(InputWorkspace='van',
                     OutputWorkspace='van',
                     LinearScatteringCoef=0.367,
                     LinearAbsorptionCoef=0.366,
                     Radius=0.2,
                     PowerLambda=0,
                     OnlySphericalAbsorption=True)

Rebin(InputWorkspace='van', OutputWorkspace='van', Params=rebin_param)
CropWorkspace(InputWorkspace='van', OutputWorkspace='van', XMin=k_min, XMax=k_max)

Rebin(InputWorkspace='van',
      OutputWorkspace='sa',
      Params=rebin_param,
      PreserveEvents=False)

GroupDetectors(InputWorkspace='van',
               ExcludeGroupNumbers=banks_to_mask,
               MapFile=os.path.join(vanadium_directory, 'grouping.xml'),
               OutputWorkspace='van')

data = mtd['van']
for i in range(data.getNumberHistograms()):
    el = data.getSpectrum(i)
    if data.readY(i)[0] > 0:
        el.divide(data.readY(i)[0], data.readE(i)[0])
 
SortEvents(InputWorkspace='van', SortBy='X Value')
IntegrateFlux(InputWorkspace='van', OutputWorkspace='flux', NPoints=1000)

#FFTDerivative(InputWorkspace='spectrum', Order=1, OutputWorkspace='spectrum')

Rebin(InputWorkspace='van', OutputWorkspace='spectrum', Params='{},{},{}'.format(k_min,(k_max-k_min)/1000,k_max), PreserveEvents=False)
mtd['spectrum'] /= (k_max-k_min)/1000

ConvertToPointData(InputWorkspace='spectrum', OutputWorkspace='spectrum')
SmoothData(InputWorkspace='spectrum', OutputWorkspace='spectrum', NPoints=10)

#FFTSmooth(InputWorkspace='flux', OutputWorkspace='flux', Filter='Zeroing', Params='1000', AllSpectra=True)

sa_name = '_'.join(['solid_angle', append_name])
flux_name = '_'.join(['flux', append_name])
bkg_name = '_'.join(['background', append_name])

SaveNexus(InputWorkspace='sa', Filename=os.path.join(vanadium_directory, sa_name+'.nxs'))
SaveNexus(InputWorkspace='flux', Filename=os.path.join(vanadium_directory, flux_name+'.nxs'))
SaveNexus(InputWorkspace='bkg', Filename=os.path.join(background_directory, bkg_name+'.nxs'))

with PdfPages(os.path.join(vanadium_directory, flux_name+'.pdf')) as pdf:

    for i in range(mtd['spectrum'].getNumberHistograms()):

        sp = mtd['spectrum'].getSpectrum(i)
        fl = mtd['flux'].getSpectrum(i)

        sp_no = sp.getSpectrumNo()
        fl_no = fl.getSpectrumNo()

        k, spy, spe = mtd['spectrum'].readX(i), mtd['spectrum'].readY(i), mtd['spectrum'].readE(i)
        k, fly, fle = mtd['flux'].readX(i), mtd['flux'].readY(i), mtd['flux'].readE(i)

        wl = 2*np.pi/k

        fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')

        ax[0].errorbar(wl, spy, linestyle='-', color='C0', rasterized=False)
        ax[0].errorbar(wl, fly, linestyle='-', color='C1', rasterized=False)

        ax[1].errorbar(k, spy, linestyle='-', color='C0', rasterized=False)
        ax[1].errorbar(k, fly, linestyle='-', color='C1', rasterized=False)

        ax[0].set_ylabel('')
        ax[1].set_ylabel('')

        ax[0].set_xlabel('Wavelength [$\AA$]')
        ax[1].set_xlabel('Momentum [$\AA^{-1}$]')

        ax[0].set_title('Bank {}'.format(sp_no))
        ax[1].set_title('Bank {}'.format(fl_no))

        ax[0].minorticks_on()
        ax[1].minorticks_on()

        pdf.savefig()
        plt.close()
 