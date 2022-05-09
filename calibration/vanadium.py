# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import os

cycle = '2022A_0309_CCR'
cycle = '2022A_0311_CCR'
cycle = '2022A_0312_CCR'
instrument = 'CORELLI'

banks_to_mask = '1-6,29-30,62-67,91,87-88,25-26'
tubes_to_mask = None #'1,16'
pixels_to_mask = '1-12,245-256'

bank_tube_to_mask = ['45,11', '49,1', '52,16']
bank_tube_pixel_to_mask = ['58,13-16,80-130', '59,1-4,80-130']

detector_calibration = '2022A/calibration.xml'
tube_calibration = 'tube/calibration_corelli_20200109.nxs.h5'

bkg_ipts = 23019
bkg_no = 242972
bkg_scale = 0.95

ipts = 23019
run_no = [242912,242971] # 309 3x3
run_no = [243794,243853] # 311 4x4
run_no = [244391,244451] # 312 5x8

append_name = '2p5_8' #'10_50meV'
energy_band = None #[10, 50] # meV

k_min, k_max = 2.5, 8
tof_min, tof_max = None, None

if energy_band is not None:
    ws = CreateSingleValuedWorkspace()
    ax = ws.getAxis(0)
    ax.setUnit('Energy')
    un = ax.getUnit()
    coeff, exp = un.quickConversion('Momentum')
    
    momentum_band = [coeff*e**exp for e in energy_band]
    momentum_band.sort()
    k_min, k_max = momentum_band

chemical_formula = 'V'
unit_cell_volume = 27.642
z_parameter = 2
sample_mass = 0.145 # g
sample_mass = 0.306 # g
sample_mass = 0.872 # g

calibration_directory = '/SNS/{}/shared/calibration/'.format(instrument)
output_directory = '/SNS/{}/shared/Vanadium/{}'.format(instrument,cycle)

file_directory = '/SNS/{}/IPTS-{}/nexus/'
file_name = '{}_{}.nxs.h5'

LoadEmptyInstrument(InstrumentName=instrument, OutputWorkspace=instrument)

SetSampleMaterial(InputWorkspace=instrument, 
                  ChemicalFormula=chemical_formula,
                  ZParameter=z_parameter,
                  UnitCellVolume=unit_cell_volume,
                  SampleMass=sample_mass)

CreateGroupingWorkspace(InputWorkspace=instrument,
                        GroupDetectorsBy='bank',
                        OutputWorkspace='group')

SaveDetectorsGrouping(InputWorkspace='group',
                      OutputFile=os.path.join(output_directory, 'grouping.xml'))

if banks_to_mask is not None:
    MaskBTP(Workspace=instrument, Bank=banks_to_mask)

if tubes_to_mask is not None:
    MaskBTP(Workspace=instrument, Tube=tubes_to_mask)

if pixels_to_mask is not None:
    MaskBTP(Workspace=instrument, Pixel=pixels_to_mask)

if bank_tube_to_mask is not None:
    for pair in bank_tube_to_mask:
        bank_to_mask, tube_to_mask = pair.split(',')
        MaskBTP(Workspace=instrument, Bank=bank_to_mask, Tube=tube_to_mask)

if bank_tube_pixel_to_mask is not None:
    for triplet in bank_tube_pixel_to_mask:
        bank_to_mask, tube_to_mask, pixel_to_mask = triplet.split(',')
        MaskBTP(Workspace=instrument, Bank=bank_to_mask, Tube=tube_to_mask, Pixel=pixel_to_mask)

ExtractMask(InputWorkspace=instrument,
            OutputWorkspace='mask')

SaveMask(InputWorkspace='mask',
         OutputFile=os.path.join(output_directory, 'mask.xml'))

if type(run_no) is list:
    files_to_load = '+'.join([os.path.join(file_directory.format(instrument,ipts), file_name.format(instrument,run)) for run in range(run_no[0],run_no[1]+1)])
else:
    files_to_load = os.path.join(file_directory.format(instrument,ipts), file_name.format(instrument,run_no))

bkg_file = os.path.join(file_directory.format(instrument,bkg_ipts), file_name.format(instrument,bkg_no)) 

mat = mtd[instrument].sample().getMaterial()

sigma_a = mat.absorbXSection()
sigma_s = mat.totalScatterXSection()

M = mat.relativeMolecularMass()
n = mat.numberDensityEffective # A^-3
N = mat.totalAtoms 

rho = (n/N)/0.6022*M
V = sample_mass/rho

R = (0.75/np.pi*V)**(1/3)

mu_s = n*sigma_s
mu_a = n*sigma_a

rebin_param = '{},{},{}'.format(k_min,k_max,k_max)

Load(Filename=files_to_load,
     OutputWorkspace='van',
     FilterByTofMin=tof_min,
     FilterByTofMax=tof_max)

NormaliseByCurrent(InputWorkspace='van',
                   OutputWorkspace='van')

Load(Filename=bkg_file,
     OutputWorkspace='bkg',
     FilterByTofMin=tof_min,
     FilterByTofMax=tof_max)
     
NormaliseByCurrent(InputWorkspace='bkg',
                   OutputWorkspace='bkg')

mtd['bkg'] *= bkg_scale

Minus(LHSWorkspace='van', RHSWorkspace='bkg', OutputWorkspace='van')

if tube_calibration is not None:

    LoadNexus(Filename=os.path.join(calibration_directory, tube_calibration),
              OutputWorkspace='tube_table')
    ApplyCalibration(Workspace='van', CalibrationTable='tube_table')

if detector_calibration is not None:

    ext = os.path.splitext(detector_calibration)[1]
    if ext == '.xml':
        LoadParameterFile(Workspace='van',
                          Filename=os.path.join(calibration_directory, detector_calibration))
    else:
        LoadIsawDetCal(InputWorkspace='van',
                       Filename=os.path.join(calibration_directory, detector_calibration))

LoadMask(Instrument=instrument,
         InputFile=os.path.join(output_directory, 'mask.xml'),
         OutputWorkspace='mask')

MaskDetectors(Workspace='van', MaskedWorkspace='mask')

ConvertUnits(InputWorkspace='van', OutputWorkspace='van', Target='Momentum')
Rebin(InputWorkspace='van', OutputWorkspace='van', Params=rebin_param)
CropWorkspace(InputWorkspace='van', OutputWorkspace='van', XMin=k_min, XMax=k_max)

AnvredCorrection(InputWorkspace='van',
                 OutputWorkspace='van',
                 LinearScatteringCoef=mu_s,
                 LinearAbsorptionCoef=mu_a,
                 Radius=R,
                 OnlySphericalAbsorption=True, 
                 PowerLambda='0')

Rebin(InputWorkspace='van',
      OutputWorkspace='sa',
      Params=rebin_param,
      PreserveEvents=False)

GroupDetectors(InputWorkspace='van',
               ExcludeGroupNumbers=banks_to_mask,
               MapFile=os.path.join(output_directory, 'grouping.xml'),
               OutputWorkspace='van')

Rebin(InputWorkspace='van', OutputWorkspace='van', Params=rebin_param)

data = mtd['van']
for i in range(data.getNumberHistograms()):
    el = data.getSpectrum(i)
    if data.readY(i)[0] > 0:
        el.divide(data.readY(i)[0], data.readE(i)[0])

SortEvents(InputWorkspace='van', SortBy='X Value')

IntegrateFlux(InputWorkspace='van', OutputWorkspace='flux', NPoints=10000)

# FFTSmooth(InputWorkspace='flux', OutputWorkspace='spectrum', Filter='Zeroing', Params='1000', AllSpectra=True)
# FFTDerivative(InputWorkspace='spectrum', Order=1, OutputWorkspace='spectrum')
# ConvertUnits(InputWorkspace='spectrum', Target='Wavelength', OutputWorkspace='spectrum')

# FFTSmooth(InputWorkspace='flux', OutputWorkspace='flux', Filter='Zeroing', Params='1000', AllSpectra=True)

sa_name = ['solid_angle'] if append_name is None else ['solid_angle', append_name] 
flux_name = ['flux'] if append_name is None else ['flux', append_name] 

SaveNexus(InputWorkspace='sa', Filename=os.path.join(output_directory, '_'.join(sa_name)+'.nxs'))
SaveNexus(InputWorkspace='flux', Filename=os.path.join(output_directory, '_'.join(flux_name)+'.nxs'))