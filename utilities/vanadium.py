# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

import multiprocessing

filename = '/SNS/TOPAZ/shared/Vanadium/2021A_0216_AG/2021A_0216_AG.inp' #sys.argv[0]#, sys.argv[1]

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

directory = os.path.dirname(os.path.abspath(filename))
cycle = dictionary['cycle']

calibration_directory = '/SNS/{}/shared/calibration/'.format(instrument)
vanadium_directory = '/SNS/{}/shared/Vanadium/{}'.format(instrument,cycle)
background_directory = '/SNS/{}/shared/Background/{}'.format(instrument,cycle)

if not os.path.exists(vanadium_directory):
    os.mkdir(vanadium_directory)

parameters.output_input_file(filename, vanadium_directory, cycle)

if dictionary.get('tube-file') is not None:
    tube_calibration = os.path.join(calibration_directory, dictionary['tube-file'])
else:
    tube_calibration = None

if dictionary['detector-file'] is not None:
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

sample_mass = dictionary['mass']

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
        MaskBTP(Workspace=instrument, Bank=bank_to_mask, Tube=tube_to_mask, Pixel=pixel_to_mask)

ExtractMask(InputWorkspace=instrument,
            OutputWorkspace='mask')

SaveMask(InputWorkspace='mask',
         OutputFile=os.path.join(vanadium_directory, 'mask.xml'))

if type(run_nos) is list:
    files_to_load = '+'.join([os.path.join(file_directory.format(instrument,ipts), file_name.format(instrument,run)) for run in run_nos])
else:
    files_to_load = os.path.join(file_directory.format(instrument,ipts), file_name.format(instrument,run_nos))

if type(bkg_nos) is list:
    bkg_files_to_load = '+'.join([os.path.join(file_directory.format(instrument,bkg_ipts), file_name.format(instrument,run)) for run in bkg_nos])
else:
    bkg_files_to_load = os.path.join(file_directory.format(instrument,bkg_ipts), file_name.format(instrument,bkg_nos))

rebin_param = '{},{},{}'.format(k_min,k_max,k_max)

if tof_min is not None and tof_max is not None:
    Load(Filename=files_to_load,
         OutputWorkspace='van',
         LoadMonitors=False,
         FilterByTofMin=tof_min,
         FilterByTofMax=tof_max)
else:
    Load(Filename=files_to_load,
         LoadMonitors=False,
         OutputWorkspace='van')

if instrument_filename is not None:
    LoadInstrument(Workspace='van', Filename=instrument_filename, RewriteSpectraMap=True)

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

CopyInstrumentParameters(InputWorkspace='van',
                         OutputWorkspace=instrument)

SaveNexusGeometry(InputWorkspace=instrument,
                  Filename=os.path.join(vanadium_directory, 'calibration.nxs'))

LoadInstrument(Workspace='van',
               Filename=os.path.join(vanadium_directory, 'calibration.nxs'),
               RewriteSpectraMap=True)

LoadMask(Instrument=instrument,
         InputFile=os.path.join(vanadium_directory, 'mask.xml'),
         RefWorkspace='van',
         OutputWorkspace='mask')

MaskDetectors(Workspace='van', MaskedWorkspace='mask')

ConvertUnits(InputWorkspace='van', OutputWorkspace='van', Target='Momentum')
Rebin(InputWorkspace='van', OutputWorkspace='van', Params=rebin_param)
CropWorkspace(InputWorkspace='van', OutputWorkspace='van', XMin=k_min, XMax=k_max)

# CloneWorkspace(InputWorkspace='van', OutputWorkspace='test')

Rebin(InputWorkspace='van',
      OutputWorkspace='sa',
      Params=rebin_param,
      PreserveEvents=False)

GroupDetectors(InputWorkspace='van',
               ExcludeGroupNumbers=banks_to_mask,
               MapFile=os.path.join(vanadium_directory, 'grouping.xml'),
               OutputWorkspace='van')

Rebin(InputWorkspace='van', OutputWorkspace='van', Params=rebin_param)

data = mtd['van']
for i in range(data.getNumberHistograms()):
    el = data.getSpectrum(i)
    if data.readY(i)[0] > 0:
        el.divide(data.readY(i)[0], data.readE(i)[0])

SortEvents(InputWorkspace='van', SortBy='X Value')

IntegrateFlux(InputWorkspace='van', OutputWorkspace='flux', NPoints=10000)

#FFTDerivative(InputWorkspace='flux', Order=1, OutputWorkspace='spectrum')
#FFTSmooth(InputWorkspace='spectrum', OutputWorkspace='spectrum', Filter='Zeroing', Params='1000', AllSpectra=True)
#ConvertUnits(InputWorkspace='spectrum', Target='Wavelength', OutputWorkspace='spectrum')

#FFTSmooth(InputWorkspace='flux', OutputWorkspace='flux', Filter='Zeroing', Params='1000', AllSpectra=True)

sa_name = '_'.join(['solid_angle', append_name])
flux_name = '_'.join(['flux', append_name])
bkg_name = '_'.join(['background', append_name])

SaveNexus(InputWorkspace='sa', Filename=os.path.join(vanadium_directory, sa_name+'.nxs'))
SaveNexus(InputWorkspace='flux', Filename=os.path.join(vanadium_directory, flux_name+'.nxs'))
SaveNexus(InputWorkspace='bkg', Filename=os.path.join(background_directory, bkg_name+'.nxs'))
