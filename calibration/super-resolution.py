# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import Pool

import sys 
import os 

#directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/home/zgf/.git/inscape/integration/')

n_proc = 5

instrument = 'SNAP'

# calibration peaks workspace -------------------------------------------------
directory = '/SNS/users/zgf/Documents/data/snap/'
peaks_workspace = 'sapphire_cal.nxs'

# super-resolution instrument definition file ---------------------------------
sr_directory = None #'/SNS/CORELLI/shared/SCDCalibration/'
sr_file = None #'CORELLI_Definition_2017-04-04_superresolution.xml'

# calibration files -----------------------------------------------------------
detector_calibration = '/SNS/users/zgf/Documents/data/snap/snap.detcal'
tube_calibration = None # '/SNS/CORELLI/shared/calibration/tube/calibration_corelli_20200109.nxs.h5'

def convert(filename):

    ws = os.path.basename(filename)

    LoadNexus(Filename=filename, OutputWorkspace=ws)

    ConvertPeaksWorkspace(PeakWorkspace=mtd[ws], 
                          InstrumentWorkspace=mtd['super_resolution'],
                          OutputWorkspace=mtd[ws])
                                            
    SaveNexus(InputWorkspace=mtd[ws], Filename=filename)

if sr_directory is not None and sr_file is not None:
    LoadEmptyInstrument(FileName=os.path.join(sr_directory,sr_file), 
                        OutputWorkspace='super_resolution')
else:
    LoadEmptyInstrument(InstrumentName=instrument, 
                        OutputWorkspace='super_resolution')

if tube_calibration is not None:
    LoadNexus(Filename=tube_calibration, OutputWorkspace='tube_table')
    ApplyCalibration(Workspace='super_resolution', CalibrationTable='tube_table')

if detector_calibration is not None:
    if os.path.splitext(detector_calibration)[1] == '.xml':
        LoadParameterFile(Workspace='super_resolution', Filename=detector_calibration)
    else:
        LoadIsawDetCal(InputWorkspace='super_resolution', Filename=detector_calibration)

peaks = LoadNexus(Filename=os.path.join(directory, peaks_workspace))

n_peaks = peaks.getNumberPeaks()

names = [os.path.join(directory,'__tmp_file_{}.nxs'.format(p)) for p in np.arange(n_proc)]

for name, ps in zip(names,np.array_split(np.arange(n_peaks),n_proc)):
    tmp = CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak')
    for p in ps:
        tmp.addPeak(peaks.getPeak(int(p)))
    SaveNexus(InputWorkspace=tmp, Filename=name)

with Pool(n_proc) as p:
    p.map(convert, names)

out = CreatePeaksWorkspace(InstrumentWorkspace='super_resolution', NumberOfPeaks=0, OutputType='Peak')

for name in names:
    tmp = LoadNexus(Filename=name)
    os.remove(name)
    CombinePeaksWorkspaces(LHSWorkspace='out', RHSWorkspace='tmp', OutputWorkspace='out')

banks = mtd['out'].column(13)
for i in range(mtd['out'].getNumberPeaks()-1,-1,-1):
    if (banks[i] == ''):
        mtd['out'].removePeak(i)

SaveNexus(InputWorkspace='out', Filename=os.path.join(directory,peaks_workspace.replace('.nxs','_sr.nxs')))
SaveIsawPeaks(InputWorkspace='out', Filename=os.path.join(directory,peaks_workspace.replace('.nxs','_sr.peaks')))
