# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import Pool

from execute import f

import sys 
import os 

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/home/zgf/.git/inscape/integration/')

# calibration peaks workspace -------------------------------------------------
peaks_workspace = 'garnet_calibration.nxs'

# super-resolution instrument definition file ---------------------------------
sr_directory = '/SNS/CORELLI/shared/SCDCalibration/'
sr_file = 'CORELLI_Definition_2017-04-04_superresolution.xml'

# calibration files -----------------------------------------------------------
detector_calibration = '/SNS/CORELLI/IPTS-23019/shared/germanium_2021b/germanium_2021B_corrected.xml'
tube_calibration = '/SNS/CORELLI/shared/calibration/tube/calibration_corelli_20200109.nxs.h5'

LoadEmptyInstrument(FileName=os.path.join(sr_directory,sr_file), 
                    OutputWorkspace='super_resolution')
                    
LoadNexus(Filename=tube_calibration, OutputWorkspace='tube_table')

ApplyCalibration(Workspace='super_resolution', CalibrationTable='tube_table')
LoadParameterFile(Workspace='super_resolution', Filename=detector_calibration)

peaks = LoadNexus(Filename=os.path.join('/home/zgf/.git/inscape/calibration/',peaks_workspace))

n_peaks = peaks.getNumberPeaks()
n_proc = 24

names = [os.path.join(directory,'test_file_{}.nxs'.format(p)) for p in np.arange(n_proc)]

for name, ps in zip(names,np.array_split(np.arange(n_peaks),n_proc)):
    tmp = CreatePeaksWorkspace(NumberOfPeaks=0,OutputType='LeanElasticPeak')
    for p in ps:
        tmp.addPeak(peaks.getPeak(int(p)))
    SaveNexus(InputWorkspace=tmp, Filename=name)

with Pool(n_proc) as p:
    p.map(f, names)
    
out = CreatePeaksWorkspace(InstrumentWorkspace='super_resolution', NumberOfPeaks=0, OutputType='Peak')
    
for name in names:
    tmp = LoadNexus(Filename=name)
    CombinePeaksWorkspaces(LHSWorkspace='out', RHSWorkspace='tmp', OutputWorkspace='out')
    
banks = mtd['out'].column(13)
for i in range(mtd['out'].getNumberPeaks()-1,-1,-1):
    if (banks[i] == ''):
        mtd['out'].removePeak(i)
    
#SaveNexus(InputWorkspace='out', Filename=os.path.join(directory,peaks_workspace.replace('.nxs','_sr.nxs')))
#SaveIsawPeaks(InputWorkspace='out', Filename=os.path.join(directory,peaks_workspace.replace('.nxs','_sr.peaks')))