import numpy as np
import matplotlib.pyplot as plt
from mantid.simpleapi import *

import sys 
import os 

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/home/zgf/.git/inscape/integration/')

# directories ------------------------------------------------------------------
calibration_name = 'garnet_2022a_refined'
peaks_workspace = 'garnet_2022a_refined_peaks_workspace.nxs'

sr_directory = '/SNS/CORELLI/shared/SCDCalibration/'
sr_file = 'CORELLI_Definition_2017-04-04_superresolution.xml'

LoadEmptyInstrument(FileName=os.path.join(sr_directory,sr_file), 
                    OutputWorkspace='corelli_superresolution')

LoadParameterFile(Workspace='corelli_superresolution', 
                  Filename=os.path.join(directory,calibration_name+'.xml'))

LoadNexus(OutputWorkspace='peaks', 
          Filename=os.path.join(directory,peaks_workspace))
          
sample_pos = mtd['peaks'].getPeak(0).getSamplePos()

banks =  list(set(mtd['peaks'].column(13)))
banks.sort()

for bank in banks:
    MoveInstrumentComponent('corelli_superresolution', 
                            bank+'/sixteenpack', 
                            X=-sample_pos[0], Y=-sample_pos[1], Z=-sample_pos[2], 
                            RelativePosition=True)
  
MoveInstrumentComponent('corelli_superresolution', 
                        'sample-position', 
                        X=-sample_pos[0], Y=-sample_pos[1], Z=-sample_pos[2], 
                        RelativePosition=True)
                        
MoveInstrumentComponent('corelli_superresolution', 
                        'moderator', 
                        X=0, Y=0, Z=-sample_pos[2], 
                        RelativePosition=True)
                        
peaks = ApplyInstrumentToPeaks('peaks', 'corelli_superresolution')
       
SCDCalibratePanels(
    PeakWorkspace="peaks",
    RecalculateUB=False,
    OutputWorkspace='testCaliTable',
    DetCalFilename=os.path.join(directory,calibration_name+'_centered.DetCal'),
    CSVFilename=os.path.join(directory,calibration_name+'_centered.csv'),
    XmlFilename=os.path.join(directory,calibration_name+'_centered.xml'),
    CalibrateT0=False,
    SearchRadiusT0=10,
    CalibrateL1=False,
    SearchRadiusL1=0.2,
    CalibrateBanks=False,
    SearchRadiusTransBank=0.5,
    SearchradiusRotXBank=0,
    SearchradiusRotYBank=0,
    SearchradiusRotZBank=0,
    VerboseOutput=True,
    SearchRadiusSamplePos=0.2,
    TuneSamplePosition=False,
) 