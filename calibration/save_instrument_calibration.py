import numpy as np
import matplotlib.pyplot as plt
from mantid.simpleapi import *
import os 

directory = os.path.dirname(os.path.realpath(__file__))

# directories ------------------------------------------------------------------
calibration_name = 'garnet_2022a_refined'
peaks_workspace = 'garnet_2022a_refined_peaks_workspace.nxs'

LoadNexus(OutputWorkspace='peaks', 
          Filename=os.path.join(directory,peaks_workspace))
       
SCDCalibratePanels(
    PeakWorkspace="peaks",
    RecalculateUB=False,
    OutputWorkspace='testCaliTable',
    DetCalFilename=os.path.join(directory,calibration_name+'.DetCal'),
    CSVFilename=os.path.join(directory,calibration_name+'.csv'),
    XmlFilename=os.path.join(directory,calibration_name+'.xml'),
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