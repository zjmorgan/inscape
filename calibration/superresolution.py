# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

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

LoadEmptyInstrument(FileName=sr_directory+sr_file, 
                    OutputWorkspace='super_resolution')

LoadParameterFile(Workspace='super_resolution', Filename=detector_calibration)

peaks = LoadNexus(filename=os.path.join(directory,peaks_workspace))

ConvertPeaksWorkspace(PeakWorkspace='peaks', 
                      InstrumentWorkspace='super_resolution', 
                      OutputWorkspace='projected_peaks')