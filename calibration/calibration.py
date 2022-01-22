import numpy as np
import matplotlib.pyplot as plt
from mantid.simpleapi import *

T, F = True, False

# directories ------------------------------------------------------------------
iptsfolder = '/SNS/CORELLI/IPTS-23019/'

import sys 
import os 

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/home/zgf/.git/inscape/integration/')

# current calibration ----------------------------------------------------------
detector_calibration = '/SNS/CORELLI/IPTS-23019/shared/germanium_2021b/germanium_2021B_corrected.xml'
detector_calibration = os.path.join(directory,'garnet_2022a_centered.xml')

# lattice constants ------------------------------------------------------------
a = 11.9157
b = 11.9157
c = 11.9157
alpha = 90
beta = 90
gamma = 90

# calibration procedure --------------------------------------------------------
sample    = [F, T, T, T, T, T, T, T, T]
panels    = [F, F, F, T, T, T, T, T, T]
moderator = [F, F, F, F, F, F, T, T, T]

# calibration files ------------------------------------------------------------
peaks_workspace = 'garnet_calibration_sr.nxs'
ub_file = 'garnet_2022a.mat'
calibration_name = 'garnet_2022a_refined'

# ------------------------------------------------------------------------------

print(os.path.join(directory,peaks_workspace))

LoadNexus(OutputWorkspace='peaks', 
          Filename=os.path.join(directory,peaks_workspace))
LoadIsawUB('peaks', FIlename=os.path.join(directory,ub_file))

#TransformHKL('peaks', HKLTransform='0,1,0,0,0,1,1,0,0')

banks = mtd['peaks'].column(13)
for i in range(mtd['peaks'].getNumberPeaks()-1,-1,-1):
    if (banks[i] == ''):
        mtd['peaks'].removePeak(i)

LoadEmptyInstrument(InstrumentName='CORELLI', OutputWorkspace='corelli')

sr_directory = '/SNS/CORELLI/shared/SCDCalibration/'
sr_file = 'CORELLI_Definition_2017-04-04_superresolution.xml'

LoadEmptyInstrument(FileName=sr_directory+sr_file, 
                    OutputWorkspace='corelli_superresolution')

LoadParameterFile(Workspace='corelli', 
                  Filename=detector_calibration)

LoadParameterFile(Workspace='corelli_superresolution', 
                  Filename=detector_calibration)
                  
peaks = ApplyInstrumentToPeaks('peaks', 'corelli_superresolution')

N = mtd['peaks'].getNumberPeaks()
sr = set()

for i in range(N):
    run = mtd['peaks'].getPeak(i).getRunNumber()
    sr.add(int(run))

runs = list(sr)
runs.sort()

gon = mtd['peaks'].run().getGoniometer()

R = []

for run in runs:
    FilterPeaks('peaks', 
                FilterVariable='RunNumber', 
                FilterValue=run, 
                Operator='=', 
                OutputWorkspace='tmp')
    gon.setR(mtd['tmp'].getPeak(0).getGoniometerMatrix())
    omega, chi, phi = gon.getEulerAngles('YZY')
    R.append([omega, chi, phi])

R = np.array(R)
    
for j in range(N):
    mtd['peaks'].getPeak(j).setGoniometerMatrix(np.eye(3))

for i, _ in enumerate(panels):
    
    FilterPeaks('peaks', 
                FilterVariable='h^2+k^2+l^2', 
                FilterValue=0, 
                Operator='!=',
                OutputWorkspace='peaks')
                
    FilterPeaks('peaks', 
                FilterVariable='QMod', 
                FilterValue=0, 
                Operator='>', 
                OutputWorkspace='peaks')
                
    FilterPeaks('peaks', 
                BankName='', 
                Criterion='!=', 
                OutputWorkspace='peaks')
    
    N = mtd['peaks'].getNumberPeaks()
    for j in range(N):
        mtd['peaks'].getPeak(j).setGoniometerMatrix(np.eye(3))
        
    CreatePeaksWorkspace(InstrumentWorkspace='peaks', 
                         NumberOfPeaks=0, 
                         OutputWorkspace='peaks_tmp')
    
    for run in runs:
        
        FilterPeaks('peaks', 
                    FilterVariable='RunNumber', 
                    FilterValue=run, 
                    Operator='=', 
                    OutputWorkspace='tmp')
                    
        CalculateUMatrix(PeaksWorkspace='tmp',
                         a=a,
                         b=b,
                         c=c,
                         alpha=alpha,
                         beta=beta,
                         gamma=gamma)
            
        for j in range(mtd['tmp'].getNumberPeaks()):
            u = mtd['tmp'].sample().getOrientedLattice().getU()
            mtd['tmp'].getPeak(j).setGoniometerMatrix(u)
            
        CombinePeaksWorkspaces(LHSWorkspace='tmp', 
                               RHSWorkspace='peaks_tmp', 
                               OutputWorkspace='peaks_tmp')
    
    CloneWorkspace(InputWorkspace='peaks_tmp', OutputWorkspace='peaks')
    
    SetUB('peaks', 
          a=a, 
          b=b, 
          c=c, 
          alpha=alpha, 
          beta=beta, 
          gamma=gamma, 
          u='0,0,1', 
          v='1,0,0')
        
    FilterPeaks('peaks', 
                FilterVariable='h^2+k^2+l^2', 
                FilterValue=0, 
                Operator='!=', 
                OutputWorkspace='pws')

    # calibration ----------------------
    SCDCalibratePanels(
        PeakWorkspace='peaks',
        RecalculateUB=False,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        OutputWorkspace='testCaliTable',
        DetCalFilename=os.path.join(directory,calibration_name+'.DetCal'),
        CSVFilename=os.path.join(directory,calibration_name+'.csv'),
        XmlFilename=os.path.join(directory,calibration_name+'.xml'),
        CalibrateT0=False,
        SearchRadiusT0=10,
        CalibrateL1=moderator[i],
        SearchRadiusL1=0.2,
        CalibrateBanks=panels[i],
        SearchRadiusTransBank=0.5,
        SearchradiusRotXBank=0,
        SearchradiusRotYBank=0,
        SearchradiusRotZBank=0,
        VerboseOutput=True,
        SearchRadiusSamplePos=0.2,
        TuneSamplePosition=sample[i],
    )  
        
    LoadParameterFile(Workspace='corelli', 
                      Filename=os.path.join(directory,calibration_name+'.xml'))
                      
    LoadParameterFile(Workspace='corelli_superresolution', 
                      Filename=os.path.join(directory,calibration_name+'.xml'))
        
    peaks = ApplyInstrumentToPeaks('peaks', 
                                   'corelli_superresolution')
    
SaveNexus(InputWorkspace='peaks', 
          Filename=os.path.join(directory,calibration_name+'_peaks_workspace.nxs'))