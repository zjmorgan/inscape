import numpy as np
import matplotlib.pyplot as plt
from mantid.simpleapi import *

T, F = True, False

instrument = 'SNAP'

import sys 
import os 

sys.path.append('/SNS/users/zgf/.git/inscape/integration/')

# current calibration ----------------------------------------------------------
directory = '/SNS/users/zgf/Documents/data/snap'
detector_calibration = directory+'/snap.detcal'

# '/SNS/MANDI/shared/calibration/2022A/garnet_cal_calibration_w_rot_centered.DetCal'
# '/SNS/MANDI/shared/calibration/2022A/garnet_calibration.xml' 
# '/SNS/CORELLI/IPTS-23019/shared/germanium_2021b/germanium_2021B_corrected.xml'

# lattice constants ------------------------------------------------------------
a     = 4.76
b     = 4.76
c     = 12.99
alpha = 90
beta  = 90
gamma = 120

# calibration procedure --------------------------------------------------------
sample    = [T, F, T, T, T, T, T, T, T, F, F, F]
panels    = [F, F, F, T, T, T, T, T, T, F, T, F]
moderator = [F, T, F, F, F, F, T, T, T, T, T, T]
size      = [F, F, F, F, F, F, F, F, F, T, F, T]
time      = [F, F, F, F, F, F, F, F, F, F, F, F]

# sample    = [F, T, T, T]
# panels    = [T, T, T, T]
# moderator = [F, T, T, T]
# size      = [F, F, F, F]
# time      = [F, F, F, F]

# calibration files ------------------------------------------------------------
peaks_workspace = 'sapphire_cal_sr.nxs'
calibration_name = 'sapphire_calibration'

# ------------------------------------------------------------------------------

LoadNexus(OutputWorkspace='peaks', 
          Filename=os.path.join(directory,peaks_workspace))

banks = mtd['peaks'].column(13)
for i in range(mtd['peaks'].getNumberPeaks()-1,-1,-1):
    if (banks[i] == ''):
        mtd['peaks'].removePeak(i)

LoadEmptyInstrument(InstrumentName=instrument, OutputWorkspace=instrument)

sr_directory = None # '/SNS/CORELLI/shared/SCDCalibration/'
sr_file = None # 'CORELLI_Definition_2017-04-04_super_resolution.xml'

if sr_directory is not None and sr_file is not None:
    LoadEmptyInstrument(FileName=sr_directory+sr_file, 
                        OutputWorkspace=detector_calibration+'_super_resolution')

if detector_calibration is not None:
    if os.path.splitext(detector_calibration)[1] == '.xml':
        LoadParameterFile(Workspace=instrument, Filename=detector_calibration)
    else:
        LoadIsawDetCal(InputWorkspace=instrument, Filename=detector_calibration)

    if sr_directory is not None and sr_file is not None:
        if os.path.splitext(detector_calibration)[1] == '.xml':
            LoadParameterFile(Workspace=instrument+'_super_resolution', Filename=detector_calibration)
        else:
            LoadIsawDetCal(InputWorkspace=instrument+'_super_resolution', Filename=detector_calibration)
        peaks = ApplyInstrumentToPeaks('peaks', instrument+'_super_resolution')
    else:
        peaks = ApplyInstrumentToPeaks('peaks', instrument)

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
    FilterPeaks(InputWorkspace='peaks', 
                FilterVariable='RunNumber', 
                FilterValue=run, 
                Operator='=', 
                OutputWorkspace='tmp')
    gon.setR(mtd['tmp'].getPeak(0).getGoniometerMatrix())
    omega, chi, phi = gon.getEulerAngles('YZY')
    print(omega,chi,phi)
    R.append([omega, chi, phi])

R = np.array(R)
    
for j in range(N):
    mtd['peaks'].getPeak(j).setGoniometerMatrix(np.eye(3))

for i, _ in enumerate(panels):

    FilterPeaks(InputWorkspace='peaks', 
                FilterVariable='h^2+k^2+l^2', 
                FilterValue=0, 
                Operator='!=',
                OutputWorkspace='peaks')

    FilterPeaks(InputWorkspace='peaks', 
                FilterVariable='QMod', 
                FilterValue=0, 
                Operator='>', 
                OutputWorkspace='peaks')

    FilterPeaks(InputWorkspace='peaks', 
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

        FilterPeaks(InputWorkspace='peaks', 
                    FilterVariable='RunNumber', 
                    FilterValue=run, 
                    Operator='=', 
                    OutputWorkspace='tmp')
        
        if mtd['tmp'].getNumberPeaks() > 10:
        
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

    FilterPeaks(InputWorkspace='peaks', 
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
        OutputWorkspace='calibration_table',
        DetCalFilename=os.path.join(directory,calibration_name+'.DetCal'),
        CSVFilename=os.path.join(directory,calibration_name+'.csv'),
        XmlFilename=os.path.join(directory,calibration_name+'.xml'),
        CalibrateT0=time[i],
        SearchRadiusT0=10,
        CalibrateL1=moderator[i],
        SearchRadiusL1=0.2,
        CalibrateBanks=panels[i],
        SearchRadiusTransBank=0.5,
        SearchRadiusRotXBank=5,
        SearchRadiusRotYBank=5,
        SearchRadiusRotZBank=5,
        VerboseOutput=True,
        SearchRadiusSamplePos=0.2,
        TuneSamplePosition=sample[i],
        CalibrateSize=size[i],
        SearchRadiusSize=0.1,
        FixAspectRatio=True,
    )

    LoadParameterFile(Workspace=instrument, 
                      Filename=os.path.join(directory,calibration_name+'.xml'))

    if sr_directory is not None and sr_file is not None:
        LoadParameterFile(Workspace=instrument+'_super_resolution', 
                          Filename=os.path.join(directory,calibration_name+'.xml'))
        peaks = ApplyInstrumentToPeaks('peaks', instrument+'_super_resolution')
    else:
        peaks = ApplyInstrumentToPeaks('peaks', instrument)
    
SaveNexus(InputWorkspace='peaks', 
          Filename=os.path.join(directory,calibration_name+'_peaks_workspace.nxs'))