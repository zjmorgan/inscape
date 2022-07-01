# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from corelli.calibration import load_calibration_set, apply_calibration
from corelli.calibration.powder import load_and_rebin

import sys 
import os 

instrument = 'CORELLI'
ipts = 28033

# directories ------------------------------------------------------------------
directory = '/SNS/{}/IPTS-{}/'.format(instrument,ipts)
output = 'shared/zgf/'

filename = directory+'nexus/{}_{}.nxs.h5'
filename = directory+'nexus/{}_{}.nxs.h5'

# calibration files -----------------------------------------------------------
detector_calibration =  '/SNS/CORELLI/shared/calibration/2022A/calibration.xml'
tube_calibration = '/SNS/CORELLI/shared/calibration/tube/calibration_corelli_20200109.nxs.h5'

if tube_calibration is not None:
    LoadNexus(Filename=tube_calibration,
              OutputWorkspace='tube_table')

# runs -------------------------------------------------------------

start = 262288
stop = 262338#262374

step = 1

# lattice information ----------------------------------------------------------
cell_type = 'Cubic'
centering = 'F'
reflection_condition = 'All-face centred'

# goniometer axis --------------------------------------------------------------
gon_axis = 'BL9:Mot:Sample:Axis3.RBV'

# ------------------------------------------------------------------------------

runs = np.arange(start,stop+1,step)

ows_to_merge = []
omd_to_merge = []
opk_to_merge = []

for r in runs:
    print('Processing run : %s' %r)
    ows = '{}_{}'.format(instrument,r)
    omd = ows+'_md'
    opk = ows+'_pk'

    ows_to_merge.append(ows)
    omd_to_merge.append(omd)
    opk_to_merge.append(opk)

    if not mtd.doesExist(ows):
        LoadEventNexus(Filename=filename.format(instrument,ipts,instrument,r), OutputWorkspace=ows)

        if tube_calibration is not None:
            ApplyCalibration(Workspace=ows, CalibrationTable='tube_table')

        if detector_calibration is not None:
            LoadParameterFile(Workspace=ows, Filename=detector_calibration)

        if r == runs[0]:
            CreatePeaksWorkspace(InstrumentWorkspace=ows, 
                                 NumberOfPeaks=0, 
                                 OutputType='Peak', 
                                 OutputWorkspace='peaks')

        if instrument == 'CORELLI':
            SetGoniometer(ows, Axis0=str(gon_axis)+',0,1,0,1') 
        else:
            SetGoniometer(ows, Goniometers='Universal') 

    if not mtd.doesExist(omd):

        ConvertToMD(InputWorkspace=ows, 
                    OutputWorkspace=omd, 
                    QDimensions='Q3D',
                    dEAnalysisMode='Elastic',
                    Q3DFrames='Q_sample',
                    LorentzCorrection=True,
                    MinValues='-20,-20,-20',
                    MaxValues='20,20,20',
                    Uproj='1,0,0',
                    Vproj='0,1,0',
                    Wproj='0,0,1')

    FindPeaksMD(InputWorkspace=omd, 
                PeakDistanceThreshold=0.5, 
                DensityThresholdFactor=1000, 
                MaxPeaks=400, 
                OutputType='Peak',
                OutputWorkspace=opk)

    CentroidPeaksMD(InputWorkspace=omd, 
                    PeaksWorkspace=opk, 
                    PeakRadius=0.1, 
                    OutputWorkspace=opk)

    IntegratePeaksMD(InputWorkspace=omd,
                     PeakRadius=0.11,
                     BackgroundInnerRadius=0.11,
                     BackgroundOuterRadius=0.13,
                     PeaksWorkspace=opk,
                     OutputWorkspace=opk,
                     IntegrateIfOnEdge=True)

    FilterPeaks(InputWorkspace=opk,
                OutputWorkspace=opk,
                FilterVariable='Intensity',
                FilterValue=10,  #use 20 for magnet, and 30 for CCR
                Operator='>')

    FilterPeaks(InputWorkspace=opk,
                OutputWorkspace=opk,
                FilterVariable='Signal/Noise',
                FilterValue=10,  
                Operator='>')        

    al_peaks = [0.78, 0.825, 0.905,0.93,1.01, 1.165, 1.215, 1.43, 2.015, 2.28, 2.32]
    peaks_on_Al_ring = []
    
    for j in range(0,len(al_peaks)):
        d_low   = al_peaks[j]-0.01
        d_upper = al_peaks[j]+0.01
        for pn in range(mtd[opk].getNumberPeaks()):
            pk = mtd[opk].getPeak(pn)
            if (pk.getDSpacing() >= round(d_low,3) and pk.getDSpacing() <= round(d_upper,3)):
                peaks_on_Al_ring.append(pn)

    print('number of peak before remove powder ring is {:}'.format(mtd[opk].getNumberPeaks()))
    DeleteTableRows(TableWorkspace=opk, Rows=peaks_on_Al_ring)
    print('number of peak before after powder ring is {:}'.format(mtd[opk].getNumberPeaks()))

    FindUBUsingFFT(PeaksWorkspace=opk, MinD=3, MaxD=20)
    IndexPeaks(PeaksWorkspace=opk, Tolerance=0.125, RoundHKLs=False)
    ShowPossibleCells(PeaksWorkspace=opk, MaxScalarError=0.125)

    FilterPeaks(InputWorkspace=opk, 
                FilterVariable='h^2+k^2+l^2', 
                FilterValue=0, 
                Operator='>', 
                OutputWorkspace=opk)

    SaveNexus(InputWorkspace=opk, Filename=directory+output+opk+'.nxs')
    SaveIsawUB(InputWorkspace=opk, Filename=directory+output+opk+'.mat')

    CombinePeaksWorkspaces(LHSWorkspace=opk, 
                           RHSWorkspace='peaks', 
                           OutputWorkspace='peaks')

if not mtd.doesExist('data'):
    data = GroupWorkspaces(ows_to_merge)
if not mtd.doesExist('md'):
    md = GroupWorkspaces(omd_to_merge)
if not mtd.doesExist('pk'):
    pk = GroupWorkspaces(opk_to_merge)

# FindUBUsingFFT(PeaksWorkspace='peaks', MinD=3, MaxD=20)
# IndexPeaks(PeaksWorkspace='peaks', Tolerance=0.125, RoundHKLs=False)
# ShowPossibleCells(PeaksWorkspace='peaks', MaxScalarError=0.125)
# 
# SelectCellOfType(PeaksWorkspace='peaks', 
#                  CellType=cell_type,
#                  Centering=centering,
#                  Apply=True, 
#                  Tolerance=0.125)
# 
# OptimizeLatticeForCellType(PeaksWorkspace='peaks', 
#                            Tolerance=0.125,
#                            CellType=cell_type, 
#                            Apply=True)