# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from corelli.calibration import load_calibration_set, apply_calibration
from corelli.calibration.powder import load_and_rebin

import sys 
import os 

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/home/zgf/.git/inscape/integration/')

# directories ------------------------------------------------------------------
filename = '/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'

# binning paramteres -----------------------------------------------------------
bin_param = '0.5,0.01,11.5'

# calibration files -----------------------------------------------------------
detector_calibration = None
tube_calibration = None

if tube_calibration is not None:
    LoadNexus(Filename=tube_calibration,
              OutputWorkspace='tube_table')

# calibration runs -------------------------------------------------------------

instrument = 'MANDI'
ipts = 8776

# garnet
start = 10646
stop = 10682

step = 1

ub_file = None# '/SNS/MANDI/IPTS-8776/shared/garnet/2022A/optimizied_UB.mat'

# lattice information ----------------------------------------------------------
cell_type = 'Cubic'
centering = 'I'
reflection_condition = 'Body centred'

# goniometer axis --------------------------------------------------------------
gon_axis = 'BL9:Mot:Sample:Axis3.RBV'

# ------------------------------------------------------------------------------

runs = np.arange(start,stop+1,step)

toMerge1 = []
toMerge2 = []
toMerge3 = []

for r in runs:
    print('Processing run : %s' %r)
    ows = '{}_{}'.format(instrument,r)
    omd = ows+'_md'
    opk = ows+'_pk'

    if not mtd.doesExist(omd):
        toMerge2.append(omd)
        if not mtd.doesExist(ows):
            toMerge1.append(ows)
            toMerge3.append(opk)
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
            SetGoniometer(ows, Axis0=str(omega)+',0,1,0,1') 
        else:
            SetGoniometer(ows, Goniometers='Universal') 

        if ub_file is not None:
            LoadIsawUB(InputWorkspace=ows, Filename=ub_file)

        ConvertToMD(InputWorkspace=ows, 
                    OutputWorkspace=omd, 
                    QDimensions='Q3D',
                    dEAnalysisMode='Elastic',
                    Q3DFrames='Q_sample' if ub_file is None else 'HKL',
                    LorentzCorrection=True,
                    MinValues='-20,-20,-20',
                    MaxValues='20,20,20',
                    Uproj='1,0,0',
                    Vproj='0,1,0',
                    Wproj='0,0,1')

        if ub_file is not None:
            PredictPeaks(InputWorkspace=omd, 
                         WavelengthMin=0.4, 
                         WavelengthMax=4,
                         MinDSpacing=0.7,
                         MaxDSpacing=20,
                         ReflectionCondition=reflection_condition,
                         OutputType='Peak',
                         OutputWorkspace=opk)
        else:
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

            FindUBUsingFFT(PeaksWorkspace=opk, MinD=3, MaxD=20)
            IndexPeaks(PeaksWorkspace=opk, Tolerance=0.125, RoundHKLs=False)
            ShowPossibleCells(PeaksWorkspace=opk, MaxScalarError=0.125)

            FilterPeaks(InputWorkspace=opk, 
                        FilterVariable='h^2+k^2+l^2', 
                        FilterValue=0, 
                        Operator='>', 
                        OutputWorkspace=opk)

            CombinePeaksWorkspaces(LHSWorkspace=opk, 
                                   RHSWorkspace='peaks', 
                                   OutputWorkspace='peaks')

data = GroupWorkspaces(toMerge1)
md = GroupWorkspaces(toMerge2)
pk = GroupWorkspaces(toMerge3)

FindUBUsingFFT(PeaksWorkspace='peaks', MinD=3, MaxD=20)
IndexPeaks(PeaksWorkspace='peaks', Tolerance=0.125, RoundHKLs=False)
ShowPossibleCells(PeaksWorkspace='peaks', MaxScalarError=0.125)

SelectCellOfType(PeaksWorkspace='peaks', 
                 CellType=cell_type,
                 Centering=centering,
                 Apply=True, 
                 Tolerance=0.125)

OptimizeLatticeForCellType(PeaksWorkspace='peaks', 
                           Tolerance=0.125,
                           CellType=cell_type, 
                           Apply=True)

# SaveIsawUB('peaks', Filename=os.path.join(directory, 'calibration_{}-{}.mat'.format(start,stop)))