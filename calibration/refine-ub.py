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
iptsfolder = '/SNS/CORELLI/IPTS-23019/'
nxfiledir = iptsfolder+'nexus/'
ccfiledir = iptsfolder+'shared/autoreduce/'

# binning paramteres -----------------------------------------------------------
bin_param = '0.5,0.01,11.5'

# calibration files -----------------------------------------------------------
detector_calibration = '/SNS/CORELLI/IPTS-23019/shared/germanium_2021b/germanium_2021B_corrected.xml'
tube_calibration = '/SNS/CORELLI/shared/calibration/tube/calibration_corelli_20200109.nxs.h5'

LoadNexus(Filename=tube_calibration, OutputWorkspace='tube_table')

# calibration runs -------------------------------------------------------------

ipts = 23019

# garnet
start = 221998
stop = 222172

step = 1

# lattice information ----------------------------------------------------------
cell_type = 'Cubic'
centering = 'I'

# use elastic ------------------------------------------------------------------
LoadCC = False

# goniometer axis --------------------------------------------------------------
gon_axis = 'BL9:Mot:Sample:Axis3.RBV'

# ------------------------------------------------------------------------------

runs = np.arange(start,stop+1,step)

toMerge1 = []
toMerge2 = []
toMerge3 = []

lat = []

for r in runs:
    print('Processing run : %s' %r)
    ows = 'COR_'+str(r)
    omd = ows+'_md'
    opk = ows+'_pks'

    if not mtd.doesExist(omd):
        toMerge2.append(omd)
        if not mtd.doesExist(ows):
            toMerge1.append(ows)
            toMerge3.append(opk)
            if LoadCC :
                filename = ccfiledir+'CORELLI_'+str(r)+'_elastic.nxs'
                if not mtd.doesExist(ows):
                    LoadNexus(Filename=filename, OutputWorkspace=ows)
            else:
                filename = nxfiledir+'CORELLI_'+str(r)+'.nxs.h5'
                if not mtd.doesExist(ows):
                    LoadEventNexus(Filename=filename, OutputWorkspace=ows) #
                                      
        ApplyCalibration(Workspace=ows, CalibrationTable='tube_table')
        LoadParameterFile(Workspace=ows, Filename=detector_calibration)

        if (r == runs[0]):
            CreatePeaksWorkspace(InstrumentWorkspace=ows, 
                                 NumberOfPeaks=0, 
                                 OutputType='Peak', 
                                 OutputWorkspace='peaks')

        owshandle = mtd[ows]
        lrun = owshandle.getRun()
        pclog = lrun.getLogData('proton_charge')
        pc = sum(pclog.value)/1e12

        print('the current proton charge :'+ str(pc))

        omega = owshandle.getRun().getLogData(gon_axis).value.mean()
        SetGoniometer(ows, Axis0=str(omega)+',0,1,0,1') 

        ConvertToMD(InputWorkspace=ows, 
                    OutputWorkspace=omd, 
                    QDimensions='Q3D',
                    dEAnalysisMode='Elastic',
                    Q3DFrames='Q_sample',
                    LorentzCorrection=1,
                    MinValues='-20,-20,-20',
                    MaxValues='20,20,20',
                    Uproj='1,0,0',
                    Vproj='0,1,0',
                    Wproj='0,0,1', 
                    SplitInto=2, 
                    SplitThreshold=50, 
                    MaxRecursionDepth=13, 
                    MinRecursionDepth=7)
                    
        FindPeaksMD(InputWorkspace=omd, 
                    PeakDistanceThreshold=0.5, 
                    DensityThresholdFactor=10000, 
                    MaxPeaks=400, 
                    OutputType='Peak',
                    OutputWorkspace=opk)
                    
        CentroidPeaksMD(InputWorkspace=omd, 
                        PeaksWorkspace=opk, 
                        PeakRadius=0.15, 
                        OutputWorkspace=opk)

        FindUBUsingFFT(PeaksWorkspace=opk, MinD=3, MaxD=20)
        IndexPeaks(PeaksWorkspace=opk, Tolerance=0.125, RoundHKLs=False)
        ShowPossibleCells(PeaksWorkspace=opk, MaxScalarError=0.125)
                                    
        FilterPeaks(InputWorkspace=opk, 
                    FilterVariable='h^2+k^2+l^2', 
                    FilterValue=0, 
                    Operator='>', 
                    OutputWorkspace=opk)
                                    
        lat.append(mtd[opk].sample().getOrientedLattice())
                  
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
                           
SaveIsawUB('peaks', Filename=os.path.join(directory,'calibration_{}-{}.mat'.format(start,stop)))