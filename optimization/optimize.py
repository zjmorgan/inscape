from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

import multiprocessing

from mantid.geometry import PointGroupFactory, SpaceGroupFactory

filename, n_proc = sys.argv[1], int(sys.argv[2])

if n_proc > os.cpu_count():
    n_proc = os.cpu_count()

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

import imp
import parameters

imp.reload(parameters)

dictionary = parameters.load_input_file(filename)

run_nos = dictionary['runs'] if type(dictionary['runs']) is list else [dictionary['runs']]

if len(run_nos) < n_proc:
    n_proc = len(run_nos)

facility, instrument = parameters.set_instrument(dictionary['instrument'])
ipts = dictionary['ipts']

working_directory = '/{}/{}/IPTS-{}/shared/'.format(facility,instrument,ipts)
shared_directory = '/{}/{}/shared/'.format(facility,instrument)

directory = os.path.dirname(os.path.abspath(filename))
outname = dictionary['name']

outdir = os.path.join(directory, outname)
if not os.path.exists(outdir):
    os.mkdir(outdir)

parameters.output_input_file(filename, directory, outname+'_opt')

if dictionary.get('tube-file') is not None:
    tube_calibration = os.path.join(shared_directory+'calibration', dictionary['tube-file'])
else:
    tube_calibration = None

if dictionary['detector-file'] is not None:
    detector_calibration = os.path.join(shared_directory+'calibration', dictionary['detector-file'])
else:
    detector_calibration = None

a = dictionary.get('a')
b = dictionary.get('b')
c = dictionary.get('c')
alpha = dictionary.get('alpha')
beta = dictionary.get('beta')
gamma = dictionary.get('gamma')
    
cell_type = dictionary.get('cell-type')
centering = dictionary.get('reflection-condition')

if cell_type == 'cubic':
    cell_type = 'Cubic'
elif cell_type == 'hexagonal':
    cell_type = 'Hexagonal'
elif cell_type == 'rhombohedral':
    cell_type = 'Rhombohedral'
elif cell_type == 'tetragonal':
    cell_type = 'Tetragonal'
elif cell_type == 'orthorhombic':
    cell_type = 'Orthorhombic'
elif cell_type == 'monoclinic':
    cell_type = 'Monoclinic'
elif cell_type == 'triclinic':
    cell_type = 'Triclinic'
    
if centering == 'P':
    reflection_condition = 'Primitive'
elif centering == 'F':
    reflection_condition = 'All-face centred'
elif centering == 'I':
    reflection_condition = 'Body centred'
elif centering == 'A':
    reflection_condition = 'A-face centred'
elif centering == 'B':
    reflection_condition = 'B-face centred'
elif centering == 'C':
    reflection_condition = 'C-face centred'
elif centering == 'R' or centering == 'Robv':
    reflection_condition = 'Rhombohedrally centred, obverse'
elif centering == 'Rrev':
    reflection_condition = 'Rhombohedrally centred, reverse'
elif centering == 'H':
     reflection_condition = 'Hexagonally centred, reverse'

gon_axis = 'BL9:Mot:Sample:Axis3.RBV'

def run_optimization(runs, p, facility, instrument, ipts, detector_calibration, tube_calibration, gon_axis, directory,
                     a, b, c, alpha, beta, gamma, cell_type, centering, reflection_condition):

    if tube_calibration is not None and not mtd.doesExist('tube_table'):
        LoadNexus(Filename=tube_calibration, OutputWorkspace='tube_table')

    for r in runs:

        LoadEventNexus(Filename='/{}/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(facility,instrument,ipts,instrument,r), 
                       OutputWorkspace='data')
                       
        if tube_calibration is not None:
            ApplyCalibration(Workspace='data', CalibrationTable='tube_table')

        if detector_calibration is not None:
            LoadParameterFile(Workspace='data', Filename=detector_calibration)

        if instrument == 'CORELLI':
            SetGoniometer('data', Axis0=str(gon_axis)+',0,1,0,1') 
        else:
            SetGoniometer('data', Goniometers='Universal') 

        min_vals, max_vals = ConvertToMDMinMaxGlobal(InputWorkspace='data',
                                                     QDimensions='Q3D',
                                                     dEAnalysisMode='Elastic',
                                                     Q3DFrames='Q')

        ConvertToMD(InputWorkspace='data', 
                    OutputWorkspace='md', 
                    QDimensions='Q3D',
                    dEAnalysisMode='Elastic',
                    Q3DFrames='Q_sample',
                    LorentzCorrection=True,
                    MinValues=min_vals,
                    MaxValues=max_vals,
                    Uproj='1,0,0',
                    Vproj='0,1,0',
                    Wproj='0,0,1')

        if not mtd.doesExist('peaks'):

            CreatePeaksWorkspace(InstrumentWorkspace='data', 
                                 NumberOfPeaks=0, 
                                 OutputType='Peak', 
                                 OutputWorkspace='peaks')

        FindPeaksMD(InputWorkspace='md', 
                    PeakDistanceThreshold=0.1, 
                    DensityThresholdFactor=10000, 
                    MaxPeaks=400, 
                    OutputType='Peak',
                    OutputWorkspace='pk')

        IntegratePeaksMD(InputWorkspace='md',
                         PeakRadius=0.11,
                         BackgroundInnerRadius=0.11,
                         BackgroundOuterRadius=0.13,
                         PeaksWorkspace='pk',
                         OutputWorkspace='pk',
                         IntegrateIfOnEdge=True)

        FilterPeaks(InputWorkspace='pk',
                    OutputWorkspace='pk',
                    FilterVariable='Intensity',
                    FilterValue=10,
                    Operator='>')

        FilterPeaks(InputWorkspace='pk',
                    OutputWorkspace='pk',
                    FilterVariable='Signal/Noise',
                    FilterValue=10,
                    Operator='>')

        if np.array([a,b,c,alpha,beta,gamma]).all():
            FindUBUsingLatticeParameters(PeaksWorkspace='pk',
                                         a=a,
                                         b=b,
                                         c=c,
                                         alpha=alpha,
                                         beta=beta,
                                         gamma=gamma)
        else:
            FindUBUsingFFT(PeaksWorkspace='pk', MinD=2, MaxD=30)
            IndexPeaks(PeaksWorkspace='pk', Tolerance=0.125, RoundHKLs=True)
            SelectCellOfType(PeaksWorkspace='pk', 
                             CellType=cell_type,
                             Centering=centering,
                             Apply=True)

        IndexPeaks(PeaksWorkspace='pk', Tolerance=0.125, RoundHKLs=False)

        FilterPeaks(InputWorkspace='pk', 
                    FilterVariable='h^2+k^2+l^2', 
                    FilterValue=0, 
                    Operator='>', 
                    OutputWorkspace='pk')

        CombinePeaksWorkspaces(LHSWorkspace='pk', 
                               RHSWorkspace='peaks', 
                               OutputWorkspace='peaks')

        DeleteWorkspace('data')
        DeleteWorkspace('md')
        DeleteWorkspace('pk')

    SaveNexus(Inputworkspace='peaks', Filename=os.path.join(outdir,'peaks_p{}.nxs'.format(p)))

if __name__ == '__main__':

    args = [facility, instrument, ipts, detector_calibration, tube_calibration, gon_axis, directory,
            a, b, c, alpha, beta, gamma, cell_type, centering, reflection_condition]

    split_runs = [split.tolist() for split in np.array_split(run_nos, n_proc)]

    join_args = [(split, i, *args) for i, split in enumerate(split_runs)]

    with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:
        pool.starmap(run_optimization, join_args)
        pool.close()
        pool.join()

    for p in range(n_proc):
        LoadNexus(Filename=os.path.join(outdir, 'peaks_p{}.nxs'.format(p)), OutputWorkspace='tmp')
        if p == 0:
            CloneWorkspace(InputWorkspace='tmp', OutputWorkspace='peaks')
        else:
            CombinePeaksWorkspaces(LHSWorkspace='peaks', RHSWorkspace='tmp', OutputWorkspace='peaks')

    SaveNexus(Inputworkspace='peaks', Filename=os.path.join(outdir,'peaks.nxs'))

    if np.array([a,b,c,alpha,beta,gamma]).all():
            FindUBUsingLatticeParameters(PeaksWorkspace='peaks',
                                         a=a,
                                         b=b,
                                         c=c,
                                         alpha=alpha,
                                         beta=beta,
                                         gamma=gamma)
    else:
        FindUBUsingFFT(PeaksWorkspace='peaks', MinD=3, MaxD=20)
        IndexPeaks(PeaksWorkspace='peaks', Tolerance=0.125, RoundHKLs=True)
        SelectCellOfType(PeaksWorkspace='peaks', 
                         CellType=cell_type,
                         Centering=centering,
                         Apply=True)

    OptimizeLatticeForCellType(PeaksWorkspace='peaks', CellType=cell_type, Apply=True)
    SaveIsawUB(InputWorkspace='peaks', Filename=os.path.join(directory, outname+'_opt.mat'))

    for p in range(n_proc):
        partfile = os.path.join(outdir,'peaks_p{}.nxs'.format(p))
        if os.path.exists(partfile):
            os.remove(partfile)            