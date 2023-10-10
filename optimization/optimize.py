from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

import multiprocess as multiprocessing

from mantid.geometry import PointGroupFactory, SpaceGroupFactory
from mantid.kernel import V3D

import scipy.optimize

from PyPDF2 import PdfFileMerger
from matplotlib.colors import LogNorm
import matplotlib.transforms as mtransforms

filename, n_proc = sys.argv[1], int(sys.argv[2])

#filename, n_proc = '/SNS/CORELLI/IPTS-23019/shared/reduction/optimization/Yb3Al5O12_300K_2022_0311_opt.inp', 1

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

exp = dictionary.get('experiment')

run_nos = dictionary['runs'] if type(dictionary['runs']) is list else [dictionary['runs']]

run_labels = '_'.join([str(r[0])+'-'+str(r[-1]) if type(r) is list else str(r) for r in run_nos if any([type(item) is list for item in run_nos])])

if run_labels == '':
    run_labels = str(run_nos[0])+'-'+str(run_nos[-1])

runs = []
for r in run_nos:
    if type(r) is list:
        runs += r
    else:
        runs += [r]

run_nos = runs

if len(run_nos) < n_proc:
    n_proc = len(run_nos)

facility, instrument = parameters.set_instrument(dictionary['instrument'])
ipts = dictionary['ipts']

working_directory = '/{}/{}/IPTS-{}/shared/'.format(facility,instrument,ipts)
shared_directory = '/{}/{}/shared/'.format(facility,instrument)

directory = os.path.dirname(os.path.abspath(filename))
outname = dictionary['name']

outdir = os.path.join(directory, outname)
dbgdir = os.path.join(outdir, 'debug')
if not os.path.exists(outdir):
    os.mkdir(outdir)
if not os.path.exists(dbgdir):
    os.mkdir(dbgdir)

if dictionary.get('tube-file') is not None:
    tube_calibration = os.path.join(shared_directory+'calibration', dictionary['tube-file'])
else:
    tube_calibration = None

if dictionary.get('detector-file') is not None:
    detector_calibration = os.path.join(shared_directory+'calibration', dictionary['detector-file'])
else:
    detector_calibration = None

a = dictionary.get('a')
b = dictionary.get('b')
c = dictionary.get('c')
alpha = dictionary.get('alpha')
beta = dictionary.get('beta')
gamma = dictionary.get('gamma')

force_constants = dictionary.get('force-constants')
if force_constants is None or not np.all([a,b,c,alpha,beta,gamma]):
    force_constants = False

cell_type = dictionary.get('cell-type').lower()

centering = dictionary.get('centering')
reflection_condition = dictionary.get('reflection-condition')

if cell_type == 'cubic':
    cell_type = 'Cubic'
elif cell_type == 'hexagonal' or cell_type == 'trigonal':
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

if np.any([key in ['P', 'Primitive'] for key in [centering, reflection_condition]]):
    reflection_condition = 'Primitive'    
    centering = 'P'
elif np.any([key in ['F', 'All-face centred'] for key in [centering, reflection_condition]]):
    reflection_condition = 'All-face centred'
    centering = 'F'
elif np.any([key in ['I', 'Body centred'] for key in [centering, reflection_condition]]):
    reflection_condition = 'Body centred'
    centering = 'I'
elif np.any([key in ['A', 'A-face centred'] for key in [centering, reflection_condition]]):
    reflection_condition = 'A-face centred'
    centering = 'A'
elif np.any([key in ['B', 'B-face centred'] for key in [centering, reflection_condition]]):
    reflection_condition = 'B-face centred'
    centering = 'B'
elif np.any([key in ['C', 'C-face centred'] for key in [centering, reflection_condition]]):
    reflection_condition = 'C-face centred'
    centering = 'C'
elif np.any([key in ['R', 'Robv', 'Rhombohedrally centred, obverse'] for key in [centering, reflection_condition]]):
    reflection_condition = 'Rhombohedrally centred, obverse'
    centering = 'R'
elif np.any([key in ['Rrev', 'Rhombohedrally centred, reverse'] for key in [centering, reflection_condition]]):
    reflection_condition = 'Rhombohedrally centred, reverse'
    centering = 'R'
elif np.any([key in ['H', 'Hexagonally centred, reverse'] for key in [centering, reflection_condition]]):
    reflection_condition = 'Hexagonally centred, reverse'
    centering = 'H'

select_cell_type = cell_type
if centering == 'R' and cell_type == 'Hexagonal':
    select_cell_type = 'Rhombohedral'

mod_vector_1 = dictionary.get('modulation-vector-1')
mod_vector_2 = dictionary.get('modulation-vector-2')
mod_vector_3 = dictionary.get('modulation-vector-3')
max_order = dictionary.get('max-order')
cross_terms = dictionary.get('cross-terms')

if mod_vector_1 is None:
    mod_vector_1 = [0,0,0]
if mod_vector_2 is None:
    mod_vector_2 = [0,0,0]
if mod_vector_3 is None:
    mod_vector_3 = [0,0,0]
if max_order is None:
    max_order = 0
if cross_terms is None:
    cross_terms = False
    
if dictionary.get('ub-file') is not None:
    ub_file = os.path.join(working_directory, dictionary['ub-file'])
else:
    ub_file = None

gon_axis = 'BL9:Mot:Sample:Axis3.RBV'

def run_optimization(runs, p, facility, instrument, ipts, detector_calibration, tube_calibration, gon_axis, directory,
                     ub_file, a, b, c, alpha, beta, gamma, force_constants, cell_type, select_cell_type, centering, reflection_condition,
                     mod_vector_1, mod_vector_2, mod_vector_3, max_order, cross_terms):

    if tube_calibration is not None and not mtd.doesExist('tube_table'):
        LoadNexus(Filename=tube_calibration, OutputWorkspace='tube_table')

    if instrument == 'CORELLI':
        k_min, k_max, two_theta_max = 2.5, 10, 148.2
    elif instrument == 'TOPAZ':
        k_min, k_max, two_theta_max = 1.8, 18, 160
    elif instrument == 'MANDI':
        k_min, k_max, two_theta_max = 1.5, 6.3, 160
    elif instrument == 'SNAP':
        k_min, k_max, two_theta_max = 1.8, 12.5, 138

    lamda_min, lamda_max = 2*np.pi/k_max, 2*np.pi/k_min

    Q_max = 4*np.pi/lamda_min*np.sin(np.deg2rad(two_theta_max)/2)
    min_d_spacing = 2*np.pi/Q_max

    max_d = 250
    if np.array([a,b,c,alpha,beta,gamma]).all():
        max_d = np.max([a,b,c])

    for r in runs:

        LoadEventNexus(Filename='/{}/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(facility,instrument,ipts,instrument,r), 
                       OutputWorkspace='data')

        NormaliseByCurrent(InputWorkspace='data', 
                           OutputWorkspace='data',
                           RecalculatePCharge=True)

        if tube_calibration is not None:
            ApplyCalibration(Workspace='data', CalibrationTable='tube_table')

        if detector_calibration is not None:
            if os.path.splitext(detector_calibration)[-1] == '.xml':
                LoadParameterFile(Workspace='data', Filename=detector_calibration)
            else:
                LoadIsawDetCal(InputWorkspace='data', Filename=detector_calibration)

        if instrument == 'CORELLI':
            SetGoniometer(Workspace='data', Axis0=str(gon_axis)+',0,1,0,1') 
        elif instrument == 'SNAP':
            SetGoniometer(Workspace='data', Axis0='omega,0,1,0,1') 
        else:
            SetGoniometer(Workspace='data', Goniometers='Universal') 

        min_vals, max_vals = ConvertToMDMinMaxLocal(InputWorkspace='data',
                                                    QDimensions='Q3D',
                                                    dEAnalysisMode='Elastic',
                                                    Q3DFrames='Q_sample',
                                                    LorentzCorrection=True,
                                                    Uproj='1,0,0',
                                                    Vproj='0,1,0',
                                                    Wproj='0,0,1')

        if not np.isfinite(min_vals).all():
            min_vals = [-20,-20,-20]
        if not np.isfinite(max_vals).all():
            max_vals = [20,20,20]

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

            CreatePeaksWorkspace(NumberOfPeaks=0,
                                 InstrumentWorkspace='data',
                                 OutputType='Peak',
                                 OutputWorkspace='peaks')

        if ub_file is None or np.array([a,b,c,alpha,beta,gamma]).all():

            FindPeaksMD(InputWorkspace='md', 
                        PeakDistanceThreshold=0.1,
                        DensityThresholdFactor=1000 if instrument == 'CORELLI' else 1000, 
                        MaxPeaks=100,
                        OutputType='Peak',
                        OutputWorkspace='pk')

        else:

            LoadIsawUB(InputWorkspace='md', Filename=ub_file)

            max_d_spacing = np.max([mtd['md'].getExperimentInfo(0).sample().getOrientedLattice().d(*hkl) for hkl in [(1,0,0),(0,1,0),(0,0,1)]])

            PredictPeaks(InputWorkspace='md',
                         WavelengthMin=lamda_min,
                         WavelengthMax=lamda_max,
                         MinDSpacing=min_d_spacing,
                         MaxDSpacing=max_d_spacing,
                         OutputType='Peak',
                         ReflectionCondition=reflection_condition if reflection_condition is not None else 'Primitive',
                         OutputWorkspace='pk')

            for _ in range(5):

                CentroidPeaksMD(InputWorkspace='md',
                                PeakRadius=0.1,
                                PeaksWorkspace='pk',
                                OutputWorkspace='pk')

        IntegrateEllipsoids(InputWorkspace='data', 
                            PeaksWorkspace='pk',
                            OutputWorkspace='pk',
                            RegionRadius=0.2,
                            CutoffIsigI=3,
                            NumSigmas=3,
                            IntegrateIfOnEdge=True)

        FilterPeaks(InputWorkspace='pk',
                    OutputWorkspace='pk',
                    FilterVariable='Intensity',
                    FilterValue=0,
                    Operator='>')

        FilterPeaks(InputWorkspace='pk',
                    OutputWorkspace='pk',
                    FilterVariable='Signal/Noise',
                    FilterValue=10,
                    Operator='>')

        FilterPeaks(InputWorkspace='pk',
                    OutputWorkspace='pk',
                    FilterVariable='DSpacing',
                    FilterValue=float('inf'),
                    Operator='<')

        if np.array([a,b,c,alpha,beta,gamma]).all():
            FindUBUsingLatticeParameters(PeaksWorkspace='pk',
                                         a=a,
                                         b=b,
                                         c=c,
                                         alpha=alpha,
                                         beta=beta,
                                         gamma=gamma,
                                         NumInitial=mtd['pk'].getNumberPeaks(),
                                         Iterations=100,
                                         Tolerance=0.1 if not force_constants else 0.1,
                                         FixParameters=force_constants)
        elif ub_file is not None:
            LoadIsawUB(InputWorkspace='md', Filename=ub_file)
            n_indx = IndexPeaks(PeaksWorkspace='pk', Tolerance=0.15, RoundHKLs=True)
        else:
            SortPeaksWorkspace(InputWorkspace='pk', OutputWorkspace='pk', ColumnNameToSortBy='DSpacing', SortAscending=False)
            max_d = mtd['pk'].getPeak(0).getDSpacing()
            FindUBUsingFFT(PeaksWorkspace='pk', MinD=0.5*max_d, MaxD=2*max_d, Iterations=30)
            n_indx = IndexPeaks(PeaksWorkspace='pk', Tolerance=0.15, RoundHKLs=True)
            SelectCellOfType(PeaksWorkspace='pk',
                             CellType=select_cell_type,
                             Centering=centering,
                             Apply=True)

        if ub_file is None or np.array([a,b,c,alpha,beta,gamma]).all():

            CopySample(InputWorkspace='pk',
                       OutputWorkspace='md',
                       CopyName=False,
                       CopyMaterial=False,
                       CopyEnvironment=False,
                       CopyShape=False,
                       CopyLattice=True)

            max_d_spacing = np.max([mtd['pk'].sample().getOrientedLattice().d(*hkl) for hkl in [(1,0,0),(0,1,0),(0,0,1)]])*1.2

            PredictPeaks(InputWorkspace='md',
                         WavelengthMin=lamda_min,
                         WavelengthMax=lamda_max,
                         MinDSpacing=min_d_spacing,
                         MaxDSpacing=max_d_spacing,
                         OutputType='Peak',
                         ReflectionCondition=reflection_condition if reflection_condition is not None else 'Primitive',
                         OutputWorkspace='pk')

            for _ in range(5):
                CentroidPeaksMD(InputWorkspace='md',
                                PeakRadius=0.1,
                                PeaksWorkspace='pk',
                                OutputWorkspace='pk')

            IntegrateEllipsoids(InputWorkspace='data', 
                                PeaksWorkspace='pk',
                                OutputWorkspace='pk',
                                RegionRadius=0.2,
                                CutoffIsigI=3,
                                NumSigmas=3,
                                IntegrateIfOnEdge=True)

            FilterPeaks(InputWorkspace='pk',
                        OutputWorkspace='pk',
                        FilterVariable='Intensity',
                        FilterValue=0,
                        Operator='>')

            FilterPeaks(InputWorkspace='pk',
                        OutputWorkspace='pk',
                        FilterVariable='Signal/Noise',
                        FilterValue=10,
                        Operator='>')

            FilterPeaks(InputWorkspace='pk',
                        OutputWorkspace='pk',
                        FilterVariable='DSpacing',
                        FilterValue=float('inf'),
                        Operator='<')

            n_indx = IndexPeaks(PeaksWorkspace='pk', Tolerance=0.08, RoundHKLs=True)

            FilterPeaks(InputWorkspace='pk',
                        FilterVariable='h^2+k^2+l^2', 
                        FilterValue=0,
                        Operator='>',
                        OutputWorkspace='pk')

            if n_indx[0] > 10:

                if not force_constants:
                    OptimizeLatticeForCellType(PeaksWorkspace='pk',
                                               CellType=cell_type,
                                               PerRun=False,
                                               Apply=True,
                                               OutputDirectory=dbgdir)

                IndexPeaks(PeaksWorkspace='pk',
                           Tolerance=0.08,
                           ToleranceForSatellite=0.08,
                           RoundHKLs=True,
                           CommonUBForAll=False,
                           ModVector1=mod_vector_1,
                           ModVector2=mod_vector_2,
                           ModVector3=mod_vector_3,
                           MaxOrder=max_order,
                           CrossTerms=cross_terms,
                           SaveModulationInfo=True if max_order > 0 else False)

                FilterPeaks(InputWorkspace='pk',
                            FilterVariable='h^2+k^2+l^2', 
                            FilterValue=0,
                            Operator='>',
                            OutputWorkspace='pk')

                SaveIsawUB(InputWorkspace='pk',
                           Filename=os.path.join(dbgdir,'{}_{}_{}_{}.mat'.format(instrument,r,cell_type,centering)))

                CombinePeaksWorkspaces(LHSWorkspace='peaks', 
                                       RHSWorkspace='pk', 
                                       OutputWorkspace='peaks')

                CopySample(InputWorkspace='pk',
                           OutputWorkspace='data',
                           CopyName=False,
                           CopyMaterial=False,
                           CopyEnvironment=False,
                           CopyShape=False,
                           CopyLattice=True)

                ConvertToMD(InputWorkspace='data', 
                            OutputWorkspace='md', 
                            QDimensions='Q3D',
                            dEAnalysisMode='Elastic',
                            Q3DFrames='HKL',
                            LorentzCorrection=True,
                            MinValues='-10,-10,-10',
                            MaxValues='10,10,10',
                            Uproj='1,0,0',
                            Vproj='0,1,0',
                            Wproj='0,0,1')

                ol = mtd['pk'].sample().getOrientedLattice()

                char_dict = {0:'0', 1:'{1}', -1:'-{1}'}
                chars = ['H','K','L']

                Ws = [np.array([[1,0,0],
                                [0,1,0],
                                [0,0,1]]),
                      np.array([[1,0,0],
                                [0,0,1],
                                [0,1,0]]),
                      np.array([[0,0,1],
                                [1,0,0],
                                [0,1,0]]),
                      np.array([[1,0,-1],
                                [1,0,1],
                                [0,1,0]]),
                      np.array([[1,0,-1],
                                [0,1,0],
                                [1,0,1]]),
                      np.array([[0,1,0],
                                [1,0,-1],
                                [1,0,1]]),
                      np.array([[-1,0,1],
                                [1,0,1],
                                [0,1,0]]),
                      np.array([[-1,0,1],
                                [0,1,0],
                                [1,0,1]]),
                      np.array([[0,1,0],
                                [-1,0,1],
                                [1,0,1]])]

                min_vals = np.array([-8,-8,-8])
                max_vals = np.array([8,8,8])
                bins = np.array([512,512,512])

                fig, axs = plt.subplots(3, 3, figsize=[19.2,19.2])
                fig.suptitle('{} {}'.format(instrument,r))

                axs = axs.flatten()

                for j, (ax, W) in enumerate(zip(axs,Ws)):

                    names = ['['+','.join(char_dict.get(j, '{0}{1}').format(j,chars[np.argmax(np.abs(W[:,i]))]) for j in W[:,i])+']' for i in range(3)]

                    BinMD(InputWorkspace='md',
                          OutputWorkspace='slice',
                          AxisAligned=False,
                          BasisVector0='{},r.l.u.,{},{},{}'.format(names[0],*W[:,0]),
                          BasisVector1='{},r.l.u.,{},{},{}'.format(names[1],*W[:,1]),
                          BasisVector2='{},r.l.u.,{},{},{}'.format(names[2],*W[:,2]),
                          OutputExtents='{},{},{},{},-0.05,0.05'.format(min_vals[W[:,0] == 1][0],
                                                                        max_vals[W[:,0] == 1][0],
                                                                        min_vals[W[:,1] == 1][0],
                                                                        max_vals[W[:,1] == 1][0]),
                          OutputBins='{},{},1'.format(bins[W[:,0] == 1][0],bins[W[:,1] == 1][0]),
                          NormalizeBasisVectors=False)

                    data = mtd['slice']

                    dims = [data.getDimension(i) for i in range(3)]

                    dmin = [dim.getMinimum() for dim in dims]
                    dmax = [dim.getMaximum() for dim in dims]

                    labels = [dim.getName().replace(',',' ').replace('[','(').replace(']',')').lower() for dim in dims]

                    signal = data.getSignalArray().copy().squeeze(axis=2)

                    signal[signal <= 0] = np.nan

                    angle = ol.recAngle(*W[:,0].astype(float),*W[:,1].astype(float))

                    transform = mtransforms.Affine2D().skew_deg(90-angle,0)

                    vmin, vmax = np.nanpercentile(signal,2), np.nanpercentile(signal,98)

                    if np.isnan(vmin):
                        vmin = 0.001
                    if np.isnan(vmax):
                        vmax = 1000

                    im = ax.imshow(signal.T, extent=[dmin[0],dmax[0],dmin[1],dmax[1]], origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax, rasterized=True)
                    ax.set_xlabel(labels[0])
                    ax.set_ylabel(labels[1])
                    ax.set_title(labels[2]+' = [-0.05,0.05]')
                    ax.minorticks_on()

                    ax.grid(which='both', alpha=0.5, transform=transform)
                    ax.xaxis.get_major_locator().set_params(integer=True)
                    ax.yaxis.get_major_locator().set_params(integer=True)

                    trans_data = transform+ax.transData
                    im.set_transform(trans_data)

                    cb = fig.colorbar(im, ax=ax)
                    cb.ax.minorticks_on()

                fig.savefig(os.path.join(dbgdir,'{}_{}_{}_{}.pdf'.format(instrument,r,cell_type,centering)))
                plt.close()

        DeleteWorkspace('data')
        DeleteWorkspace('md')
        DeleteWorkspace('pk')

    SaveNexus(InputWorkspace='peaks', Filename=os.path.join(dbgdir, 'peaks_p{}.nxs'.format(p)))

if __name__ == '__main__':

    parameters.output_input_file(filename, directory, outname+'_opt')

    if instrument == 'CORELLI':
        k_min, k_max, two_theta_max = 2.5, 10, 148.2
    elif instrument == 'TOPAZ':
        k_min, k_max, two_theta_max = 1.8, 18, 160
    elif instrument == 'MANDI':
        k_min, k_max, two_theta_max = 1.5, 6.3, 160
    elif instrument == 'SNAP':
        k_min, k_max, two_theta_max = 1.8, 12.5, 138

    lamda_min, lamda_max = 2*np.pi/k_max, 2*np.pi/k_min

    Q_max = 4*np.pi/lamda_min*np.sin(np.deg2rad(two_theta_max)/2)
    min_d_spacing = 2*np.pi/Q_max

    max_d = 250
    if np.array([a,b,c,alpha,beta,gamma]).all():
        max_d = np.max([a,b,c])

    #if not os.path.exists(os.path.join(dbgdir,'peaks.nxs')):

    args = [facility, instrument, ipts, detector_calibration, tube_calibration, gon_axis, directory,
            ub_file, a, b, c, alpha, beta, gamma, force_constants, cell_type, select_cell_type, centering, reflection_condition,
            mod_vector_1, mod_vector_2, mod_vector_3, max_order, cross_terms]

    split_runs = [split.tolist() for split in np.array_split(run_nos, n_proc)]

    join_args = [(split, i, *args) for i, split in enumerate(split_runs)]

    if not True: # os.path.exists(os.path.join(dbgdir,'peaks.nxs')):

        LoadNexus(Filename=os.path.join(dbgdir,'peaks.nxs'), OutputWorkspace='peak')

    else:

        config['MultiThreaded.MaxCores'] == 1
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['_SC_NPROCESSORS_ONLN'] = '1'

        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:
            pool.starmap(run_optimization, join_args)
            pool.close()
            pool.join()

        # run_optimization(*join_args[0])

        config['MultiThreaded.MaxCores'] == 4
        os.environ.pop('OPENBLAS_NUM_THREADS', None)
        os.environ.pop('OMP_NUM_THREADS', None)
        os.environ.pop('_SC_NPROCESSORS_ONLN', None)

        #run_optimization(*join_args)

        for p in range(n_proc):
            LoadNexus(Filename=os.path.join(dbgdir, 'peaks_p{}.nxs'.format(p)), OutputWorkspace='pk')
            FilterPeaks(InputWorkspace='pk',
                        OutputWorkspace='pk',
                        FilterVariable='DSpacing',
                        FilterValue=float('inf'),
                        Operator='<')
            if p == 0:
                CloneWorkspace(InputWorkspace='pk', OutputWorkspace='peak')
            else: 
                CombinePeaksWorkspaces(LHSWorkspace='peak', RHSWorkspace='pk', OutputWorkspace='peak')
            DeleteWorkspace('pk')

        SaveNexus(InputWorkspace='peak', Filename=os.path.join(dbgdir,'peaks.nxs'))

        for p in range(n_proc):
            os.remove(os.path.join(dbgdir, 'peaks_p{}.nxs'.format(p)))

    if cell_type == 'Triclinic':
        bl = PointGroupFactory.createPointGroup('-1')
    elif cell_type == 'Monoclinic':
        bl = PointGroupFactory.createPointGroup('2/m')
    elif cell_type == 'Orthorhombic':
        bl = PointGroupFactory.createPointGroup('mmm')
    elif cell_type == 'Tetragonal':
        bl = PointGroupFactory.createPointGroup('4/mmm')
    elif cell_type == 'Rhombohedral':
        bl = PointGroupFactory.createPointGroup('-3m')
    elif cell_type == 'Hexagonal':
        bl = PointGroupFactory.createPointGroup('6/mmm')
    elif cell_type == 'Cubic':
        bl = PointGroupFactory.createPointGroup('m-3m')

    Ws = []
    for op in bl.getSymmetryOperations():
        W = np.column_stack([op.transformHKL(vec) for vec in ([1,0,0],[0,1,0],[0,0,1])])
        Ws.append(W)

    runs = list(set(mtd['peak'].column(0)))

    for run in runs:
        FilterPeaks(InputWorkspace='peak',
                    FilterVariable='RunNumber',
                    FilterValue=run,
                    Operator='=',
                    OutputWorkspace='peak_{}'.format(run))

    CreatePeaksWorkspace(InstrumentWorkspace='peak',  
                         NumberOfPeaks=0,
                         OutputWorkspace='test')

    if ub_file is not None:
        LoadIsawUB(InputWorkspace='test', Filename=ub_file)
    else:
        LoadIsawUB(InputWorkspace='test',
                   Filename=os.path.join(dbgdir,'{}_{}_{}_{}.mat'.format(instrument,runs[0],cell_type,centering)))

    U_ref = mtd['test'].sample().getOrientedLattice().getU().copy()

    Rs = []

    for r in enumerate(runs):
        R = mtd['peak_{}'.format(run)].getPeak(0).getGoniometerMatrix().copy()
        Rs.append(R)

    for i, run in enumerate(runs):

        for pk in mtd['peak_{}'.format(run)]:
            pk.setGoniometerMatrix(np.eye(3))

#         FindUBUsingIndexedPeaks(PeaksWorkspace='peak_{}'.format(run))
#         U = mtd['peak_{}'.format(run)].sample().getOrientedLattice().getU().copy()
#         B = mtd['peak_{}'.format(run)].sample().getOrientedLattice().getB().copy()
# 
#         for pk in mtd['peak_{}'.format(run)]:
#             pk.setGoniometerMatrix(np.eye(3))
# 
#         Us = []
#         for j, W in enumerate(Ws):
#             CloneWorkspace(InputWorkspace='peak_{}'.format(run),
#                            OutputWorkspace='tmp_{}'.format(j))
# 
#             for pk in mtd['tmp_{}'.format(j)]:
#                 HKL = np.dot(W, pk.getIntHKL())
#                 pk.setIntHKL(V3D(*HKL))
#                 pk.setHKL(*HKL)
# 
#             FindUBUsingIndexedPeaks(PeaksWorkspace='tmp_{}'.format(j))
#             U = mtd['tmp_{}'.format(j)].sample().getOrientedLattice().getU().copy()
#             Us.append(U)
# 
#         norm = []
#         for U in Us:
#             norm.append(np.linalg.norm(U-U_ref))
# 
#         j = np.argmin(norm)
# 
#         CloneWorkspace(InputWorkspace='tmp_{}'.format(j),
#                        OutputWorkspace='peak_{}'.format(run))

        FindUBUsingIndexedPeaks(PeaksWorkspace='peak_{}'.format(run))

        U = mtd['peak_{}'.format(run)].sample().getOrientedLattice().getU().copy()
        for pk in mtd['peak_{}'.format(run)]:
            pk.setGoniometerMatrix(U)

        CombinePeaksWorkspaces(LHSWorkspace='test',
                               RHSWorkspace='peak_{}'.format(run),
                               OutputWorkspace='test')

    #FindUBUsingIndexedPeaks(PeaksWorkspace='test')
    OptimizeLatticeForCellType(PeaksWorkspace='test',
                               CellType=cell_type,
                               PerRun=False,
                               Apply=True,
                               OutputDirectory=dbgdir)

    CloneWorkspace(InputWorkspace='test',
                   OutputWorkspace='tmp')

    IndexPeaks(PeaksWorkspace='test',
               RoundHKLs=False,
               CommonUBForAll=True)

    SaveNexus(InputWorkspace='test',
              Filename=os.path.join(outdir, 'peaks_opt.nxs'))

    SaveIsawUB(InputWorkspace='test',
               Filename=os.path.join(outdir, '{}_{}_{}_opt.mat'.format(instrument,cell_type,centering)))

    ol = mtd['test'].sample().getOrientedLattice()

    a, b, c, alpha, beta, gamma = ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma()

    for i, r in enumerate(runs):

        FilterPeaks(InputWorkspace='test',
                    FilterVariable='RunNumber',
                    FilterValue=r,
                    Operator='=',
                    OutputWorkspace='peak_{}'.format(r))

        R = Rs[i]

        for pk in mtd['peak_{}'.format(r)]:
            pk.setGoniometerMatrix(np.eye(3))

        CalculateUMatrix(PeaksWorkspace='peak_{}'.format(r),
                         a=a,
                         b=b,
                         c=c,
                         alpha=alpha,
                         beta=beta,
                         gamma=gamma)

        ol = mtd['peak_{}'.format(r)].sample().getOrientedLattice()

        RUB = ol.getUB()

        UB = np.dot(R.T, RUB)

        for pk in mtd['peak_{}'.format(r)]:
            pk.setGoniometerMatrix(R)

        SetUB(Workspace='peak_{}'.format(r), UB=UB)

        IndexPeaks(PeaksWorkspace='peak_{}'.format(r),
                   RoundHKLs=True,
                   CommonUBForAll=True)

        SaveIsawUB(InputWorkspace='peak_{}'.format(r),
                   Filename=os.path.join(dbgdir, '{}_{}_{}_{}_opt.mat'.format(instrument,r,cell_type,centering)))