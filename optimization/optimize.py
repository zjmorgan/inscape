from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

import multiprocess as multiprocessing

from mantid.geometry import PointGroupFactory, SpaceGroupFactory

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

def __U_matrix(phi, theta, omega):

    ux = np.cos(phi)*np.sin(theta)
    uy = np.sin(phi)*np.sin(theta)
    uz = np.cos(theta)

    U = np.array([[np.cos(omega)+ux**2*(1-np.cos(omega)), ux*uy*(1-np.cos(omega))-uz*np.sin(omega), ux*uz*(1-np.cos(omega))+uy*np.sin(omega)],
                  [uy*ux*(1-np.cos(omega))+uz*np.sin(omega), np.cos(omega)+uy**2*(1-np.cos(omega)), uy*uz*(1-np.cos(omega))-ux*np.sin(omega)],
                  [uz*ux*(1-np.cos(omega))-uy*np.sin(omega), uz*uy*(1-np.cos(omega))+ux*np.sin(omega), np.cos(omega)+uz**2*(1-np.cos(omega))]])

    return U

def __B_matrix(a, b, c, alpha, beta, gamma):

    G = np.array([[a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
                  [b*a*np.cos(gamma), b**2, b*c*np.cos(alpha)],
                  [c*a*np.cos(beta), c*b*np.cos(alpha), c**2]])

    B = scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

    return B

def __cub(x):

    a, *params = x

    return (a, a, a, np.pi/2, np.pi/2, np.pi/2, *params)

def __rhom(x):

    a, alpha, *params = x

    return (a, a, a, alpha, alpha, alpha, *params)

def __tet(x):

    a, c, *params = x

    return (a, a, c, np.pi/2, np.pi/2, np.pi/2, *params)

def __hex(x):

    a, c, *params = x

    return (a, a, c, np.pi/2, np.pi/2, 2*np.pi/3, *params)

def __ortho(x):

    a, b, c, *params = x

    return (a, b, c, np.pi/2, np.pi/2, np.pi/2, *params)

def __mono1(x):

    a, b, c, beta, *params = x

    return (a, b, c, np.pi/2, beta, np.pi/2, *params)

def __mono2(x):

    a, b, c, gamma, *params = x

    return (a, b, c, np.pi/2, np.pi/2, gamma, *params)

def __tri(x):

    a, b, c, alpha, beta, gamma, *params = x

    return (a, b, c, alpha, beta, gamma, *params)

def __res(x, hkls, Qs, fun):

    a, b, c, alpha, beta, gamma, *angles = fun(x)

    B = __B_matrix(a, b, c, alpha, beta, gamma)

    diff = []

    for i, (hkl, Q) in enumerate(zip(hkls,Qs)):

        if len(hkl) > 0:

            omega = angles[3*i+0]
            theta = angles[3*i+1]
            phi = angles[3*i+2]

            U = __U_matrix(phi, theta, omega)

            UB = np.dot(U,B)

            diff += (np.einsum('ij,lj->li', UB, hkl)*2*np.pi-Q).flatten().tolist()

    return np.array(diff)

def run_optimization(runs, p, facility, instrument, ipts, detector_calibration, tube_calibration, gon_axis, directory,
                     ub_file, a, b, c, alpha, beta, gamma, cell_type, select_cell_type, centering, reflection_condition,
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

        min_vals, max_vals = ConvertToMDMinMaxGlobal(InputWorkspace='data',
                                                     QDimensions='Q3D',
                                                     dEAnalysisMode='Elastic',
                                                     Q3DFrames='Q')

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

        if ub_file is not None:

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

            CentroidPeaksMD(InputWorkspace='md',
                            PeakRadius=0.1,
                            PeaksWorkspace='pk',
                            OutputWorkspace='pk')

            CentroidPeaksMD(InputWorkspace='md',
                            PeakRadius=0.1,
                            PeaksWorkspace='pk',
                            OutputWorkspace='pk')

        else:

            # min_Q = 2*np.pi/max_d

            FindPeaksMD(InputWorkspace='md', 
                        PeakDistanceThreshold=0.1,
                        DensityThresholdFactor=5000, 
                        MaxPeaks=400,
                        OutputType='Peak',
                        OutputWorkspace='pk')

        IntegrateEllipsoids(InputWorkspace='data', 
                            PeaksWorkspace='pk',
                            OutputWorkspace='pk',
                            RegionRadius=0.25,
                            CutoffIsigI=1,
                            NumSigmas=3,
                            IntegrateIfOnEdge=True)

        # IntegratePeaksMD(InputWorkspace='md',
        #                  PeakRadius=0.11,
        #                  BackgroundInnerRadius=0.11,
        #                  BackgroundOuterRadius=0.13,
        #                  PeaksWorkspace='pk',
        #                  OutputWorkspace='pk',
        #                  IntegrateIfOnEdge=True)

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

        if mtd['pk'].getNumberPeaks() > 10:

            if ub_file is not None:
                LoadIsawUB(InputWorkspace='md', Filename=ub_file)
                n_indx = IndexPeaks(PeaksWorkspace='pk', Tolerance=0.15, RoundHKLs=True)
            elif np.array([a,b,c,alpha,beta,gamma]).all():
                FindUBUsingLatticeParameters(PeaksWorkspace='pk',
                                             a=a,
                                             b=b,
                                             c=c,
                                             alpha=alpha,
                                             beta=beta,
                                             gamma=gamma,
                                             Tolerance=0.15,
                                             FixParameters=False)
                n_indx = IndexPeaks(PeaksWorkspace='pk', Tolerance=0.15, RoundHKLs=True)
            else:
                SortPeaksWorkspace(InputWorkspace='pk', OutputWorkspace='pk', ColumnNameToSortBy='DSpacing', SortAscending=False)
                max_d = mtd['pk'].getPeak(0).getDSpacing()
                FindUBUsingFFT(PeaksWorkspace='pk', MinD=0.25*max_d, MaxD=4*max_d, Iterations=15)
                n_indx = IndexPeaks(PeaksWorkspace='pk', Tolerance=0.15, RoundHKLs=True)
                SelectCellOfType(PeaksWorkspace='pk',
                                 CellType=select_cell_type,
                                 Centering=centering,
                                 Apply=True)

            if n_indx[0] > 10:

                OptimizeLatticeForCellType(PeaksWorkspace='pk',
                                           CellType=cell_type,
                                           PerRun=False,
                                           Apply=True,
                                           OutputDirectory=dbgdir)

                IndexPeaks(PeaksWorkspace='pk',
                           Tolerance=0.15,
                           ToleranceForSatellite=0.15,
                           RoundHKLs=False,
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

                # FindUBUsingIndexedPeaks('pk', Tolerance=0.15, ToleranceForSatellite=0.15, CommonUBForAll=False)

                # FilterPeaks(InputWorkspace='pk',
                #             FilterVariable='h^2+k^2+l^2',
                #             FilterValue=0,
                #             Operator='>',
                #             OutputWorkspace='pk')

                CombinePeaksWorkspaces(LHSWorkspace='peaks', 
                                       RHSWorkspace='pk', 
                                       OutputWorkspace='peaks')

                # FilterPeaks(InputWorkspace='pk',
                #             FilterVariable='DSpacing', 
                #             FilterValue=min_d_spacing,
                #             Operator='>',
                #             OutputWorkspace='pk')

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
                          OutputExtents='{},{},{},{},-0.05,0.05'.format(min_vals[W[:,0] == 1][0],max_vals[W[:,0] == 1][0],min_vals[W[:,1] == 1][0],max_vals[W[:,1] == 1][0]),
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

                    #ax.set_aspect((dmax[0]-dmin[0])/(dmax[1]-dmin[1]))
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

    if facility == 'SNS':

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
                ub_file, a, b, c, alpha, beta, gamma, cell_type, select_cell_type, centering, reflection_condition,
                mod_vector_1, mod_vector_2, mod_vector_3, max_order, cross_terms]

        split_runs = [split.tolist() for split in np.array_split(run_nos, n_proc)]

        join_args = [(split, i, *args) for i, split in enumerate(split_runs)]

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
                CloneWorkspace(InputWorkspace='pk', OutputWorkspace='peaks')
            else: 
                CombinePeaksWorkspaces(LHSWorkspace='peaks', RHSWorkspace='pk', OutputWorkspace='peaks')
            DeleteWorkspace('pk')

        SaveNexus(InputWorkspace='peaks', Filename=os.path.join(dbgdir,'peaks.nxs'))

        merger = PdfFileMerger()

        for r in run_nos:
            partfile = os.path.join(dbgdir,'{}_{}_{}_{}.pdf'.format(instrument,r,cell_type,centering))
            if os.path.exists(partfile):
                merger.append(partfile)

        merger.write(os.path.join(outdir,'{}_{}_opt.pdf'.format(cell_type,centering)))       
        merger.close()

        for r in run_nos:
            partfile = os.path.join(dbgdir,'{}_{}_{}_{}.pdf'.format(instrument,r,cell_type,centering))
            if os.path.exists(partfile):
                os.remove(partfile)
 
#         else:
# 
#             LoadNexus(OutputWorkspace='peaks', Filename=os.path.join(dbgdir, 'peaks.nxs'))
# 
        if ub_file is not None:
            FindUBUsingIndexedPeaks('peaks', Tolerance=0.15, ToleranceForSatellite=0.15, CommonUBForAll=False)
        elif np.array([a,b,c,alpha,beta,gamma]).all():
            FindUBUsingLatticeParameters(PeaksWorkspace='peaks',
                                         a=a,
                                         b=b,
                                         c=c,
                                         alpha=alpha,
                                         beta=beta,
                                         gamma=gamma)
            IndexPeaks(PeaksWorkspace='peaks', Tolerance=0.15, RoundHKLs=True)
        else:
            SortPeaksWorkspace(InputWorkspace='peaks', OutputWorkspace='peaks', ColumnNameToSortBy='DSpacing', SortAscending=False)
            max_d = mtd['peaks'].getPeak(0).getDSpacing()
            FindUBUsingFFT(PeaksWorkspace='peaks', Tolerance=0.15, MinD=0.25*max_d, MaxD=4*max_d, Iterations=150)
            IndexPeaks(PeaksWorkspace='peaks', Tolerance=0.15, RoundHKLs=True)
            SelectCellOfType(PeaksWorkspace='peaks', 
                             CellType=select_cell_type,
                             Centering=centering,
                             Apply=True)

        IndexPeaks(PeaksWorkspace='peaks',
                   Tolerance=0.15,
                   ToleranceForSatellite=0.15,
                   RoundHKLs=False,
                   CommonUBForAll=False,
                   ModVector1=mod_vector_1,
                   ModVector2=mod_vector_2,
                   ModVector3=mod_vector_3,
                   MaxOrder=max_order,
                   CrossTerms=cross_terms,
                   SaveModulationInfo=True if max_order > 0 else False)

        common_ind = True

        for r in run_nos:

            FilterPeaks(InputWorkspace='peaks',
                        FilterVariable='RunNumber',
                        FilterValue=r,
                        Operator='=',
                        OutputWorkspace='peaks_run_{}'.format(r))

            FilterPeaks(InputWorkspace='peaks_run_{}'.format(r),
                        FilterVariable='h^2+k^2+l^2',
                        FilterValue=0,
                        Operator='>',
                        OutputWorkspace='tmp')

            if mtd['tmp'].getNumberPeaks() < 0.8*mtd['peaks_run_{}'.format(r)].getNumberPeaks():

                common_ind = False

        if common_ind:

            FindUBUsingIndexedPeaks(PeaksWorkspace='peaks', Tolerance=0.15, ToleranceForSatellite=0.15, CommonUBForAll=False)

            OptimizeLatticeForCellType(PeaksWorkspace='peaks', CellType=cell_type, PerRun=False, Apply=True, OutputDirectory=dbgdir)

            SaveIsawUB(InputWorkspace='peaks', Filename=os.path.join(outdir, '{}_{}_opt.mat'.format(cell_type,centering)))

        for p in range(n_proc):
            partfile = os.path.join(dbgdir,'peaks_p{}.nxs'.format(p))
            if os.path.exists(partfile):
                os.remove(partfile)

        CreatePeaksWorkspace(NumberOfPeaks=0,
                             InstrumentWorkspace='peaks',
                             OutputType='Peak',
                             OutputWorkspace='reindx')

        if np.array([a,b,c,alpha,beta,gamma]).all():

            FindUBUsingLatticeParameters(PeaksWorkspace='peaks',
                                         a=a,
                                         b=b,
                                         c=c,
                                         alpha=alpha,
                                         beta=beta,
                                         gamma=gamma,
                                         FixParameters=False)

        if not np.array([a,b,c,alpha,beta,gamma]).all():

            ol = mtd['peaks'].sample().getOrientedLattice()

            a, b, c, alpha, beta, gamma = ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma()

        ol = mtd['peaks'].sample().getOrientedLattice()
        U = ol.getU().copy()
        UB = ol.getUB().copy()

        for r in run_nos:

            if common_ind:

                FindUBUsingIndexedPeaks(PeaksWorkspace='peaks_run_{}'.format(r),
                                        Tolerance=0.15,
                                        ToleranceForSatellite=0.15,
                                        CommonUBForAll=False)

            else:

                LoadIsawUB(InputWorkspace='peaks_run_{}'.format(r),
                           Filename=os.path.join(dbgdir,'{}_{}_{}_{}.mat'.format(instrument,r,cell_type,centering)))

                IndexPeaks(PeaksWorkspace='peaks_run_{}'.format(r),
                           Tolerance=0.15,
                           ToleranceForSatellite=0.15,
                           RoundHKLs=False,
                           CommonUBForAll=False,
                           ModVector1=mod_vector_1,
                           ModVector2=mod_vector_2,
                           ModVector3=mod_vector_3,
                           MaxOrder=max_order,
                           CrossTerms=cross_terms,
                           SaveModulationInfo=True if max_order > 0 else False)

            FilterPeaks(InputWorkspace='peaks_run_{}'.format(r),
                        FilterVariable='h^2+k^2+l^2', 
                        FilterValue=0,
                        Operator='>',
                        OutputWorkspace='peaks_run_{}'.format(r))

            CombinePeaksWorkspaces(LHSWorkspace='reindx', 
                                   RHSWorkspace='peaks_run_{}'.format(r), 
                                   OutputWorkspace='reindx')
                                   
        a_min = b_min = c_min = 0.5
        a_max = b_max = c_max = 300

        alpha_min = beta_min = gamma_min = 10
        alpha_max = beta_max = gamma_max = 170

        if (np.allclose([a, b], c) and np.allclose([alpha, beta, gamma], 90)) or select_cell_type == 'Cubic':
            fun = __cub
            x0 = (a, )
            xmin = (a_min, )
            xmax = (a_max, )
        elif (np.allclose([a, b], c) and np.allclose([alpha, beta], gamma)) or select_cell_type == 'Rhombohedral':
            fun = __rhom
            x0 = (a, np.deg2rad(alpha))
            xmin = (a_min, np.deg2rad(alpha_min))
            xmax = (a_max, np.deg2rad(alpha_max))
        elif (np.isclose(a, b) and np.allclose([alpha, beta, gamma], 90)) or select_cell_type == 'Tetragonal':
            fun = __tet
            x0 = (a, c)
            xmin = (a_min, c_min)
            xmax = (a_max, c_max)
        elif (np.isclose(a, b) and np.allclose([alpha, beta], 90) and np.isclose(gamma, 120)) or select_cell_type == 'Hexagonal':
            fun = __hex
            x0 = (a, c)
            xmin = (a_min, c_min)
            xmax = (a_max, c_max)
        elif (np.allclose([alpha, beta, gamma], 90)) or select_cell_type == 'Orthorhombic':
            fun = __ortho
            x0 = (a, b, c)
            xmin = (a_min, b_min, c_min)
            xmax = (a_max, b_max, c_max)
        elif np.allclose([alpha, gamma], 90) or select_cell_type == 'Monoclinic':
            fun = __mono1
            x0 = (a, b, c, np.deg2rad(beta))
            xmin = (a_min, b_min, c_min, np.deg2rad(beta_min))
            xmax = (a_max, b_max, c_max, np.deg2rad(beta_max))
        else:
            fun = __tri
            x0 = (a, b, c, np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma))
            xmin = (a_min, b_min, c_min, np.deg2rad(alpha_min), np.deg2rad(beta_min), np.deg2rad(gamma_min))
            xmax = (a_max, b_max, c_max, np.deg2rad(alpha_max), np.deg2rad(beta_max), np.deg2rad(gamma_max))

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

        omega, theta, phi = [], [], []

        hkl, Q = [], []

        numbers = []

        for r in run_nos:

            ol = mtd['peaks_run_{}'.format(r)].sample().getOrientedLattice()

            U = ol.getU()

            _omega = np.arccos((np.trace(U)-1)/2)

            val, vec = np.linalg.eig(U)

            ux, uy, uz = vec[:,np.argwhere(np.isclose(val, 1))[0][0]].real

            _theta = np.arccos(uz)
            _phi = np.arctan2(uy,ux)

            N = mtd['peaks_run_{}'.format(r)].getNumberPeaks()

            numbers.append(N)

            _hkl, _Q = [], []

            for pn in range(N):
                pk = mtd['peaks_run_{}'.format(r)].getPeak(pn)
                h, k, l = pk.getIntHKL()
                m, n, p = pk.getIntMNP()

                if h**2+k**2+l**2 > 0 or m**2+n**2+p**2 > 0:
                    dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)

                    _hkl.append([h+dh,k+dk,l+dl])
                    _Q.append(pk.getQSampleFrame())

            omega.append(_omega)
            theta.append(_theta)
            phi.append(_phi)

            hkl.append(np.array(_hkl))
            Q.append(np.array(_Q))

        angles = tuple(np.column_stack((omega,theta,phi)).flatten().tolist())

        min_angles = tuple(np.row_stack([0,0,-np.pi]*len(hkl)).flatten().tolist())
        max_angles = tuple(np.row_stack([np.pi,np.pi,np.pi]*len(hkl)).flatten().tolist())

        sol = scipy.optimize.least_squares(__res, x0=x0+angles, args=(hkl,Q,fun), method='lm') #bounds=(xmin+min_angles,xmax+max_angles),

        a, b, c, alpha, beta, gamma, *angles = fun(sol.x)
        print(x0)
        print(a, b, c, alpha, beta, gamma)

        B = __B_matrix(a, b, c, alpha, beta, gamma)

        J = sol.jac
        cov = np.linalg.inv(J.T.dot(J))

        chi2dof = np.sum(sol.fun**2)/(sol.fun.size-sol.x.size)
        cov *= chi2dof

        sig = np.sqrt(np.diagonal(cov))

        sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma, *sig_angles = fun(sig)

        alpha, beta, gamma = np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)
        sig_alpha, sig_beta, sig_gamma = np.rad2deg(sig_alpha), np.rad2deg(sig_beta), np.rad2deg(sig_gamma)

        if np.isclose(a,sig_a):
            sig_a = 0
        if np.isclose(b,sig_b):
            sig_b = 0
        if np.isclose(c,sig_c):
            sig_c = 0

        if np.isclose(alpha,sig_alpha):
            sig_alpha = 0
        if np.isclose(beta,sig_beta):
            sig_beta = 0
        if np.isclose(gamma,sig_gamma):
            sig_gamma = 0

        remove = []

        for i, r in enumerate(run_nos):

            omega = angles[3*i+0]
            theta = angles[3*i+1]
            phi = angles[3*i+2]

            U = __U_matrix(phi, theta, omega)

            UB = np.dot(U,B)

            SetUB(Workspace='peaks_run_{}'.format(r), UB=UB)

            mtd['peaks_run_{}'.format(r)].sample().getOrientedLattice().setError(sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma)

            IndexPeaks(PeaksWorkspace='peaks_run_{}'.format(r),
                       Tolerance=0.15,
                       ToleranceForSatellite=0.15,
                       RoundHKLs=False,
                       ModVector1=mod_vector_1,
                       ModVector2=mod_vector_2,
                       ModVector3=mod_vector_3,
                       MaxOrder=max_order,
                       CrossTerms=cross_terms,
                       SaveModulationInfo=True if max_order > 0 else False)

            if numbers[i] > 10:

                SaveIsawUB(InputWorkspace='peaks_run_{}'.format(r),
                           Filename=os.path.join(dbgdir,'{}_{}_{}_{}_opt.mat'.format(instrument,r,cell_type,centering)))

            else:

                remove.append(r)

        for r in remove:
            print('Remove run {}'.format(r))

        SaveNexus(InputWorkspace='reindx', Filename=os.path.join(outdir,'{}_{}_opt.nxs'.format(cell_type,centering)))

    else:

        if instrument == 'HB3A':
            two_theta_max = 155
        elif instrument == 'HB2C':
            two_theta_max = 156

        if instrument == 'HB3A':
            data_files = ','.join(['/HFIR/{}/IPTS-{}/shared/autoreduce/{}_exp{:04}_scan{:04}.nxs'.format(instrument,ipts,instrument,exp,r) for r in run_nos])

            LoadMD(Filename='/HFIR/{}/IPTS-{}/shared/autoreduce/{}_exp{:04}_scan{:04}.nxs'.format(instrument,ipts,instrument,exp,run_nos[0]), 
                   OutputWorkspace='data')

            lamda = float(mtd['data'].getExperimentInfo(0).run().getProperty('wavelength').value)

            Q_max = 4*np.pi/lamda*np.sin(np.deg2rad(two_theta_max)/2)

            DeleteWorkspace(Workspace='data')

            HB3AAdjustSampleNorm(Filename=data_files, 
                                 DetectorHeightOffset=0,
                                 DetectorDistanceOffset=0,
                                 OutputType='Q-sample events',
                                 MergeInputs=True,
                                 MinValues='{},{},{}'.format(-Q_max,-Q_max,-Q_max),
                                 MaxValues='{},{},{}'.format(Q_max,Q_max,Q_max),
                                 OutputWorkspace='md')

        elif instrument == 'HB2C':

            LoadWANDSCD(IPTS=ipts,
                        RunNumbers=','.join([str(val) for val in run_nos]),
                        NormalizedBy='None',
                        Grouping='4x4', 
                        OutputWorkspace='data')

            SetGoniometer(Workspace='data',
                          Axis0='s1,0,1,0,1',
                          Average=False)

            lamda = 1.486 if instrument == 'HB2C' else float(mtd['data'].getExperimentInfo(0).run().getProperty('wavelength').value)

            Q_max = 4*np.pi/lamda*np.sin(np.deg2rad(two_theta_max)/2)

            ConvertHFIRSCDtoMDE(InputWorkspace='data',
                                Wavelength=lamda,
                                MinValues='{},{},{}'.format(-Q_max,-Q_max,-Q_max),
                                MaxValues='{},{},{}'.format(Q_max,Q_max,Q_max),
                                SplitInto=5,
                                SplitThreshold=1000,
                                MaxRecursionDepth=13,
                                OutputWorkspace='md')

        if ub_file is not None:
            LoadIsawUB(InputWorkspace='md', Filename=ub_file)

            lamda = 1.486 if instrument == 'HB2C' else float(mtd['data'].getExperimentInfo(0).run().getProperty('wavelength').value)

            lamda_min, lamda_max = 0.95*lamda, 1.05*lamda

            Q_max = 4*np.pi/lamda_min*np.sin(np.deg2rad(two_theta_max)/2)

            min_d_spacing = 2*np.pi/Q_max
            max_d_spacing = np.max([mtd['md'].getExperimentInfo(0).sample().getOrientedLattice().d(*hkl) for hkl in [(1,0,0),(0,1,0),(0,0,1)]])

            PredictPeaks(InputWorkspace='md',
                         WavelengthMin=lamda_min,
                         WavelengthMax=lamda_max,
                         MinDSpacing=min_d_spacing,
                         MaxDSpacing=max_d_spacing,
                         OutputType='LeanElasticPeak',
                         ReflectionCondition=reflection_condition if reflection_condition is not None else 'Primitive',
                         OutputWorkspace='peaks')

            CentroidPeaksMD(InputWorkspace='md',
                            PeakRadius=0.1,
                            PeaksWorkspace='peaks',
                            OutputWorkspace='peaks')
        else:
            FindPeaksMD(InputWorkspace='md', 
                        PeakDistanceThreshold=0.2,
                        DensityThresholdFactor=1000, 
                        MaxPeaks=400,
                        OutputType='Peak',
                        OutputWorkspace='peaks')

        IntegratePeaksMD(InputWorkspace='md',
                         PeakRadius=0.11,
                         BackgroundInnerRadius=0.11,
                         BackgroundOuterRadius=0.13,
                         PeaksWorkspace='peaks',
                         OutputWorkspace='peaks',
                         IntegrateIfOnEdge=True)

        FilterPeaks(InputWorkspace='peaks',
                    OutputWorkspace='peaks',
                    FilterVariable='Intensity',
                    FilterValue=10,
                    Operator='>')

        FilterPeaks(InputWorkspace='peaks',
                    OutputWorkspace='peaks',
                    FilterVariable='Signal/Noise',
                    FilterValue=10,
                    Operator='>')

        if ub_file is not None:
            LoadIsawUB(InputWorkspace='peaks', Filename=ub_file)
        elif np.array([a,b,c,alpha,beta,gamma]).all():
            FindUBUsingLatticeParameters(PeaksWorkspace='peaks',
                                         a=a,
                                         b=b,
                                         c=c,
                                         alpha=alpha,
                                         beta=beta,
                                         gamma=gamma)
        else:
            FindUBUsingFFT(PeaksWorkspace='peaks', MinD=3, MaxD=25, Iterations=15)
            IndexPeaks(PeaksWorkspace='peaks', Tolerance=0.15, RoundHKLs=True)
            SelectCellOfType(PeaksWorkspace='peaks', 
                             CellType=select_cell_type,
                             Centering=centering,
                             Apply=True)

        IndexPeaks(PeaksWorkspace='peaks',
                   Tolerance=0.15,
                   ToleranceForSatellite=0.15,
                   RoundHKLs=False,
                   CommonUBForAll=False,
                   ModVector1=mod_vector_1,
                   ModVector2=mod_vector_2,
                   ModVector3=mod_vector_3,
                   MaxOrder=max_order,
                   CrossTerms=cross_terms,
                   SaveModulationInfo=True if max_order > 0 else False)

        FilterPeaks(InputWorkspace='peaks',
                    FilterVariable='h^2+k^2+l^2', 
                    FilterValue=0,
                    Operator='>',
                    OutputWorkspace='peaks')

        if ub_file is not None:
            FindUBUsingIndexedPeaks('peaks', Tolerance=0.15, ToleranceForSatellite=0.15, CommonUBForAll=False)
        elif np.array([a,b,c,alpha,beta,gamma]).all():
            FindUBUsingLatticeParameters(PeaksWorkspace='peaks',
                                         a=a,
                                         b=b,
                                         c=c,
                                         alpha=alpha,
                                         beta=beta,
                                         gamma=gamma)
        else:
            FindUBUsingFFT(PeaksWorkspace='peaks', MinD=3, MaxD=25)
            IndexPeaks(PeaksWorkspace='peaks', Tolerance=0.15, RoundHKLs=True, CommonUBForAll=False)
            SelectCellOfType(PeaksWorkspace='peaks', 
                             CellType=select_cell_type,
                             Centering=centering,
                             Apply=True)

        IndexPeaks(PeaksWorkspace='peaks',
                   Tolerance=0.15,
                   ToleranceForSatellite=0.15,
                   RoundHKLs=False,
                   CommonUBForAll=False,
                   ModVector1=mod_vector_1,
                   ModVector2=mod_vector_2,
                   ModVector3=mod_vector_3,
                   MaxOrder=max_order,
                   CrossTerms=cross_terms,
                   SaveModulationInfo=True if max_order > 0 else False)

        FindUBUsingIndexedPeaks('peaks', Tolerance=0.15, ToleranceForSatellite=0.15, CommonUBForAll=False)

        OptimizeLatticeForCellType(PeaksWorkspace='peaks', CellType=cell_type, PerRun=True, Apply=True, OutputDirectory=dbgdir)

        IndexPeaks(PeaksWorkspace='peaks',
                   Tolerance=0.15,
                   ToleranceForSatellite=0.15,
                   RoundHKLs=False,
                   CommonUBForAll=False,
                   ModVector1=mod_vector_1,
                   ModVector2=mod_vector_2,
                   ModVector3=mod_vector_3,
                   MaxOrder=max_order,
                   CrossTerms=cross_terms,
                   SaveModulationInfo=True if max_order > 0 else False)

        SaveIsawUB(InputWorkspace='peaks', Filename=os.path.join(outdir, 'UB_opt.mat'))
