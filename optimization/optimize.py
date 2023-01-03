from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

import multiprocess as multiprocessing

from mantid.geometry import PointGroupFactory, SpaceGroupFactory

import scipy.optimize

filename, n_proc = sys.argv[1], int(sys.argv[2])

#filename, n_proc = '/HFIR/HB3A/IPTS-29609/shared/zgf/HB3A_opt.config', 1

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
if not os.path.exists(outdir):
    os.mkdir(outdir)

rundir = os.path.join(directory,'{}_{}'.format(instrument,run_labels))
if not os.path.exists(rundir):
    os.mkdir(rundir)

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

    a, b, c, alpha, beta, gamma, *angles= fun(x)

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

    max_d = 25
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
        else:
            SetGoniometer(Workspace='data', Goniometers='Universal') 

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

            CreatePeaksWorkspace(NumberOfPeaks=0,
                                 OutputType='LeanElasticPeak',
                                 OutputWorkspace='peaks')

        if ub_file is not None:

            LoadIsawUB(InputWorkspace='md', Filename=ub_file)

            max_d_spacing = np.max([mtd['md'].getExperimentInfo(0).sample().getOrientedLattice().d(*hkl) for hkl in [(1,0,0),(0,1,0),(0,0,1)]])

            PredictPeaks(InputWorkspace='md',
                         WavelengthMin=lamda_min,
                         WavelengthMax=lamda_max,
                         MinDSpacing=min_d_spacing,
                         MaxDSpacing=max_d_spacing,
                         OutputType='LeanElasticPeak',
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

            min_Q = 2*np.pi/max_d

            FindPeaksMD(InputWorkspace='md', 
                        PeakDistanceThreshold=min_Q*0.9,
                        DensityThresholdFactor=100, 
                        MaxPeaks=400,
                        OutputType='LeanElasticPeak',
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
                    FilterVariable='Signal/Noise',
                    FilterValue=5,
                    Operator='>')

        if mtd['pk'].getNumberPeaks() > 10:

            if ub_file is not None:
                LoadIsawUB(InputWorkspace='md', Filename=ub_file)
            elif np.array([a,b,c,alpha,beta,gamma]).all():
                FindUBUsingLatticeParameters(PeaksWorkspace='pk',
                                             a=a,
                                             b=b,
                                             c=c,
                                             alpha=alpha,
                                             beta=beta,
                                             gamma=gamma)
            else:
                FindUBUsingFFT(PeaksWorkspace='pk', MinD=min_d_spacing, MaxD=max_d, Iterations=15)
                IndexPeaks(PeaksWorkspace='pk', Tolerance=0.125, RoundHKLs=True)
                SelectCellOfType(PeaksWorkspace='pk',
                                 CellType=select_cell_type,
                                 Centering=centering,
                                 Apply=True)

            IndexPeaks(PeaksWorkspace='pk',
                       Tolerance=0.125,
                       ToleranceForSatellite=0.125,
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
                       Filename=os.path.join(outdir,'{}_{}_UB.mat'.format(instrument,r)))

        # FindUBUsingIndexedPeaks('pk', Tolerance=0.125, ToleranceForSatellite=0.125, CommonUBForAll=False)

        # FilterPeaks(InputWorkspace='pk',
        #             FilterVariable='h^2+k^2+l^2',
        #             FilterValue=0,
        #             Operator='>',
        #             OutputWorkspace='pk')

        CombinePeaksWorkspaces(LHSWorkspace='peaks', 
                               RHSWorkspace='pk', 
                               OutputWorkspace='peaks')

        DeleteWorkspace('data')
        DeleteWorkspace('md')
        DeleteWorkspace('pk')

    SaveNexus(InputWorkspace='peaks', Filename=os.path.join(outdir, 'peaks_p{}.nxs'.format(p)))

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

        max_d = 25
        if np.array([a,b,c,alpha,beta,gamma]).all():
            max_d = np.max([a,b,c])

        if not os.path.exists(os.path.join(outdir,'peaks.nxs')):

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

            config['MultiThreaded.MaxCores'] == 4
            os.environ.pop('OPENBLAS_NUM_THREADS', None)
            os.environ.pop('OMP_NUM_THREADS', None)
            os.environ.pop('_SC_NPROCESSORS_ONLN', None)

            #run_optimization(*join_args)

            CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace='peaks')

            for p in range(n_proc):
                LoadNexus(Filename=os.path.join(outdir, 'peaks_p{}.nxs'.format(p)), OutputWorkspace='pk')
                CombinePeaksWorkspaces(LHSWorkspace='pk', RHSWorkspace='peaks', OutputWorkspace='peaks')
                DeleteWorkspace('pk')

            SaveNexus(InputWorkspace='peaks', Filename=os.path.join(outdir,'peaks.nxs'))

        else:

            LoadNexus(OutputWorkspace='peaks', Filename=os.path.join(outdir,'peaks.nxs'))

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
            FindUBUsingFFT(PeaksWorkspace='peaks', Tolerance=0.15, MinD=min_d_spacing, MaxD=max_d, Iterations=15)
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

        FindUBUsingIndexedPeaks(PeaksWorkspace='peaks', Tolerance=0.15, ToleranceForSatellite=0.15, CommonUBForAll=False)

        OptimizeLatticeForCellType(PeaksWorkspace='peaks', CellType=cell_type, PerRun=False, Apply=True, OutputDirectory=rundir)

        SaveIsawUB(InputWorkspace='peaks', Filename=os.path.join(outdir, 'UB_opt.mat'))

        for p in range(n_proc):
            partfile = os.path.join(outdir,'peaks_p{}.nxs'.format(p))
            if os.path.exists(partfile):
                os.remove(partfile)

        CreatePeaksWorkspace(NumberOfPeaks=0,
                             OutputType='LeanElasticPeak',
                             OutputWorkspace='reindx')

        if np.array([a,b,c,alpha,beta,gamma]).all():

            FindUBUsingLatticeParameters(PeaksWorkspace='peaks',
                                         a=a,
                                         b=b,
                                         c=c,
                                         alpha=alpha,
                                         beta=beta,
                                         gamma=gamma,
                                         FixParameters=True)

        if not np.array([a,b,c,alpha,beta,gamma]).all():

            ol = mtd['peaks'].sample().getOrientedLattice()

            a, b, c, alpha, beta, gamma = ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma()

        ol = mtd['peaks'].sample().getOrientedLattice()
        U = ol.getU().copy()
        UB = ol.getUB().copy()

        for r in run_nos:

            FilterPeaks(InputWorkspace='peaks',
                        FilterVariable='RunNumber',
                        FilterValue=r,
                        Operator='=',
                        OutputWorkspace='peaks_run_{}'.format(r))

            if mtd['peaks_run_{}'.format(r)].getNumberPeaks() > 10:

                FindUBUsingLatticeParameters(PeaksWorkspace='peaks_run_{}'.format(r),
                                             a=a,
                                             b=b,
                                             c=c,
                                             alpha=alpha,
                                             beta=beta,
                                             gamma=gamma)

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

                UBm = mtd['peaks_run_{}'.format(r)].sample().getOrientedLattice().getUB()

                T = np.dot(np.linalg.inv(UB), UBm)

                if np.all(np.abs(T).max(axis=0) > 0.95):

                    print(T)
                    print(np.linalg.det(T))

                    T = np.sign(T)*np.isclose(np.abs(T),np.abs(T).max(axis=0))

                    print(T)
                    print(np.linalg.det(T))

                    TransformHKL(PeaksWorkspace='peaks_run_{}'.format(r), HKLTransform=T, FindError=False)

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

                CombinePeaksWorkspaces(LHSWorkspace='reindx', 
                                       RHSWorkspace='peaks_run_{}'.format(r), 
                                       OutputWorkspace='reindx')

        if (np.allclose([a, b], c) and np.allclose([alpha, beta, gamma], 90)) or cell_type == 'Cubic':
            fun = __cub
            x0 = (a, )
        elif (np.allclose([a, b], c) and np.allclose([alpha, beta], gamma)) or cell_type == 'Rhombohedral':
            fun = __rhom
            x0 = (a, np.deg2rad(alpha))
        elif (np.isclose(a, b) and np.allclose([alpha, beta, gamma], 90)) or cell_type == 'Tetragonal':
            fun = __tet
            x0 = (a, c)
        elif (np.isclose(a, b) and np.allclose([alpha, beta], 90) and np.isclose(gamma, 120)) or cell_type == 'Hexagonal':
            fun = __hex
            x0 = (a, c)
        elif (np.allclose([alpha, beta, gamma], 90)) or cell_type == 'Orthorhombic':
            fun = __ortho
            x0 = (a, b, c)
        elif np.allclose([alpha, gamma], 90) or cell_type == 'Monoclinic':
            fun = __mono1
            x0 = (a, b, c, np.deg2rad(beta))
        # elif np.allclose([alpha, beta], 90) or cell_type == 'Monoclinic2':
        #     fun = __mono1
        #     x0 = (a, b, c, np.deg2rad(gamma))
        else:
            fun = __tri
            x0 = (a, b, c, np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma))

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

        sol = scipy.optimize.least_squares(__res, x0=x0+angles, args=(hkl,Q,fun), method='lm')

        a, b, c, alpha, beta, gamma, *angles = fun(sol.x)

        B = __B_matrix(a, b, c, alpha, beta, gamma)

        remove = []

        for i, r in enumerate(run_nos):

            omega = angles[3*i+0]
            theta = angles[3*i+1]
            phi = angles[3*i+2]

            U = __U_matrix(phi, theta, omega)

            UB = np.dot(U,B)

            SetUB(Workspace='peaks_run_{}'.format(r), UB=UB)

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
                           Filename=os.path.join(outdir,'{}_{}_UB_opt.mat'.format(instrument,r)))

            else:

                remove.append(r)

        for r in remove:
            print('Remove run {}'.format(r))

        SaveNexus(InputWorkspace='reindx', Filename=os.path.join(outdir,'peaks_opt.nxs'))

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
            IndexPeaks(PeaksWorkspace='peaks', Tolerance=0.125, RoundHKLs=True)
            SelectCellOfType(PeaksWorkspace='peaks', 
                             CellType=select_cell_type,
                             Centering=centering,
                             Apply=True)

        IndexPeaks(PeaksWorkspace='peaks',
                   Tolerance=0.125,
                   ToleranceForSatellite=0.125,
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
            FindUBUsingIndexedPeaks('peaks', Tolerance=0.125, ToleranceForSatellite=0.125, CommonUBForAll=False)
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
            IndexPeaks(PeaksWorkspace='peaks', Tolerance=0.125, RoundHKLs=True, CommonUBForAll=False)
            SelectCellOfType(PeaksWorkspace='peaks', 
                             CellType=select_cell_type,
                             Centering=centering,
                             Apply=True)

        IndexPeaks(PeaksWorkspace='peaks',
                   Tolerance=0.125,
                   ToleranceForSatellite=0.125,
                   RoundHKLs=False,
                   CommonUBForAll=False,
                   ModVector1=mod_vector_1,
                   ModVector2=mod_vector_2,
                   ModVector3=mod_vector_3,
                   MaxOrder=max_order,
                   CrossTerms=cross_terms,
                   SaveModulationInfo=True if max_order > 0 else False)

        FindUBUsingIndexedPeaks('peaks', Tolerance=0.125, ToleranceForSatellite=0.125, CommonUBForAll=False)

        OptimizeLatticeForCellType(PeaksWorkspace='peaks', CellType=cell_type, PerRun=True, Apply=True, OutputDirectory=rundir)

        IndexPeaks(PeaksWorkspace='peaks',
                   Tolerance=0.125,
                   ToleranceForSatellite=0.125,
                   RoundHKLs=False,
                   CommonUBForAll=False,
                   ModVector1=mod_vector_1,
                   ModVector2=mod_vector_2,
                   ModVector3=mod_vector_3,
                   MaxOrder=max_order,
                   CrossTerms=cross_terms,
                   SaveModulationInfo=True if max_order > 0 else False)

        SaveIsawUB(InputWorkspace='peaks', Filename=os.path.join(outdir, 'UB_opt.mat'))