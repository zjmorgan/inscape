from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

import multiprocessing

from mantid.geometry import PointGroupFactory, SpaceGroupFactory

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

run_labels = '_'.join([str(r[0])+'-'+str(r[-1]) if type(r) is list else str(r) for r in run_nos if any([type(item) is list for item in run_nos])])

if run_labels == '':
    run_labels = str(run_nos[0])+'-'+str(run_nos[-1])
        
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

parameters.output_input_file(filename, directory, outname+'_opt')

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
                     ub_file, a, b, c, alpha, beta, gamma, cell_type, centering, reflection_condition,
                     mod_vector_1, mod_vector_2, mod_vector_3, max_order, cross_terms):

    if tube_calibration is not None and not mtd.doesExist('tube_table'):
        LoadNexus(Filename=tube_calibration, OutputWorkspace='tube_table')

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

            CreatePeaksWorkspace(NumberOfPeaks=0,
                                 OutputType='LeanElasticPeak',
                                 OutputWorkspace='peaks')

        if ub_file is not None:
            LoadIsawUB(InputWorkspace='md', Filename=ub_file)

            if instrument == 'CORELLI':
                k_min, k_max, two_theta_max = 2.5, 10, 148.2
            elif instrument == 'TOPAZ':
                k_min, k_max, two_theta_max = 1.8, 12.5, 160
            elif instrument == 'MANDI':
                k_min, k_max, two_theta_max = 1.5, 3.0, 160
            elif instrument == 'SNAP':
                k_min, k_max, two_theta_max = 1.8, 12.5, 138

            lamda_min, lamda_max = 2*np.pi/k_max, 2*np.pi/k_min

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
                         OutputWorkspace='pk')

            CentroidPeaksMD(InputWorkspace='md',
                            PeakRadius=0.1,
                            PeaksWorkspace='pk',
                            OutputWorkspace='pk')
        else:
            FindPeaksMD(InputWorkspace='md', 
                        PeakDistanceThreshold=0.1,
                        DensityThresholdFactor=1000, 
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
                    FilterVariable='Intensity',
                    FilterValue=10,
                    Operator='>')

        FilterPeaks(InputWorkspace='pk',
                    OutputWorkspace='pk',
                    FilterVariable='Signal/Noise',
                    FilterValue=3,
                    Operator='>')

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
            FindUBUsingFFT(PeaksWorkspace='pk', MinD=3, MaxD=25)
            IndexPeaks(PeaksWorkspace='pk', Tolerance=0.125, RoundHKLs=True)
            SelectCellOfType(PeaksWorkspace='pk', 
                             CellType=cell_type,
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
    
    if facility == 'SNS':
        
        if not os.path.exists(os.path.join(outdir,'peaks.nxs')):

            args = [facility, instrument, ipts, detector_calibration, tube_calibration, gon_axis, directory,
                    ub_file, a, b, c, alpha, beta, gamma, cell_type, centering, reflection_condition,
                    mod_vector_1, mod_vector_2, mod_vector_3, max_order, cross_terms]

            split_runs = [split.tolist() for split in np.array_split(run_nos, n_proc)]

            join_args = [(split, i, *args) for i, split in enumerate(split_runs)]

            multiprocessing.set_start_method('spawn', force=True)
            with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:
                pool.starmap(run_optimization, join_args)
                pool.close()
                pool.join()

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
            FindUBUsingIndexedPeaks('peaks', Tolerance=0.125, ToleranceForSatellite=0.125, CommonUBForAll=True)
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
            IndexPeaks(PeaksWorkspace='peaks', Tolerance=0.125, RoundHKLs=True, CommonUBForAll=True)
            SelectCellOfType(PeaksWorkspace='peaks', 
                             CellType=cell_type,
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

        FindUBUsingIndexedPeaks('peaks', Tolerance=0.125, ToleranceForSatellite=0.125, CommonUBForAll=True)

        OptimizeLatticeForCellType(PeaksWorkspace='peaks', CellType=cell_type, PerRun=False, Apply=True, OutputDirectory=rundir)

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

        for p in range(n_proc):
            partfile = os.path.join(outdir,'peaks_p{}.nxs'.format(p))
            if os.path.exists(partfile):
                os.remove(partfile)      

        ol = mtd['peaks'].sample().getOrientedLattice()
        a, b, c, alpha, beta, gamma = ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma()

        Um = ol.getU().copy()

        for r in run_nos:

            FilterPeaks(InputWorkspace='peaks',
                        FilterVariable='RunNumber',
                        FilterValue=r,
                        Operator='=',
                        OutputWorkspace='tmp')

            FilterPeaks(InputWorkspace='tmp', 
                        FilterVariable='h^2+k^2+l^2', 
                        FilterValue=0, 
                        Operator='>', 
                        OutputWorkspace='tmp')

            R = mtd['tmp'].getPeak(0).getGoniometerMatrix().copy()
            N = mtd['tmp'].getNumberPeaks()

            # for pn in range(N):
            #     pk = mtd['tmp'].getPeak(pn)
            #     pk.setGoniometerMatrix(np.eye(3))

            if N > 25:

                CalculateUMatrix(PeaksWorkspace='tmp', a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

                U = mtd['tmp'].sample().getOrientedLattice().getU().copy()
                B = mtd['tmp'].sample().getOrientedLattice().getB().copy()

                print(np.dot(U.T,Um))

                UB = np.dot(U,B)

                SetUB(Workspace='tmp', UB=UB)

            SaveIsawUB(InputWorkspace='tmp',
                       Filename=os.path.join(rundir,'{}_{}_UB.mat'.format(instrument,r)))

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
                             CellType=cell_type,
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
                             CellType=cell_type,
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