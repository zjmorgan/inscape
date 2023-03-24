from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

import multiprocess as multiprocessing

from mantid.geometry import PointGroupFactory, SpaceGroupFactory

import scipy.optimize

from matplotlib.colors import LogNorm
import matplotlib.transforms as mtransforms
from matplotlib.backends.backend_pdf import PdfPages

filename = sys.argv[1]
#filename, n_proc = '/SNS/CORELLI/IPTS-23019/shared/reduction/optimization/Yb3Al5O12_300K_2022_0311_opt.inp', 1

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

if dictionary['flux-file'] is not None:
    spectrum_file = os.path.join(shared_directory+'Vanadium', dictionary['flux-file'])
else:
    spectrum_file = None

if dictionary['vanadium-file'] is not None:
    counts_file = os.path.join(shared_directory+'Vanadium', dictionary['vanadium-file'])
else:
    counts_file = None

if dictionary.get('background-file') is not None:
    background_file = os.path.join(shared_directory+'Background', dictionary['background-file'])
else:
    background_file = None

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

gon_axis = 'BL9:Mot:Sample:Axis3.RBV'

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

data_to_merge = []
md_to_merge = []

for r in runs:

    LoadEventNexus(Filename='/{}/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(facility,instrument,ipts,instrument,r),
                   FilterByTimeStop='60',
                   OutputWorkspace='data_{}'.format(r))

    if tube_calibration is not None:
        ApplyCalibration(Workspace='data_{}'.format(r), CalibrationTable='tube_table')

    if r == runs[0]:
        if detector_calibration is not None:
            if os.path.splitext(detector_calibration)[-1] == '.xml':
                LoadParameterFile(Workspace='data_{}'.format(r), Filename=detector_calibration)
            else:
                LoadIsawDetCal(InputWorkspace='data_{}'.format(r), Filename=detector_calibration)
    else:
        CopyInstrumentParameters(InputWorkspace=data_to_merge[-1], OutputWorkspace='data_{}'.format(r))

    if instrument == 'CORELLI':
        SetGoniometer(Workspace='data_{}'.format(r), Axis0=str(gon_axis)+',0,1,0,1') 
    elif instrument == 'SNAP':
        SetGoniometer(Workspace='data_{}'.format(r), Axis0='omega,0,1,0,1') 
    else:
        SetGoniometer(Workspace='data_{}'.format(r), Goniometers='Universal') 

    min_vals = [-20,-20,-20]
    max_vals = [20,20,20]

    ConvertToMD(InputWorkspace='data_{}'.format(r), 
                OutputWorkspace='md_{}'.format(r), 
                QDimensions='Q3D',
                dEAnalysisMode='Elastic',
                Q3DFrames='Q_sample',
                LorentzCorrection=True,
                MinValues=min_vals,
                MaxValues=max_vals,
                Uproj='1,0,0',
                Vproj='0,1,0',
                Wproj='0,0,1')

    data_to_merge.append('data_{}'.format(r))
    md_to_merge.append('md_{}'.format(r))

data = GroupWorkspaces(data_to_merge)

md = MergeMD(md_to_merge)

FindPeaksMD(InputWorkspace='md',
            MaxPeaks=800,
            DensityThresholdFactor=5000,
            OutputWorkspace='pk')

if np.array([a,b,c,alpha,beta,gamma]).all():
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

    SaveIsawUB(InputWorkspace='pk',
               Filename=os.path.join(outdir,'{}_{}_{}_init.mat'.format(instrument,cell_type,centering)))

    SaveNexus(InputWorkspace='pk', Filename=os.path.join(outdir,'{}_{}_{}_init.nxs'.format(instrument,cell_type,centering)))

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

    md = MergeMD('md')

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

    fig.savefig(os.path.join(outdir,'{}_{}_{}_init.pdf'.format(instrument,cell_type,centering)))
    plt.close()