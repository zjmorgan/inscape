from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm
import matplotlib.transforms as mtransforms
from matplotlib.backends.backend_pdf import PdfPages

import os
import sys

import multiprocess as multiprocessing

from mantid.geometry import PointGroupFactory, SpaceGroupFactory

from mantid import config
#config.setLogLevel(0, quiet=True)

filename, n_proc = sys.argv[1], int(sys.argv[2])

#filename, n_proc = '/SNS/CORELLI/IPTS-31865/shared/normalization_YbFe6Ge6/YbFe6Ge6_small.conf', 1

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

run_list = []
for run in run_nos:
    if type(run) is list:
        for run_no in run:
            run_list.append(run_no)
    else:
        run_list.append(run)

run_nos = run_list

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
    os.mkdir(dbgdir)

if dictionary['flux-file'] is not None:
    spectrum_file = os.path.join(shared_directory+'Vanadium', dictionary['flux-file'])
    if not os.path.exists(spectrum_file):
        spectrum_file = os.path.join(working_directory, dictionary['flux-file'])
else:
    spectrum_file = None

if dictionary['vanadium-file'] is not None:
    counts_file = os.path.join(shared_directory+'Vanadium', dictionary['vanadium-file'])
    if not os.path.exists(counts_file):
        counts_file = os.path.join(working_directory, dictionary['vanadium-file'])
else:
    counts_file = None

calculated_bkg = False
if dictionary.get('background-file') is not None:
    background_file = os.path.join(shared_directory+'Background', dictionary['background-file'])
    if not os.path.exists(background_file):
        background_file = os.path.join(working_directory, dictionary['background-file'])
        if '*' in background_file:
            background_file = [background_file.replace('*', str(run)) for run in run_nos]
        calculated_bkg = True
else:
    background_file = None

if dictionary.get('tube-file') is not None:
    tube_calibration = os.path.join(shared_directory+'calibration', dictionary['tube-file'])
else:
    tube_calibration = None

if dictionary['detector-file'] is not None:
    detector_calibration = os.path.join(shared_directory+'calibration', dictionary['detector-file'])
else:
    detector_calibration = None

if dictionary.get('mask-file') is not None:
    mask_file = os.path.join(shared_directory+'Vanadium', dictionary['mask-file'])
    if not os.path.exists(mask_file):
        mask_file = os.path.join(working_directory, dictionary['mask-file'])
else:
    mask_file = None

if dictionary['ub-file'] is not None:
    ub_file = os.path.join(working_directory, dictionary['ub-file'])
    if '*' in ub_file:
        ub_file = [ub_file.replace('*', str(run)) for run in run_nos]
else:
    ub_file = None

projection = dictionary['projection']

u_proj = projection[0]
v_proj = projection[1]
w_proj = projection[2]

extents = dictionary['extents']

u_lims = extents[0]
v_lims = extents[1]
w_lims = extents[2]

bins = dictionary['bins']

u_bins = bins[0]
v_bins = bins[1]
w_bins = bins[2]

u_bin_size = (u_lims[1]-u_lims[0])/(u_bins-1) if u_bins > 1 else u_lims[1]-u_lims[0]
v_bin_size = (v_lims[1]-v_lims[0])/(v_bins-1) if v_bins > 1 else v_lims[1]-v_lims[0]
w_bin_size = (w_lims[1]-w_lims[0])/(w_bins-1) if w_bins > 1 else w_lims[1]-w_lims[0]

u_binning = [u_lims[0]-(u_bin_size/2 if u_bins > 1 else 0),u_bin_size,u_lims[1]+(u_bin_size/2 if u_bins > 1 else 0)]
v_binning = [v_lims[0]-(v_bin_size/2 if v_bins > 1 else 0),v_bin_size,v_lims[1]+(v_bin_size/2 if v_bins > 1 else 0)]
w_binning = [w_lims[0]-(w_bin_size/2 if w_bins > 1 else 0),w_bin_size,w_lims[1]+(w_bin_size/2 if w_bins > 1 else 0)]

group = dictionary['group']

elastic = dictionary.get('elastic')
timing_offset = dictionary.get('time-offset')

if elastic:
    outname += '_cc'

if timing_offset is None:
    timing_offset = 18000

if group is not None:    
    pgs = [pg.replace(' ', '') for pg in PointGroupFactory.getAllPointGroupSymbols()]
    sgs = [sg.replace(' ', '') for sg in SpaceGroupFactory.getAllSpaceGroupSymbols()]
    if type(group) is int:
        pg = SpaceGroupFactory.createSpaceGroup(SpaceGroupFactory.subscribedSpaceGroupSymbols(group)[0]).getPointGroup().getHMSymbol()
    elif group in pgs:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index(group)]).getHMSymbol()
    else:
        pg = SpaceGroupFactory.createSpaceGroup(SpaceGroupFactory.getAllSpaceGroupSymbols()[sgs.index(group)]).getPointGroup().getHMSymbol()

    group = pg.replace(' ', '')

    if group in ['1','-1']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('-1')]).getHMSymbol()
    elif group in ['2','m','2/m']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('2/m')]).getHMSymbol()
    elif group in ['112','11m','112/m']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('112/m')]).getHMSymbol()
    elif group in ['222','2mm','m2m','mm2','mmm']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('mmm')]).getHMSymbol()
    elif group in ['4','-4','4/m']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('4/m')]).getHMSymbol()
    elif group in ['422','4mm','-4m2','-42m','4/mmm']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('4/mmm')]).getHMSymbol()
    elif group in ['3','-3']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('-3')]).getHMSymbol()
    elif group in ['3r','-3r']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('-3r')]).getHMSymbol()
    elif group in ['32','3m','-3m']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('-3m')]).getHMSymbol()
    elif group in ['32r','3mr','-3mr']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('-3mr')]).getHMSymbol()
    elif group in ['321','3m1','-3m1']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('-3m1')]).getHMSymbol()
    elif group in ['312','31m','-31m']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('-31m')]).getHMSymbol()
    elif group in ['6','-6','6/m']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('6/m')]).getHMSymbol()
    elif group in ['622','6mm','-62m','-6m2','6/mmm']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('6/mmm')]).getHMSymbol()
    elif group in ['23','m-3']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('m-3')]).getHMSymbol()
    elif group in ['432','-43m','m-3m']:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index('m-3m')]).getHMSymbol()
    symmetry = pg

else:
    symmetry = '-1'

def run_normalization(runs, p, facility, instrument, ipts, detector_calibration, tube_calibration, 
                      directory, counts_file, spectrum_file, background_file, mask_file,
                      u_proj, v_proj, w_proj, u_binning, v_binning, w_binning, symmetry, elastic, timing_offset):

    if not mtd.doesExist(instrument):
        LoadEmptyInstrument(InstrumentName=instrument, OutputWorkspace=instrument)

    if not mtd.doesExist('sa'):
        LoadNexus(Filename=counts_file, OutputWorkspace='sa')

    if not mtd.doesExist('flux'):
        LoadNexus(Filename=spectrum_file, OutputWorkspace='flux')

    if tube_calibration is not None and not mtd.doesExist('tube_table'):
        LoadNexus(Filename=tube_calibration, OutputWorkspace='tube_table')

    if tube_calibration is not None:
        ApplyCalibration(Workspace='sa', CalibrationTable='tube_table')

    #LoadInstrument(Workspace='sa', InstrumentName=instrument, RewriteSpectraMap=True)

    #if instrument != 'CORELLI':
     #  SumNeighbours(InputWorkspace='sa', SumX=2, SumY=2, OutputWorkspace='sa')

    if detector_calibration is not None:
        _, ext =  os.path.splitext(detector_calibration)
        if ext == '.xml':
            LoadParameterFile(Workspace='sa', Filename=detector_calibration)
        else:
            LoadIsawDetCal(InputWorkspace='sa', Filename=detector_calibration)

    if mask_file is not None:
        LoadMask(Instrument=instrument, 
                 InputFile=mask_file,
                 RefWorkspace='sa',
                 OutputWorkspace='mask')
        MaskDetectors(Workspace='sa', MaskedWorkspace='mask')

    ExtractMask(InputWorkspace='sa', OutputWorkspace='mask')

    for i, r in enumerate(runs):

        fname = '/{}/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(facility,instrument,ipts,instrument,r)
        if not os.path.exists(fname):
            fname = '/{}/{}/IPTS-{}/shared/data/{}_{}.nxs.h5'.format(facility,instrument,ipts,instrument,r)

        LoadEventNexus(Filename=fname,
                       OutputWorkspace='data')

        if elastic:
            CopyInstrumentParameters(InputWorkspace=instrument, OutputWorkspace='data')
            CorelliCrossCorrelate(InputWorkspace='data', OutputWorkspace='data', TimingOffset=timing_offset)
            # LoadNexus(Filename='/{}/{}/IPTS-{}/shared/autoreduce/{}_{}_elastic.nxs'.format(facility,instrument,ipts,instrument,r), 
            #           OutputWorkspace='data') 

        if type(ub_file) is list:
            ind = [str(r) in ub for ub in ub_file].index(True)
            LoadIsawUB(InputWorkspace='data', Filename=ub_file[ind])
        elif type(ub_file) is str:
            LoadIsawUB(InputWorkspace='data', Filename=ub_file)
        else:
            UB = mtd['data'].getExperimentInfo(0).sample().getOrientedLattice().getUB()
            SetUB(Workspace='data', UB=UB)

        if tube_calibration is not None:
            ApplyCalibration(Workspace='data', CalibrationTable='tube_table')

        if detector_calibration is not None:
            _, ext =  os.path.splitext(detector_calibration)
            if ext == '.xml':
                LoadParameterFile(Workspace='data', Filename=detector_calibration)
            else:
                LoadIsawDetCal(InputWorkspace='data', Filename=detector_calibration)

        MaskDetectors(Workspace='data', MaskedWorkspace='mask')

        if instrument == 'CORELLI':
            gon_axis = 'BL9:Mot:Sample:Axis3'
            possible_axes = ['BL9:Mot:Sample:Axis1', 'BL9:Mot:Sample:Axis2', 'BL9:Mot:Sample:Axis3', 
                             'BL9:Mot:Sample:Axis1.RBV', 'BL9:Mot:Sample:Axis2.RBV', 'BL9:Mot:Sample:Axis3.RBV'] #.RBV
            for possible_axis in possible_axes:
                if mtd['data'].run().hasProperty(possible_axis):
                    angle = np.mean(mtd['data'].run().getProperty(possible_axis).value)
                    if not np.isclose(angle,0):
                        gon_axis = possible_axis
            SetGoniometer(Workspace='data', Axis0='{},0,1,0,1'.format(gon_axis))
        else:
            SetGoniometer(Workspace='data', Goniometers='Universal')
            SumNeighbours(InputWorkspace='data', OutputWorkspace='data', SumX=4, SumY=4)

        ConvertUnits(InputWorkspace='data', OutputWorkspace='data', EMode='Elastic', Target='Momentum')

        CropWorkspaceForMDNorm(InputWorkspace='data',
                               XMin=mtd['flux'].dataX(0).min(),
                               XMax=mtd['flux'].dataX(0).max(),
                               OutputWorkspace='data')

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
                    LorentzCorrection=False,
                    MinValues=min_vals,
                    MaxValues=max_vals,
                    Uproj='1,0,0',
                    Vproj='0,1,0',
                    Wproj='0,0,1')

        RecalculateTrajectoriesExtents(InputWorkspace='md',
                                       OutputWorkspace='md')

        if background_file is not None and not mtd.doesExist('bkg'):
            if type(background_file) is list:
                ind = [str(r) in bkg for bkg in background_file].index(True)
                LoadNexus(OutputWorkspace='bkg', Filename=background_file[ind])
            elif type(background_file) is str:
                LoadNexus(OutputWorkspace='bkg', Filename=background_file)
 
            if instrument != 'CORELLI':
                SumNeighbours(InputWorkspace='bkg', OutputWorkspace='bkg', SumX=4, SumY=4)
 
        if mtd.doesExist('bkg') and not mtd.doesExist('bkg_md'):

            ConvertToMD(InputWorkspace='bkg', 
                        OutputWorkspace='bkg_md' if not calculated_bkg else 'bkg_mde', 
                        QDimensions='Q3D',
                        dEAnalysisMode='Elastic',
                        Q3DFrames='Q_lab',
                        LorentzCorrection=False,
                        MinValues=min_vals,
                        MaxValues=max_vals,
                        Uproj='1,0,0',
                        Vproj='0,1,0',
                        Wproj='0,0,1')

            if type(background_file) is list:
                DeleteWorkspace('bkg')

        if calculated_bkg:

            pc = mtd['data'].run().getProperty('gd_prtn_chrg').value

            CreateSingleValuedWorkspace(DataValue=pc, OutputWorkspace='pc_scale')

            MultiplyMD(LHSWorkspace='bkg_mde', RHSWorkspace='pc_scale', OutputWorkspace='bkg_md')

            AddSampleLog(Workspace='bkg_md',
                         LogName='gd_prtn_chrg',
                         LogText=str(pc),
                         LogType='Number',
                         NumberType='Double')

        DeleteWorkspace('data')

        MDNorm(InputWorkspace='md',
               SolidAngleWorkspace='sa',
               FluxWorkspace='flux',
               BackgroundWorkspace='bkg_md' if mtd.doesExist('bkg_md') else None,
               QDimension0='{},{},{}'.format(*u_proj),
               QDimension1='{},{},{}'.format(*v_proj),
               QDimension2='{},{},{}'.format(*w_proj),
               Dimension0Name='QDimension0',
               Dimension1Name='QDimension1',
               Dimension2Name='QDimension2',
               Dimension0Binning='{},{},{}'.format(*u_binning),
               Dimension1Binning='{},{},{}'.format(*v_binning),
               Dimension2Binning='{},{},{}'.format(*w_binning),
               SymmetryOperations=symmetry,
               TemporaryDataWorkspace='dataMD' if mtd.doesExist('dataMD') else None,
               TemporaryNormalizationWorkspace='normMD' if mtd.doesExist('normMD') else None,
               TemporaryBackgroundDataWorkspace='bkgDataMD' if mtd.doesExist('bkgDataMD') else None,
               TemporaryBackgroundNormalizationWorkspace='bkgNormMD' if mtd.doesExist('bkgNormMD') else None,
               OutputWorkspace='normData',
               OutputDataWorkspace='dataMD',
               OutputNormalizationWorkspace='normMD',
               OutputBackgroundDataWorkspace='bkgDataMD' if mtd.doesExist('bkg_md') else None,
               OutputBackgroundNormalizationWorkspace='bkgNormMD' if mtd.doesExist('bkg_md') else None)

        MDNorm(InputWorkspace='md',
               SolidAngleWorkspace='sa',
               FluxWorkspace='flux',
               BackgroundWorkspace='bkg_md' if mtd.doesExist('bkg_md') else None,
               QDimension0='{},{},{}'.format(*u_proj),
               QDimension1='{},{},{}'.format(*v_proj),
               QDimension2='{},{},{}'.format(*w_proj),
               Dimension0Name='QDimension0',
               Dimension1Name='QDimension1',
               Dimension2Name='QDimension2',
               Dimension0Binning='{},{},{}'.format(*u_binning),
               Dimension1Binning='{},{},{}'.format(*v_binning),
               Dimension2Binning='{},{},{}'.format(*w_binning),
               SymmetryOperations=None,
               TemporaryDataWorkspace='dataMD_no_symm' if mtd.doesExist('dataMD_no_symm') else None,
               TemporaryNormalizationWorkspace='normMD_no_symm' if mtd.doesExist('normMD_no_symm') else None,
               TemporaryBackgroundDataWorkspace='bkgDataMD_no_symm' if mtd.doesExist('bkgDataMD_no_symm') else None,
               TemporaryBackgroundNormalizationWorkspace='bkgNormMD_no_symm' if mtd.doesExist('bkgNormMD_no_symm') else None,
               OutputWorkspace='normData_no_symm',
               OutputDataWorkspace='dataMD_no_symm',
               OutputNormalizationWorkspace='normMD_no_symm',
               OutputBackgroundDataWorkspace='bkgDataMD_no_symm' if mtd.doesExist('bkg_md') else None,
               OutputBackgroundNormalizationWorkspace='bkgNormMD_no_symm' if mtd.doesExist('bkg_md') else None)

        DeleteWorkspace('md')

        if type(background_file) is list:
            DeleteWorkspace('bkg_md')

    SaveMD(Inputworkspace='dataMD', Filename=os.path.join(dbgdir,'data_{}_p{}.nxs'.format(symmetry.replace('/','_').strip(),p)), SaveHistory=False, SaveInstrument=False, SaveSample=False, SaveLogs=False)
    SaveMD(Inputworkspace='normMD', Filename=os.path.join(dbgdir,'norm_{}_p{}.nxs'.format(symmetry.replace('/','_').strip(),p)), SaveHistory=False, SaveInstrument=False, SaveSample=False, SaveLogs=False)

    SaveMD(Inputworkspace='dataMD_no_symm', Filename=os.path.join(dbgdir,'data_p{}.nxs'.format(p)), SaveHistory=False, SaveInstrument=False, SaveSample=False, SaveLogs=False)
    SaveMD(Inputworkspace='normMD_no_symm', Filename=os.path.join(dbgdir,'norm_p{}.nxs'.format(p)), SaveHistory=False, SaveInstrument=False, SaveSample=False, SaveLogs=False)

    DeleteWorkspace('dataMD')
    DeleteWorkspace('normMD')

    DeleteWorkspace('dataMD_no_symm')
    DeleteWorkspace('normMD_no_symm')

    if mtd.doesExist('bkg_md') or calculated_bkg:

        SaveMD(Inputworkspace='bkgDataMD', Filename=os.path.join(dbgdir,'bkg_data_{}_p{}.nxs'.format(symmetry.replace('/','_').strip(),p)), SaveHistory=False, SaveInstrument=False, SaveSample=False, SaveLogs=False)
        SaveMD(Inputworkspace='bkgNormMD', Filename=os.path.join(dbgdir,'bkg_norm_{}_p{}.nxs'.format(symmetry.replace('/','_').strip(),p)), SaveHistory=False, SaveInstrument=False, SaveSample=False, SaveLogs=False)

        SaveMD(Inputworkspace='bkgDataMD_no_symm', Filename=os.path.join(dbgdir,'bkg_data_p{}.nxs'.format(p)), SaveHistory=False, SaveInstrument=False, SaveSample=False, SaveLogs=False)
        SaveMD(Inputworkspace='bkgNormMD_no_symm', Filename=os.path.join(dbgdir,'bkg_norm_p{}.nxs'.format(p)), SaveHistory=False, SaveInstrument=False, SaveSample=False, SaveLogs=False)

        DeleteWorkspace('bkgDataMD')
        DeleteWorkspace('bkgNormMD')

        DeleteWorkspace('bkgDataMD_no_symm')
        DeleteWorkspace('bkgNormMD_no_symm')

    DeleteWorkspace('sa')
    DeleteWorkspace('flux')

    if mtd.doesExist('bkg_md'):
        DeleteWorkspace('bkg_md')

if __name__ == '__main__':

    parameters.output_input_file(filename, directory, outname+'_norm')

    args = [facility, instrument, ipts, detector_calibration, tube_calibration, 
            directory, counts_file, spectrum_file, background_file, mask_file,
            u_proj, v_proj, w_proj, u_binning, v_binning, w_binning, symmetry, elastic, timing_offset]

    split_runs = [split.tolist() for split in np.array_split(run_nos, n_proc)]

    join_args = [(split, i, *args) for i, split in enumerate(split_runs)]

    config['MultiThreaded.MaxCores'] == 1
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['_SC_NPROCESSORS_ONLN'] = '1'

    #run_normalization(*join_args[0])
    multiprocessing.set_start_method('spawn', force=True)    
    with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:
        pool.starmap(run_normalization, join_args)
        pool.close()
        pool.join()

    config['MultiThreaded.MaxCores'] == 4
    os.environ.pop('OPENBLAS_NUM_THREADS', None)
    os.environ.pop('OMP_NUM_THREADS', None)
    os.environ.pop('_SC_NPROCESSORS_ONLN', None)

    for app in ['', '_'+symmetry.replace('/','_').strip()]:

        for p in range(n_proc):
            LoadMD(OutputWorkspace='tmpDataMD', Filename=os.path.join(dbgdir,'data{}_p{}.nxs'.format(app,p)), LoadHistory=False)
            LoadMD(OutputWorkspace='tmpNormMD', Filename=os.path.join(dbgdir,'norm{}_p{}.nxs'.format(app,p)), LoadHistory=False)

            if os.path.exists(os.path.join(dbgdir,'bkg_data{}_p{}.nxs'.format(app,p))):
                LoadMD(OutputWorkspace='tmpBkgDataMD', Filename=os.path.join(dbgdir,'bkg_data{}_p{}.nxs'.format(app,p)), LoadHistory=False)
                LoadMD(OutputWorkspace='tmpBkgNormMD', Filename=os.path.join(dbgdir,'bkg_norm{}_p{}.nxs'.format(app,p)), LoadHistory=False)

            if p == 0:
                CloneMDWorkspace(InputWorkspace='tmpDataMD', OutputWorkspace='dataMD')
                CloneMDWorkspace(InputWorkspace='tmpNormMD', OutputWorkspace='normMD')
                if mtd.doesExist('tmpBkgDataMD'):
                    CloneMDWorkspace(InputWorkspace='tmpBkgDataMD', OutputWorkspace='bkgDataMD')
                    CloneMDWorkspace(InputWorkspace='tmpBkgNormMD', OutputWorkspace='bkgNormMD')    
            else:
                PlusMD(LHSWorkspace='dataMD', RHSWorkspace='tmpDataMD', OutputWorkspace='dataMD')
                PlusMD(LHSWorkspace='normMD', RHSWorkspace='tmpNormMD', OutputWorkspace='normMD')
                if mtd.doesExist('tmpBkgDataMD'):
                    PlusMD(LHSWorkspace='bkgDataMD', RHSWorkspace='tmpBkgDataMD', OutputWorkspace='bkgDataMD')
                    PlusMD(LHSWorkspace='bkgNormMD', RHSWorkspace='tmpBkgNormMD', OutputWorkspace='bkgNormMD')

        DivideMD(LHSWorkspace='dataMD', RHSWorkspace='normMD', OutputWorkspace='normData')
        if mtd.doesExist('bkgDataMD'):
            DivideMD(LHSWorkspace='bkgDataMD', RHSWorkspace='bkgNormMD', OutputWorkspace='bkgNormData')
            MinusMD(LHSWorkspace='normData', RHSWorkspace='bkgNormData', OutputWorkspace='normDataSub')

        CreateSingleValuedWorkspace(OutputWorkspace='ws')

        W = np.array([u_proj,v_proj,w_proj]).T

        W_MATRIX = '{},{},{},{},{},{},{},{},{}'.format(*W.flatten())

        if type(ub_file) is list:
            LoadIsawUB(InputWorkspace='ws', Filename=ub_file[0])
        elif type(ub_file) is str:
            LoadIsawUB(InputWorkspace='ws', Filename=ub_file)

        for ws in ['normData', 'dataMD', 'normMD', 'bkgNormData', 'bkgDataMD', 'bkgNormMD', 'normDataSub']:
            if mtd.doesExist(ws):
                AddSampleLog(Workspace=ws, LogName='W_MATRIX', LogText=W_MATRIX, LogType='String')
                mtd[ws].getExperimentInfo(0).run().addProperty('W_MATRIX', list(W.flatten()*1.), True)
                CopySample(InputWorkspace='ws',
                           OutputWorkspace=ws,
                           CopyName=False,
                           CopyMaterial=False,
                           CopyEnvironment=False,
                           CopyLattice=True,
                           CopyOrientationOnly=False)

        SaveMD(Inputworkspace='normData', Filename=os.path.join(outdir,outname+app+'.nxs'), SaveHistory=False, SaveInstrument=True, SaveSample=True, SaveLogs=True)

        SaveMD(Inputworkspace='dataMD', Filename=os.path.join(outdir,outname+app+'_data.nxs'), SaveHistory=False, SaveInstrument=True, SaveSample=True, SaveLogs=True)
        SaveMD(Inputworkspace='normMD', Filename=os.path.join(outdir,outname+app+'_norm.nxs'), SaveHistory=False, SaveInstrument=True, SaveSample=True, SaveLogs=True)
        if mtd.doesExist('bkgNormData'):
            SaveMD(Inputworkspace='bkgDataMD', Filename=os.path.join(outdir,outname+app+'_bkg_data.nxs'), SaveHistory=False, SaveInstrument=True, SaveSample=True, SaveLogs=True)
            SaveMD(Inputworkspace='bkgNormMD', Filename=os.path.join(outdir,outname+app+'_bkg_norm.nxs'), SaveHistory=False, SaveInstrument=True, SaveSample=True, SaveLogs=True)
            SaveMD(Inputworkspace='bkgNormData', Filename=os.path.join(outdir,outname+app+'_bkg.nxs'), SaveHistory=False, SaveInstrument=True, SaveSample=True, SaveLogs=True)
            SaveMD(Inputworkspace='normDataSub', Filename=os.path.join(outdir,outname+app+'_sub_bkg.nxs'), SaveHistory=False, SaveInstrument=True, SaveSample=True, SaveLogs=True)

        for p in range(n_proc):
            partfile = os.path.join(dbgdir,'data{}_p{}.nxs'.format(app,p))
            if os.path.exists(partfile):
                os.remove(partfile)
            partfile = os.path.join(dbgdir,'norm{}_p{}.nxs'.format(app,p))
            if os.path.exists(partfile):
                os.remove(partfile)
            partfile = os.path.join(dbgdir,'bkg_data{}_p{}.nxs'.format(app,p))
            if os.path.exists(partfile):
                os.remove(partfile)
            partfile = os.path.join(dbgdir,'bkg_norm{}_p{}.nxs'.format(app,p))
            if os.path.exists(partfile):
                os.remove(partfile)

        data = mtd['normData']

        n = data.getNumDims()

        if n == 3:

            if app == '':
                app += '_no_symm'

            with PdfPages(os.path.join(outdir,outname+app+'.pdf')) as pdf:

                ol = mtd['ws'].sample().getOrientedLattice()

                dims = [data.getDimension(i) for i in range(n)]

                dmin = [dim.getMinimum() for dim in dims]
                dmax = [dim.getMaximum() for dim in dims]

                labels = [dim.getName().replace(',',' ').replace('[','(').replace(']',')').lower() for dim in dims]

                proj = [u_proj,v_proj,w_proj]

                for i in range(n):

                    hslice = IntegrateMDHistoWorkspace(InputWorkspace='normData',
                                                       P1Bin=[-0.0001,0.0001] if i == 0 else None, 
                                                       P2Bin=[-0.0001,0.0001] if i == 1 else None, 
                                                       P3Bin=[-0.0001,0.0001] if i == 2 else None)

                    signal = hslice.getSignalArray().copy().squeeze(axis=i)

                    signal[signal <= 0] = np.nan

                    j, k = np.sort([(i+1) % n, (i+2) % n])

                    angle = ol.recAngle(*proj[j],*proj[k])

                    transform = mtransforms.Affine2D().skew_deg(90-angle,0)

                    fig, ax = plt.subplots()

                    vmin, vmax = np.nanpercentile(signal,2), np.nanpercentile(signal,98)

                    im = ax.imshow(signal.T, extent=[dmin[j],dmax[j],dmin[k],dmax[k]], origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax, rasterized=True)
                    ax.set_xlabel(labels[j])
                    ax.set_ylabel(labels[k])
                    ax.set_title(labels[i]+' = 0')
                    ax.minorticks_on()

                    ax.grid(which='both', alpha=0.5, transform=transform)
                    ax.xaxis.get_major_locator().set_params(integer=True)
                    ax.yaxis.get_major_locator().set_params(integer=True)

                    trans_data = transform+ax.transData
                    im.set_transform(trans_data)
                    #ax.set_aspect((dmax[j]-dmin[j])/(dmax[k]-dmin[k]))

                    cb = fig.colorbar(im, ax=ax)
                    cb.ax.minorticks_on()

                    pdf.savefig()
                    plt.close()