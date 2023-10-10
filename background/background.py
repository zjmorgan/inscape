# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import os
import re

import scipy.ndimage

import multiprocessing

from mantid import config
#config.setLogLevel(0, quiet=True)

filename, n_proc = sys.argv[1], int(sys.argv[2])

#filename, n_proc = '/SNS/CORELLI/IPTS-28033/shared/background/Ce2Zr2O7_5K.conf', 1
#filename, n_proc = '/SNS/TOPAZ/IPTS-31189/shared/normalization/TOPAZ_bkg.conf', 1

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

#if len(run_nos) < n_proc:
#    n_proc = len(run_nos)

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

k_min, k_max = dictionary['k-range']
k_step = dictionary['k-step']

rows, cols = dictionary['grouping']

calibration_directory = shared_directory+'calibration'
vanadium_directory = shared_directory+'Vanadium'

if dictionary.get('tube-file') is not None:
    tube_calibration = os.path.join(calibration_directory, dictionary['tube-file'])
else:
    tube_calibration = None

if dictionary.get('detector-file') is not None:
    detector_calibration = os.path.join(calibration_directory, dictionary['detector-file'])
else:
    detector_calibration = None

if dictionary.get('mask-file') is not None:
    mask_file = os.path.join(vanadium_directory, dictionary['mask-file'])
else:
    mask_file = None
    
N_ws_bkg = dictionary['n-runs']
if N_ws_bkg % 2 == 0:
    N_ws_bkg += 1

# ---

LoadEmptyInstrument(InstrumentName=instrument, OutputWorkspace=instrument)
ExtractMonitors(InputWorkspace=instrument, DetectorWorkspace=instrument, MonitorWorkspace='-')

LoadEmptyInstrument(InstrumentName=instrument, OutputWorkspace=instrument)

if mask_file is not None:
    LoadMask(Instrument=instrument, InputFile=mask_file, RefWorkspace=instrument, OutputWorkspace='mask')
    MaskDetectors(Workspace=instrument, MaskedWorkspace='mask')

CreateGroupingWorkspace(InputWorkspace=instrument, GroupDetectorsBy='bank', OutputWorkspace='group')

GroupDetectors(InputWorkspace=instrument, OutputWorkspace='{}_banks'.format(instrument), CopyGroupingFromWorkspace='group')
PreprocessDetectorsToMD(InputWorkspace='{}_banks'.format(instrument), OutputWorkspace='det')

detID = mtd['det'].column('DetectorID')
detMask = mtd['det'].column('detMask')

inst = mtd[instrument].getInstrument()

all_banks = [int(re.findall('bank[0-9]*', inst.getDetector(i).getFullName())[0].replace('bank','')) for i, m in zip(detID,detMask) if m == 0]

banks = int(re.findall('bank[0-9]*', inst.getDetector(detID[-1]).getFullName())[0].replace('bank',''))

if instrument != 'CORELLI':
    banks += 1

comp = 'bank'+str(all_banks[0])
if instrument == 'CORELLI':
    comp += '/sixteenpack'

rect = inst.getComponentByName(comp)

if instrument == 'CORELLI':
    det_size = [rect.nelements(), inst.getComponentByName(comp+'/tube1').nelements()]
else:
    det_size = [rect.xpixels(),rect.ypixels()]

def background(all_banks, proc, outname, dbgdir, tube_calibration, detector_calibration, mask_file, runs, instrument, ipts, banks, det_size, N_ws_bkg, cols, rows, k_min, k_max, k_step):

    if tube_calibration is not None:
        LoadNexus(Filename=os.path.join(calibration_directory, tube_calibration),
                  OutputWorkspace='tube_table')

    if mask_file is not None:
        LoadMask(Instrument=instrument,
                 InputFile=mask_file, 
                 RefWorkspace=instrument,
                 OutputWorkspace='mask')

    config['MultiThreaded.MaxCores'] == 1
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['_SC_NPROCESSORS_ONLN'] = '1'

    n_all_banks = len(all_banks)
    n_runs = len(runs)

    pc = []
    angles = []

    for r in runs:
        LoadEventNexus(Filename='/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(instrument,ipts,instrument,r),
                       OutputWorkspace='logs',
                       MetaDataOnly=True)

        if instrument == 'CORELLI':
            gon_axis = 'BL9:Mot:Sample:Axis3'
            possible_axes = ['BL9:Mot:Sample:Axis1', 'BL9:Mot:Sample:Axis2', 'BL9:Mot:Sample:Axis3', 
                             'BL9:Mot:Sample:Axis1.RBV', 'BL9:Mot:Sample:Axis2.RBV', 'BL9:Mot:Sample:Axis3.RBV'] #.RBV
            for possible_axis in possible_axes:
                if mtd['logs'].run().hasProperty(possible_axis):
                    angle = np.mean(mtd['logs'].run().getProperty(possible_axis).value)
                    if not np.isclose(angle,0):
                        gon_axis = possible_axis
            SetGoniometer(Workspace='logs', Axis0='{},0,1,0,1'.format(gon_axis))
        else:
            SetGoniometer(Workspace='logs', Goniometers='Universal') 

        omega, chi, phi = mtd['logs'].run().getGoniometer().getEulerAngles()

        pc.append(mtd['logs'].run().getProtonCharge())
        angles.append(omega)

    runs = np.array(runs)[np.argsort(angles)]

    for bind, b in enumerate(all_banks):

        for rind, r in enumerate(runs):
            LoadEventNexus(Filename='/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(instrument,ipts,instrument,r),
                           BankName='bank{}'.format(b),
                           SingleBankPixelsOnly=True,
                           OutputWorkspace='{}_{}'.format(instrument,r),
                           LoadMonitors=False,
                           LoadLogs=False)

            AddSampleLog(Workspace='{}_{}'.format(instrument,r),
                         LogName='gd_prtn_chrg',
                         LogText=str(pc[rind]),
                         LogType='Number',
                         NumberType='Double')

            NormaliseByCurrent(InputWorkspace='{}_{}'.format(instrument,r),
                               OutputWorkspace='{}_{}'.format(instrument,r))

            DeleteLog(Workspace='{}_{}'.format(instrument,r), Name='gd_prtn_chrg')

            if mtd.doesExist('tube_table'):
                ApplyCalibration(Workspace='{}_{}'.format(instrument,r), CalibrationTable='tube_table')

            if detector_calibration is not None:
                ext = os.path.splitext(detector_calibration)[1]
                if ext == '.xml':
                    LoadParameterFile(Workspace='{}_{}'.format(instrument,r),
                                      Filename=os.path.join(calibration_directory, detector_calibration))
                else:
                    LoadIsawDetCal(InputWorkspace='{}_{}'.format(instrument,r),
                                   Filename=os.path.join(calibration_directory, detector_calibration))

            if mask_file is not None:
                MaskDetectors(Workspace='{}_{}'.format(instrument,r), MaskedWorkspace='mask')

            ConvertUnits(InputWorkspace='{}_{}'.format(instrument,r), 
                         OutputWorkspace='{}_{}'.format(instrument,r),
                         Target='Momentum')

            Rebin(InputWorkspace='{}_{}'.format(instrument,r),
                  OutputWorkspace='{}_{}'.format(instrument,r),
                  Params='{},{},{}'.format(k_min,k_step,k_max),
                  PreserveEvents=True)

        ws_ev = GroupWorkspaces(InputWorkspaces=','.join(['{}_{}'.format(instrument,r) for r in runs]))

        wsg = PreprocessDetectorsToMD('{}_{}'.format(instrument,runs[0]))
        detID = wsg.column('DetectorID')#[:-3]

        bank, col, row = np.unravel_index(detID, (banks,*det_size))

        col_nsteps = det_size[0] // cols
        row_nsteps = det_size[1] // rows

        det_group_list = [[[] for _ in range(row_nsteps)] for _ in range(col_nsteps)]

        for c, r, d in zip(col,row,detID):
            ind_col = c // cols
            ind_row = r // rows
            det_group_list[ind_col][ind_row].append(d)

        detlist = []
        with open(os.path.join(dbgdir,'{}_group_p{}_{}x{}.xml'.format(instrument,proc,rows,cols)),'wt+') as f:
            f.write('<?xml version="1.0" encoding="UTF-8" ?>\n<detector-grouping instrument="{}">\n'.format(instrument))
            det_group = -1
            for i in range(col_nsteps):
                for j in range(row_nsteps):
                    dets = det_group_list[i][j]
                    if len(dets) > 0:
                        detlist.append(dets)
                        det_group += 1
                        f.write('<group name="{}"><detids val="{}"/> </group>\n'.format(det_group,str(dets).lstrip('[').rstrip(']')))
            f.write('</detector-grouping>')

        ws_bkg = GenerateGoniometerIndependentBackground(InputWorkspaces='ws_ev',
                                                         GroupingFile=os.path.join(dbgdir,'{}_group_p{}_{}x{}.xml'.format(instrument,proc,rows,cols)),
                                                         PercentMin=50*(1-N_ws_bkg/n_runs/2),
                                                         PercentMax=50*(1+N_ws_bkg/n_runs/2))

        if not mtd.doesExist('ws_bkg_tot'):
            CloneWorkspace(InputWorkspace=ws_bkg, OutputWorkspace='ws_bkg_tot')
        else:
            MergeRuns(InputWorkspaces='ws_bkg,ws_bkg_tot', OutputWorkspace='ws_bkg_tot')

        RemoveWorkspaceHistory(Workspace='ws_bkg_tot')

        os.remove(os.path.join(dbgdir,'{}_group_p{}_{}x{}.xml'.format(instrument,proc,rows,cols)))

    RemoveWorkspaceHistory(Workspace='ws_bkg_tot')
    SaveNexus(InputWorkspace='ws_bkg_tot',
              Filename=os.path.join(dbgdir, '{}_bkg_p{}.nxs'.format(instrument,proc)))

if __name__ == '__main__':

    args = [outname, dbgdir, tube_calibration, detector_calibration, mask_file, run_nos, instrument, ipts, banks, det_size, N_ws_bkg, cols, rows, k_min, k_max, k_step]

    split_banks = [split.tolist() for split in np.array_split(all_banks, n_proc)]

    join_args = [(split, i, *args) for i, split in enumerate(split_banks)]

    config['MultiThreaded.MaxCores'] == 1
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['_SC_NPROCESSORS_ONLN'] = '1'

    #background(*join_args[0])
    multiprocessing.set_start_method('spawn', force=True)    
    with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:
        pool.starmap(background, join_args)
        pool.close()
        pool.join()


    for proc in range(n_proc):
        LoadNexus(OutputWorkspace='tmp_es_{}'.format(proc), 
                  Filename=os.path.join(dbgdir, '{}_bkg_p{}.nxs'.format(instrument,proc)))

    ws_ev = GroupWorkspaces(GlobExpression='tmp_es*')
    ws_bkg = MergeRuns(InputWorkspaces='ws_ev', OutputWorkspace='ws_bkg')

    RemoveWorkspaceHistory(Workspace=ws_bkg)
    SaveNexus(InputWorkspace=ws_bkg, Filename=os.path.join(outdir, '{}_bkg.nxs'.format(outname)))
    DeleteWorkspace(ws_ev)
    DeleteWorkspace(ws_bkg)

    # with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:
    #     pool.starmap(merge, join_args)
    #     pool.close()
    #     pool.join()

    config['MultiThreaded.MaxCores'] == 4
    os.environ.pop('OPENBLAS_NUM_THREADS', None)
    os.environ.pop('OMP_NUM_THREADS', None)
    os.environ.pop('_SC_NPROCESSORS_ONLN', None)

    for proc in range(n_proc):
       os.remove(os.path.join(dbgdir, '{}_bkg_p{}.nxs'.format(instrument,proc)))