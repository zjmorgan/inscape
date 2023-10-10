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

#filename, n_proc = '/SNS/CORELLI/IPTS-28033/shared/background/Ce2Zr2O7_50mK.conf', 1

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

def background(runs, proc, all_banks, all_runs, outname, outdir, dbgdir, tube_calibration, detector_calibration, mask_file, instrument, ipts, banks, det_size, N_ws_bkg, cols, rows, k_min, k_max, k_step):

    LoadEmptyInstrument(InstrumentName=instrument, OutputWorkspace=instrument)

    if mask_file is not None:
        LoadMask(Instrument=instrument, InputFile=mask_file, RefWorkspace=instrument, OutputWorkspace='mask')

    if tube_calibration is not None:
        LoadNexus(Filename=os.path.join(calibration_directory, tube_calibration),
                  OutputWorkspace='tube_table')

    DeleteWorkspace(instrument)

    config['MultiThreaded.MaxCores'] == 1
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['_SC_NPROCESSORS_ONLN'] = '1'

    n_runs = len(all_runs)
    boundary = 'mirror'

    for i, b in enumerate(all_banks):

        for r in runs:

            ind = all_runs.tolist().index(r)

            if boundary == 'wrap':
                runs_to_merge = [all_runs[(ind+window)%n_runs] for window in range(-N_ws_bkg//2+1,N_ws_bkg//2+1)]
            else:
                runs_to_merge = [all_runs[np.min([np.max([0,ind+window]),n_runs-1])] for window in range(-N_ws_bkg//2+1,N_ws_bkg//2+1)]

            for run_to_merge in runs_to_merge:

                if not mtd.doesExist('{}_{}'.format(instrument,run_to_merge)):

                    LoadEventNexus(Filename='/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(instrument,ipts,instrument,run_to_merge),
                                   OutputWorkspace='{}_{}'.format(instrument,run_to_merge),
                                   BankName='bank{}'.format(b),
                                   SingleBankPixelsOnly=True,
                                   LoadMonitors=False,
                                   LoadLogs=False)

                    if mtd.doesExist('tube_table'):
                        ApplyCalibration(Workspace='{}_{}'.format(instrument,run_to_merge), CalibrationTable='tube_table')

                    if detector_calibration is not None:
                        ext = os.path.splitext(detector_calibration)[1]
                        if ext == '.xml':
                            LoadParameterFile(Workspace='{}_{}'.format(instrument,run_to_merge),
                                              Filename=os.path.join(calibration_directory, detector_calibration))
                        else:
                            LoadIsawDetCal(InputWorkspace='{}_{}'.format(instrument,run_to_merge),
                                           Filename=os.path.join(calibration_directory, detector_calibration))

                    if mask_file is not None:
                        MaskDetectors(Workspace='{}_{}'.format(instrument,run_to_merge), MaskedWorkspace='mask')

                    ConvertUnits(InputWorkspace='{}_{}'.format(instrument,run_to_merge), 
                                 OutputWorkspace='{}_{}'.format(instrument,run_to_merge),
                                 Target='Momentum')

                    Rebin(InputWorkspace='{}_{}'.format(instrument,run_to_merge),
                          OutputWorkspace='{}_{}'.format(instrument,run_to_merge),
                          Params='{},{},{}'.format(k_min,k_step,k_max),
                          PreserveEvents=True)

            ws_ev = GroupWorkspaces(InputWorkspaces=','.join(['{}_{}'.format(instrument,run_to_merge) for run_to_merge in runs_to_merge]))

            ws_bkg = GenerateGoniometerIndependentBackground(InputWorkspaces='ws_ev',
                                                             GroupingFile=os.path.join(dbgdir, '{}_group_bank_{}_{}x{}.xml'.format(instrument,b,rows,cols)),
                                                             PercentMin=50*(1-1/N_ws_bkg/2),
                                                             PercentMax=50*(1+1/N_ws_bkg/2))

            RemoveWorkspaceHistory(Workspace='ws_bkg')

            SaveNexus(InputWorkspace='ws_bkg',
                      Filename=os.path.join(dbgdir, '{}_{}_{}_bkg.nxs'.format(instrument,b,r)))

            DeleteWorkspace('ws_bkg')
            DeleteWorkspace('{}_{}'.format(instrument,runs_to_merge[0]))

        DeleteWorkspace('ws_ev')

def merge(runs, banks):

    for r in runs:
        for b in banks:
            LoadNexus(OutputWorkspace='tmp_es_{}'.format(b), 
                      Filename=os.path.join(dbgdir, '{}_{}_{}_bkg.nxs'.format(instrument,b,r)))

        ws_ev = GroupWorkspaces(GlobExpression='tmp_es*')
        ws_bkg = MergeRuns(InputWorkspaces='ws_ev', OutputWorkspace='ws_bkg')

        RemoveWorkspaceHistory(Workspace=ws_bkg)
        SaveNexus(InputWorkspace=ws_bkg, Filename=os.path.join(outdir, '{}_{}_bkg.nxs'.format(instrument,r)))
        DeleteWorkspace(ws_ev)
        DeleteWorkspace(ws_bkg)

if __name__ == '__main__':

    LoadEmptyInstrument(InstrumentName=instrument, OutputWorkspace=instrument)

    i = 0
    spectrumInfo = mtd[instrument].spectrumInfo()
    while spectrumInfo.isMonitor(i):
        i += 1
    CropWorkspace(InputWorkspace=instrument, StartWorkspaceIndex=i, OutputWorkspace=instrument)

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

    #banks = int(re.findall('bank[0-9]*', inst.getDetector(detID[-1]).getFullName())[0].replace('bank',''))

    comp = 'bank'+str(all_banks[0])
    if instrument == 'CORELLI':
        comp += '/sixteenpack'

    rect = inst.getComponentByName(comp)

    if instrument == 'CORELLI':
        det_size = [rect.nelements(), inst.getComponentByName(comp+'/tube1').nelements()]
    else:
        det_size = [rect.xpixels(),rect.ypixels()]

    PreprocessDetectorsToMD(InputWorkspace=instrument, OutputWorkspace='det_inst')

    detID = mtd['det_inst'].column('DetectorID')
    detMask = mtd['det_inst'].column('detMask')

    banks = len(detID)//det_size[0]//det_size[1]

    spectra = np.arange(len(detID)).reshape(-1,det_size[0],det_size[1])
    
    for i, bank in enumerate(all_banks):

        with open(os.path.join(dbgdir,'{}_group_bank_{}_{}x{}.xml'.format(instrument,bank,rows,cols)),'wt+') as f:

            f.write('<?xml version="1.0" encoding="UTF-8" ?>\n<detector-grouping instrument="'+instrument+'">\n')
            group_num = 0
            for j in range(0,det_size[0],cols):
                for k in range(0,det_size[1],rows):
                    group_name = str(group_num)
                    ids = spectra[i,j:j+cols,k:k+rows].reshape(-1)
                    detids = []
                    for l in ids:
                        detids.append(mtd[instrument].getDetector(int(l)).getID())

                    detids = str(detids).replace("[","").replace("]","")
                    f.write('<group name="'+group_name+'">\n   <detids val="'+detids+'"/> \n</group>\n')
                    group_num += 1

            f.write('</detector-grouping>')

    angles = []
    proton_charge = []

    gon_axis = 'BL9:Mot:Sample:Axis3' if instrument == 'CORELLI' else 'omega,chi,phi'

    for r in run_nos:
        LoadNexusLogs(Filename='/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(instrument,ipts,instrument,r),
                      Workspace=instrument,
                      AllowList='gd_prtn_chrg,'+gon_axis)

        if instrument == 'CORELLI':
            SetGoniometer(Workspace=instrument, Axis0='{},0,1,0,1'.format(gon_axis))
        else:
            SetGoniometer(Workspace=instrument, Goniometers='Universal') 

        omega, chi, phi = mtd[instrument].run().getGoniometer().getEulerAngles()
        pc = mtd[instrument].run().getProtonCharge()

        angles.append(omega)
        proton_charge.append(pc)

    sort = np.argsort(angles)

    run_nos = np.array(run_nos)[sort]
    proton_charge = np.array(proton_charge)[sort]

    DeleteWorkspace(instrument)

    args = [all_banks, run_nos, outname, outdir, dbgdir, tube_calibration, detector_calibration, mask_file, instrument, ipts, banks, det_size, N_ws_bkg, cols, rows, k_min, k_max, k_step]

    split_runs = [split.tolist() for split in np.array_split(run_nos, n_proc)]

    join_args = [(split, i, *args) for i, split in enumerate(split_runs)]

    config['MultiThreaded.MaxCores'] == 1
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['_SC_NPROCESSORS_ONLN'] = '1'

    # background(*join_args[0])
    multiprocessing.set_start_method('spawn', force=True)    
    with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:
        pool.starmap(background, join_args)
        pool.close()
        pool.join()

    args = [all_banks]

    split_runs = [split.tolist() for split in np.array_split(run_nos, n_proc)]

    join_args = [(split, *args) for i, split in enumerate(split_runs)]

    multiprocessing.set_start_method('spawn', force=True)    
    with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:
        pool.starmap(merge, join_args)
        pool.close()
        pool.join()

    for i, bank in enumerate(all_banks):

         os.remove(os.path.join(dbgdir,'{}_group_bank_{}_{}x{}.xml'.format(instrument,bank,rows,cols)))