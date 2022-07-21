from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

import multiprocessing

filename, n_proc = sys.argv[1], int(sys.argv[2])

if n_proc > os.cpu_count():
    n_proc = os.cpu_count()
    
directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

import parameters
print(filename)
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
    #shutil.rmtree(outdir)
    os.mkdir(outdir)

parameters.output_input_file(filename, directory, outname+'_norm')

if dictionary['flux-file'] is not None:
    spectrum_file = os.path.join(shared_directory+'Vanadium', dictionary['flux-file'])
else:
    spectrum_file = None

if dictionary['vanadium-file'] is not None:
    counts_file = os.path.join(shared_directory+'Vanadium', dictionary['vanadium-file'])
else:
    counts_file = None

if dictionary.get('tube-file') is not None:
    tube_calibration = os.path.join(shared_directory+'calibration', dictionary['tube-file'])
else:
    tube_calibration = None

if dictionary['detector-file'] is not None:
    detector_calibration = os.path.join(shared_directory+'calibration', dictionary['detector-file'])
else:
    detector_calibration = None

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

# goniometer axis --------------------------------------------------------------
gon_axis = 'BL9:Mot:Sample:Axis3.RBV'

u_bin_size = (u_lims[1]-u_lims[0])/(u_bins-1)
v_bin_size = (v_lims[1]-v_lims[0])/(v_bins-1)
w_bin_size = (w_lims[1]-w_lims[0])/(w_bins-1)

u_binning = [u_lims[0]-u_bin_size/2,u_bin_size,u_lims[1]+u_bin_size/2]
v_binning = [v_lims[0]-v_bin_size/2,v_bin_size,v_lims[1]+v_bin_size/2]
w_binning = [w_lims[0]-w_bin_size/2,w_bin_size,w_lims[1]+w_bin_size/2]

def run_normalization(runs, p, facility, instrument, ipts, detector_calibration, tube_calibration, 
                      gon_axis, directory, counts_file, spectrum_file,
                      u_proj, v_proj, w_proj, u_binning, v_binning, w_binning):

    if not mtd.doesExist('sa'):
        LoadNexus(Filename=counts_file, OutputWorkspace='sa')

    if not mtd.doesExist('flux'):
        LoadNexus(Filename=spectrum_file, OutputWorkspace='flux')

    if tube_calibration is not None and not mtd.doesExist('tube_table'):
        LoadNexus(Filename=tube_calibration, OutputWorkspace='tube_table')

    if tube_calibration is not None:
        ApplyCalibration(Workspace='sa', CalibrationTable='tube_table')

    if detector_calibration is not None:
        _, ext =  os.path.splitext(detector_calibration)
        if ext == '.xml':
            LoadParameterFile(Workspace='sa', Filename=detector_calibration)
        else:
            LoadIsawDetCal(InputWorkspace='sa', Filename=detector_calibration)

    for r in runs:

        LoadEventNexus(Filename='/{}/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(facility,instrument,ipts,instrument,r), OutputWorkspace='data')

        if type(ub_file) is list:
            LoadIsawUB(InputWorkspace='data', Filename=ub_file[i])
        elif type(ub_file) is str:
            LoadIsawUB(InputWorkspace='data', Filename=ub_file)
        else:
            UB = mtd['data'].getExperimentInfo(0).sample().getOrientedLattice().getUB()
            SetUB(Workspace='data', UB=UB)

        CopyInstrumentParameters(InputWorkspace='sa', OutputWorkspace='data')

        if instrument == 'CORELLI':
            SetGoniometer('data', Axis0=str(gon_axis)+',0,1,0,1') 
        else:
            SetGoniometer('data', Goniometers='Universal') 

        ConvertUnits(InputWorkspace='data', OutputWorkspace='data', EMode='Elastic', Target='Momentum')

        CropWorkspaceForMDNorm(InputWorkspace='data',
                               XMin=mtd['flux'].dataX(0).min(),
                               XMax=mtd['flux'].dataX(0).max(),
                               OutputWorkspace='data')

        min_vals, max_vals = ConvertToMDMinMaxGlobal(InputWorkspace='data',
                                                     QDimensions='Q3D',
                                                     dEAnalysisMode='Elastic',
                                                     Q3DFrames='Q')

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

        DeleteWorkspace('data')

        MDNorm(InputWorkspace='md',
               SolidAngleWorkspace='sa',
               FluxWorkspace='flux',
               QDimension0='{},{},{}'.format(*u_proj),
               QDimension1='{},{},{}'.format(*v_proj),
               QDimension2='{},{},{}'.format(*w_proj),
               Dimension0Name='QDimension0',
               Dimension1Name='QDimension1',
               Dimension2Name='QDimension2',
               Dimension0Binning='{},{},{}'.format(*u_binning),
               Dimension1Binning='{},{},{}'.format(*v_binning),
               Dimension2Binning='{},{},{}'.format(*w_binning),
               TemporaryDataWorkspace='dataMD' if mtd.doesExist('dataMD') else None,
               TemporaryNormalizationWorkspace='normMD' if mtd.doesExist('normMD') else None,
               OutputWorkspace='normData',
               OutputDataWorkspace='dataMD',
               OutputNormalizationWorkspace='normMD')

        DeleteWorkspace('md')

    SaveMD(Inputworkspace='dataMD', Filename=os.path.join(outdir,'data_p{}.nxs'.format(p)))
    SaveMD(Inputworkspace='normMD', Filename=os.path.join(outdir,'norm_p{}.nxs'.format(p)))

if __name__ == '__main__':

    args = [facility, instrument, ipts, detector_calibration, tube_calibration, 
            gon_axis, directory, counts_file, spectrum_file,
            u_proj, v_proj, w_proj, u_binning, v_binning, w_binning]

    split_runs = [split.tolist() for split in np.array_split(run_nos, n_proc)]

    join_args = [(split, i, *args) for i, split in enumerate(split_runs)]

    with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:
        pool.starmap(run_normalization, join_args)
        pool.close()
        pool.join()

    for p in range(n_proc):
        LoadMD(OutputWorkspace='tmpDataMD', Filename=os.path.join(outdir,'data_p{}.nxs'.format(p)))
        LoadMD(OutputWorkspace='tmpNormMD', Filename=os.path.join(outdir,'norm_p{}.nxs'.format(p)))

        if p == 0:
            CloneMDWorkspace(InputWorkspace='tmpDataMD', OutputWorkspace='dataMD')
            CloneMDWorkspace(InputWorkspace='tmpNormMD', OutputWorkspace='normMD')
        else:
            PlusMD(LHSWorkspace='dataMD', RHSWorkspace='tmpDataMD', OutputWorkspace='dataMD')
            PlusMD(LHSWorkspace='normMD', RHSWorkspace='tmpNormMD', OutputWorkspace='normMD')

    DivideMD(LHSWorkspace='dataMD', RHSWorkspace='normMD', OutputWorkspace='normData')

    SaveMD(Inputworkspace='normData', Filename=os.path.join(directory,outname+'.nxs'))
    SaveMD(Inputworkspace='dataMD', Filename=os.path.join(directory,outname+'_data.nxs'))
    SaveMD(Inputworkspace='normMD', Filename=os.path.join(directory,outname+'_norm.nxs'))

    for p in range(n_proc):
        partfile = os.path.join(outdir,'data_p{}.nxs'.format(p))
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(outdir,'norm_p{}.nxs'.format(p))
        if os.path.exists(partfile):
            os.remove(partfile)