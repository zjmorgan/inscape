from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

import os
import sys

filename = sys.argv[1] #'/HFIR/HB3A/IPTS-18227/shared/pnd/HB3A.conf'

tutorial = '' if 'shared/examples/IPTS' not in filename else '/shared/examples' 

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

import imp
import parameters

imp.reload(parameters)

dictionary = parameters.load_input_file(filename)

up_run_nos = dictionary['up-scans'] if type(dictionary['up-scans']) is list else [dictionary['up-scans']]
down_run_nos = dictionary['down-scans'] if type(dictionary['down-scans']) is list else [dictionary['down-scans']]

if np.any([type(run) is list for run in up_run_nos]):  
    up_run_nos = [run for run_no in up_run_nos for run in run_no]
if np.any([type(run) is list for run in down_run_nos]):  
    down_run_nos = [run for run_no in down_run_nos for run in run_no]

facility, instrument = 'HFIR', 'HB3A'
ipts = dictionary['ipts']
exp = dictionary['experiment']

working_directory = '/{}/{}/IPTS-{}/shared/'.format(facility,instrument+tutorial,ipts)
shared_directory = '/{}/{}/shared/'.format(facility,instrument)

directory = os.path.dirname(os.path.abspath(filename))
outname = dictionary['name']

# data normalization -----------------------------------------------------------
normalization = dictionary['normalization']

if normalization == 'monitor':
    normalize_by = 'Monitor'
elif normalization == 'time':
    normalize_by = 'Time'
else:
    normalize_by = 'Time'

scale_by_motor_step = True

# integrate peaks --------------------------------------------------------------
x_pixels = dictionary['x-pixels'] 
y_pixels = dictionary['y-pixels'] 

integration_method = 'Fitted' # 'Fitted', 'Counts', 'CountsWithFitting'
number_of_backgroud_points = 3  # 'Counts' only

apply_lorentz = True
optimize_q_vector = True

scale_factor = dictionary['scale-factor']
min_signal_noise_ratio = dictionary['minimum-signal-noise-ratio']
max_chi_square = dictionary['maximum-chi-square']

# ---

def gaussian(x, parameters):
    bkg, amp, mu, sigma, _ = parameters
    return bkg+amp*np.exp(-(x-mu)**2/sigma**2)

rootname = '/HFIR/HB3A/IPTS-{}/'.format(ipts)
scanfile = 'HB3A_exp{:04}_scan{:04}'

up_data = [scanfile.format(exp,s) for s in up_run_nos]
down_data = [scanfile.format(exp,s) for s in down_run_nos]

data_channels = [up_data,down_data]

for i, pk in enumerate(['up', 'down']):

    CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace=pk)

    for data in data_channels[i]:

        filename = rootname+'shared/autoreduce/'+data+'.nxs'

        HB3AAdjustSampleNorm(Filename=filename,
                             NormaliseBy=normalize_by,
                             ScaleByMotorStep=scale_by_motor_step,
                             OutputType='Detector',
                             OutputWorkspace=data)

        CreatePeaksWorkspace(data, NumberOfPeaks=0, OutputType='Peak', OutputWorkspace='tmp')

        run = mtd[data].getExperimentInfo(0).run()
        R = run.getGoniometer(run.getNumGoniometers()//2).getR()
        mtd['tmp'].run().getGoniometer(0).setR(R)

        CopySample(InputWorkspace=data,
                   OutputWorkspace='tmp',
                   CopyName=False,
                   CopyMaterial=False,
                   CopyEnvironment=False,
                   CopyShape=False)

        title = run['scan_title'].value
        hkl = np.array(title.split('(')[-1].split(')')[0].split(' ')).astype(float)

        peak = mtd['tmp'].createPeakHKL([*hkl])

        row, col = peak.getRow(), peak.getCol()

        HB3AIntegrateDetectorPeaks(InputWorkspace=data,
                                   Method=integration_method,
                                   NumBackgroundPts=number_of_backgroud_points,
                                   LowerLeft=[col-x_pixels,row-y_pixels],
                                   UpperRight=[col+x_pixels,row+y_pixels],
                                   ChiSqMax=max_chi_square,
                                   SignalNoiseMin=min_signal_noise_ratio,
                                   ScaleFactor=scale_factor,
                                   ApplyLorentz=apply_lorentz,
                                   OptimizeQVector=optimize_q_vector,
                                   OutputFitResults=True,
                                   OutputWorkspace='peaks')

        if mtd['peaks'].getNumberPeaks() > 0:
            mtd['peaks'].getPeak(0).setHKL(*hkl)

        CombinePeaksWorkspaces(LHSWorkspace='peaks', RHSWorkspace=pk, OutputWorkspace=pk)

        DeleteWorkspace('peaks')
        DeleteWorkspace('tmp')

        if mtd['peaks_fit_results'].size() > 0:
            RenameWorkspace(InputWorkspace='peaks_fit_results', OutputWorkspace=data+'_fit_results')
        else:
            DeleteWorkspace('peaks_fit_results')

for pk in ['up', 'down']:

    with PdfPages(os.path.join(directory,outname+'_'+pk+'.pdf')) as pdf:

        N = mtd[pk].getNumberPeaks()
        for i in range(N):
            scan = mtd[pk].getPeak(i).getRunNumber()
            hkl = mtd[pk].getPeak(i).getIntHKL()

            fig = plt.figure(figsize=(12,6))
            ax1 = fig.add_subplot(121, projection='mantid')
            ax2 = fig.add_subplot(122, projection='mantid')
            im = ax1.pcolormesh(mtd['peaks_'+scanfile.format(exp,scan)+'_ROI'], transpose=True)
            im.set_edgecolor('face')
            ax1.set_title('({:.0f} {:.0f} {:.0f})'.format(*hkl))
            ax1.minorticks_on()
            ax1.set_aspect(1)
            ax2.errorbar(mtd['peaks_'+scanfile.format(exp,scan)+'_Workspace'], wkspIndex=0, marker='o', linestyle='', label='data')
            if 'Fit' in integration_method:
                output = mtd['peaks_'+scanfile.format(exp,scan)+'_Parameters'].column(1)
                xdim = mtd['peaks_'+scanfile.format(exp,scan)+'_Workspace'].getXDimension()
                x = np.linspace(xdim.getMinimum(), xdim.getMaximum(),500)
                y = gaussian(x, output)
                ax2.plot(x, y, label='calc')
            ax2.plot(mtd['peaks_'+scanfile.format(exp,scan)+'_Workspace'], wkspIndex=2, marker='o', linestyle='--', label='diff')
            ax2.legend()
            ax2.set_title('Exp #{}, Scan #{}'.format(exp,scan))
            ax2.minorticks_on()
            pdf.savefig()
            plt.close()

    SaveHKLCW(pk, os.path.join(directory,outname+'_'+pk+'_SHELX_dir_cos.hkl'), DirectionCosines=True)
    SaveHKLCW(pk, os.path.join(directory,outname+'_'+pk+'_SHELX.hkl'), DirectionCosines=False)
    SaveReflections(pk, os.path.join(directory,outname+'_'+pk+'_FullProf.int'), Format='Fullprof')

CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace='ratio')

CopySample(InputWorkspace='up',
           OutputWorkspace='ratio',
           CopyName=False,
           CopyMaterial=False,
           CopyEnvironment=False,
           CopyShape=False)

data = {}

for pk in ['up', 'down']:

    for pn in range(mtd[pk].getNumberPeaks()):
        peak = mtd[pk].getPeak(pn)

        hklmnp = np.array(peak.getIntHKL()).astype(int).tolist()+np.array(peak.getIntMNP()).astype(int).tolist()
        hkl = peak.getHKL()

        key = str(hklmnp)

        if data.get(key) is None:
            data[key] = [[hkl],[peak.getIntensity()],[peak.getSigmaIntensity()]]
        else:
            HKL, I, sig = data[key]
            HKL.append(peak.getHKL())
            I.append(peak.getIntensity())
            sig.append(peak.getSigmaIntensity())
            data[key] = [HKL,I,sig]

for key in data.keys():

    HKL, I, sig = data[key]

    if len(HKL) == 2:
        peak = mtd['ratio'].createPeakHKL(HKL[0])
        peak.setIntensity(I[1]/I[0])
        peak.setSigmaIntensity(I[1]/I[0]*np.sqrt((sig[0]/I[0])**2+(sig[1]/I[1])**2))
        mtd['ratio'].addPeak(peak)

SaveHKLCW('ratio', os.path.join(directory,outname+'_ratio.hkl'), DirectionCosines=False, Header=False)
