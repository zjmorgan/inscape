from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

import os
import sys

filename = sys.argv[1] #'/HFIR/HB3A/IPTS-18227/shared/pnd/HB3A.conf'

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

working_directory = '/{}/{}/IPTS-{}/shared/'.format(facility,instrument,ipts)
shared_directory = '/{}/{}/shared/'.format(facility,instrument)

directory = os.path.dirname(os.path.abspath(filename))
outname = dictionary['name']

# data normalization -----------------------------------------------------------
normalize_by = 'Monitor' # 'Time', 'Monitor', 'None'
scale_by_motor_step = True

# integrate peaks --------------------------------------------------------------
lower_left_roi = dictionary['lower-left-roi'] # x pixels, y pixels
upper_right_roi = dictionary['upper-right-roi']  # x pixels, y pixels

integration_method = 'Fitted' # 'Fitted', 'Counts', 'CountsWithFitting'
number_of_backgroud_points = 3  # 'Counts' only

apply_lorentz = True
optimize_q_vector = True

scale_factor = 1e7
min_signal_noise_ratio = 1  # 'Fitted' and 'CountsWithFitting' only
max_chi_square = 100        # 'Fitted' and 'CountsWithFitting' only

# ---

def gaussian(x, parameters):
    bkg, amp, mu, sigma, _ = parameters
    return bkg+amp*np.exp(-(x-mu)**2/sigma**2)

rootname = '/HFIR/HB3A/IPTS-'
scanfile = 'HB3A_exp{:04}_scan{:04}'

filename = rootname+'{}/shared/autoreduce/'+scanfile+'.nxs'
up_data_files = ', '.join([filename.format(ipts,exp,s) for s in up_run_nos])
down_data_files = ', '.join([filename.format(ipts,exp,s) for s in down_run_nos])

HB3AAdjustSampleNorm(Filename=up_data_files,
                     NormaliseBy=normalize_by,
                     ScaleByMotorStep=scale_by_motor_step,
                     OutputType='Detector',
                     OutputWorkspace='up_data')

HB3AAdjustSampleNorm(Filename=down_data_files,
                     NormaliseBy=normalize_by,
                     ScaleByMotorStep=scale_by_motor_step,
                     OutputType='Detector',
                     OutputWorkspace='down_data')

HB3AIntegrateDetectorPeaks(InputWorkspace='up_data',
                           Method=integration_method,
                           NumBackgroundPts=number_of_backgroud_points,
                           LowerLeft=lower_left_roi,
                           UpperRight=upper_right_roi,
                           ChiSqMax=max_chi_square,
                           SignalNoiseMin=min_signal_noise_ratio,
                           ScaleFactor=scale_factor,
                           ApplyLorentz=apply_lorentz,
                           OptimizeQVector=optimize_q_vector,
                           OutputFitResults=True,
                           OutputWorkspace='up_peaks')

HB3AIntegrateDetectorPeaks(InputWorkspace='down_data',
                           Method=integration_method,
                           NumBackgroundPts=number_of_backgroud_points,
                           LowerLeft=lower_left_roi,
                           UpperRight=upper_right_roi,
                           ChiSqMax=max_chi_square,
                           SignalNoiseMin=min_signal_noise_ratio,
                           ScaleFactor=scale_factor,
                           ApplyLorentz=apply_lorentz,
                           OptimizeQVector=optimize_q_vector,
                           OutputFitResults=True,
                           OutputWorkspace='down_peaks')

for pk in ['up', 'down']:

    with PdfPages(os.path.join(directory,outname+'_'+pk+'.pdf')) as pdf:

        N = mtd[pk+'_peaks'].getNumberPeaks()
        for i in range(N):
            scan = mtd[pk+'_peaks'].getPeak(i).getRunNumber()
            hkl = mtd[pk+'_peaks'].getPeak(i).getIntHKL()

            fig = plt.figure(figsize=(12,6))
            ax1 = fig.add_subplot(121, projection='mantid')
            ax2 = fig.add_subplot(122, projection='mantid')
            im = ax1.pcolormesh(mtd[pk+'_peaks_'+pk+'_data_'+scanfile.format(exp,scan)+'_ROI'], transpose=True)
            im.set_edgecolor('face')
            ax1.set_title('({:.0f} {:.0f} {:.0f})'.format(*hkl))
            ax1.minorticks_on()
            ax1.set_aspect(1)
            ax2.errorbar(mtd[pk+'_peaks_'+pk+'_data_'+scanfile.format(exp,scan)+'_Workspace'], wkspIndex=0, marker='o', linestyle='', label='data')
            if 'Fit' in integration_method:
                output = mtd[pk+'_peaks_'+pk+'_data_'+scanfile.format(exp,scan)+'_Parameters'].column(1)
                xdim = mtd[pk+'_peaks_'+pk+'_data_'+scanfile.format(exp,scan)+'_Workspace'].getXDimension()
                x = np.linspace(xdim.getMinimum(), xdim.getMaximum(),500)
                y = gaussian(x, output)
                ax2.plot(x, y, label='calc')
            ax2.plot(mtd[pk+'_peaks_'+pk+'_data_'+scanfile.format(exp,scan)+'_Workspace'], wkspIndex=2, marker='o', linestyle='--', label='diff')
            ax2.legend()
            ax2.set_title('Exp #{}, Scan #{}'.format(exp,scan))
            ax2.minorticks_on()
            pdf.savefig()
            plt.close()

    SaveHKLCW(pk+'_peaks', os.path.join(directory,outname+'_'+pk+'_SHELX_dir_cos.hkl'), DirectionCosines=True)
    SaveHKLCW(pk+'_peaks', os.path.join(directory,outname+'_'+pk+'_SHELX.hkl'), DirectionCosines=False)
    SaveReflections(pk+'_peaks', os.path.join(directory,outname+'_'+pk+'_FullProf.int'), Format='Fullprof')
