from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

import os
import sys

from mantid import config
config['Q.convention'] = 'Inelastic'

filename = sys.argv[1]

directory = os.path.dirname('/SNS/software/scd/dev/inscape/polarization/')
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

fullpath = os.path.abspath(filename)
tutorial = '' if not 'shared/examples/IPTS' in fullpath else '/shared/examples' 

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

if normalization.lower() == 'monitor':
    normalize_by = 'Monitor'
elif normalization.lower() == 'time':
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
    return bkg+amp*np.exp(-0.5*(x-mu)**2/sigma**2)

rootname = '/HFIR/HB3A{}/IPTS-{}/'.format(tutorial,ipts)
scanfile = 'HB3A_exp{:04}_scan{:04}'

up_data = [scanfile.format(exp,s) for s in up_run_nos]
down_data = [scanfile.format(exp,s) for s in down_run_nos]

CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace='up')
CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace='down')

for data_up, data_down in zip(up_data, down_data):

    for data in [data_up, data_down]:

        filename = working_directory+'autoreduce/'+data+'.nxs'

        HB3AAdjustSampleNorm(Filename=filename,
                             NormaliseBy=normalize_by,
                             ScaleByMotorStep=scale_by_motor_step,
                             OutputType='Detector',
                             OutputWorkspace=data)

        pws = data+'_pk'

        CreatePeaksWorkspace(InstrumentWorkspace=data, NumberOfPeaks=0, OutputType='Peak', OutputWorkspace='peaks')
        CreatePeaksWorkspace(InstrumentWorkspace=data, NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace=pws)

        run = mtd[data].getExperimentInfo(0).run()
        R = run.getGoniometer(run.getNumGoniometers()//2).getR()

        mtd['peaks'].run().getGoniometer(0).setR(R)
        mtd[pws].run().getGoniometer(0).setR(R)

        CopySample(InputWorkspace=data,
                   OutputWorkspace=pws,
                   CopyName=False,
                   CopyMaterial=False,
                   CopyEnvironment=False,
                   CopyShape=False)

        CopySample(InputWorkspace=data,
                   OutputWorkspace='peaks',
                   CopyName=False,
                   CopyMaterial=False,
                   CopyEnvironment=False,
                   CopyShape=False)

        title = run['scan_title'].value
        hkl = np.array(title.split('(')[-1].split(')')[0].split(' ')).astype(float)

        peak = mtd['peaks'].createPeakHKL([*hkl])

        row, col = peak.getRow(), peak.getCol()

        peak = mtd[pws].createPeakHKL([*hkl])
        mtd[pws].addPeak(peak)

        iws = data+'_int'

        IntegrateMDHistoWorkspace(InputWorkspace=data,
                                  P1Bin='{},{}'.format(row-y_pixels,row+y_pixels),
                                  P2Bin='{},{}'.format(col-x_pixels,col+x_pixels),
                                  OutputWorkspace=iws,
                                  EnableLogging=False)

        ConvertMDHistoToMatrixWorkspace(InputWorkspace=iws, OutputWorkspace=iws, EnableLogging=False)
        ConvertToPointData(InputWorkspace=iws, OutputWorkspace=iws, EnableLogging=False)

        IntegrateMDHistoWorkspace(InputWorkspace=data,
                                  P1Bin='{},0,{}'.format(row-y_pixels,row+y_pixels),
                                  P2Bin='{},0,{}'.format(col-x_pixels,col+x_pixels),
                                  P3Bin='0,{}'.format(mtd[data].getDimension(2).getNBins()),
                                  OutputWorkspace=data+'_ROI', EnableLogging=False)

    x0 = mtd[data_up+'_int'].extractX().flatten()
    y0 = mtd[data_up+'_int'].extractY().flatten()

    x1 = mtd[data_down+'_int'].extractX().flatten()
    y1 = mtd[data_down+'_int'].extractY().flatten()

    if (y0 > 0).sum() > 5 and (y1 > 0).sum() > 5 and y0.size == y1.size:

        multi_domain_function = FunctionFactory.createInitializedMultiDomainFunction('name=CompositeFunction', 2)

        flat = FunctionFactory.createInitialized('name=FlatBackground')
        gauss = FunctionFactory.createInitialized('name=Gaussian')

        composite1 = multi_domain_function.getFunction(0)
        composite1.add(flat)
        composite1.add(gauss)

        composite2 = multi_domain_function.getFunction(1)
        composite2.add(flat)
        composite2.add(gauss)

        mask = np.logical_and(np.isfinite(y0), np.isfinite(y1))

        A0 = (np.min(y0[mask])+np.min(y1[mask]))/2
        mu = (x0[mask][np.argmax(y0[mask])]+x1[mask][np.argmax(y1[mask])])/2
        sigma = (np.sqrt(np.average((x0[mask]-mu)**2, weights=y0[mask]))+np.sqrt(np.average((x1[mask]-mu)**2, weights=y1[mask])))/2

        h0 = np.max(y0[mask])-np.min(y0[mask])
        h1 = np.max(y1[mask])-np.min(y1[mask])

        composite1.setParameter(0, A0)
        composite2.setParameter(0, A0)

        composite1.setParameter(1, h0)
        composite2.setParameter(1, h1)

        composite1.setParameter(2, mu)
        composite2.setParameter(2, mu)

        composite1.setParameter(3, sigma)
        composite2.setParameter(3, sigma)

        function = str(multi_domain_function)
        function += ';ties=(f1.f0.A0=f0.f0.A0,f1.f1.PeakCentre=f0.f1.PeakCentre,f1.f1.Sigma=f0.f1.Sigma)'

        fit_result = ''
        fit_result = Fit(Function=function, 
                         InputWorkspace=data_up+'_int', 
                         InputWorkspace_1=data_down+'_int', 
                         Output='fit',
                         IgnoreInvalidData=True,
                         EnableLogging=False)

        bkg, A0, mu, sigma, _, A1, _, _, _ = fit_result.OutputParameters.toDict()['Value']
        _, errA0, _, errS, _, errA1, _, _, _ = fit_result.OutputParameters.toDict()['Error']

        I0 = A0*sigma*np.sqrt(2*np.pi)
        I1 = A1*sigma*np.sqrt(2*np.pi)

        corr_A0S = np.abs(fit_result.OutputNormalisedCovarianceMatrix.cell(1, 3) / 100
                         *fit_result.OutputParameters.cell(1, 2)
                         *fit_result.OutputParameters.cell(3, 2))

        corr_A1S = np.abs(fit_result.OutputNormalisedCovarianceMatrix.cell(3, 1) / 100
                         *fit_result.OutputParameters.cell(1, 2)
                         *fit_result.OutputParameters.cell(5, 2))

        sig0 = np.sqrt(2*np.pi*(A0**2*errS**2+sigma**2*errA0**2+2*A0*sigma*corr_A0S))
        sig1 = np.sqrt(2*np.pi*(A1**2*errS**2+sigma**2*errA1**2+2*A1*sigma*corr_A1S))

        mtd[data_up+'_pk'].getPeak(0).setIntensity(I0)
        mtd[data_down+'_pk'].getPeak(0).setIntensity(I1)

        mtd[data_up+'_pk'].getPeak(0).setSigmaIntensity(sig0)
        mtd[data_down+'_pk'].getPeak(0).setSigmaIntensity(sig1)

        CombinePeaksWorkspaces(LHSWorkspace=data_up+'_pk', RHSWorkspace='up', OutputWorkspace='up')
        CombinePeaksWorkspaces(LHSWorkspace=data_down+'_pk', RHSWorkspace='down', OutputWorkspace='down')

        CloneWorkspace(InputWorkspace='fit_Parameters', OutputWorkspace=data_up+'_Parameters')
        CloneWorkspace(InputWorkspace='fit_Parameters', OutputWorkspace=data_down+'_Parameters')

        CloneWorkspace(InputWorkspace='fit_Workspaces', OutputWorkspace=data_up+'_Workspaces')
        CloneWorkspace(InputWorkspace='fit_Workspaces', OutputWorkspace=data_down+'_Workspaces')

        CloneWorkspace(InputWorkspace='fit_NormalisedCovarianceMatrix', OutputWorkspace=data_up+'_NormalisedCovarianceMatrix')
        CloneWorkspace(InputWorkspace='fit_NormalisedCovarianceMatrix', OutputWorkspace=data_down+'_NormalisedCovarianceMatrix')

        DeleteWorkspace('fit_Parameters')
        DeleteWorkspace('fit_Workspaces')
        DeleteWorkspace('fit_NormalisedCovarianceMatrix')

    DeleteWorkspace('peaks')

for pk in ['up', 'down']:

    with PdfPages(os.path.join(directory,outname+'_'+pk+'.pdf')) as pdf:

        N = mtd[pk].getNumberPeaks()
        for i in range(N):
            scan = mtd[pk].getPeak(i).getRunNumber()
            hkl = mtd[pk].getPeak(i).getIntHKL()

            fig = plt.figure(figsize=(12,6))
            ax1 = fig.add_subplot(121, projection='mantid')
            ax2 = fig.add_subplot(122, projection='mantid')
            im = ax1.pcolormesh(mtd[scanfile.format(exp,scan)+'_ROI'], transpose=True)
            im.set_edgecolor('face')
            ax1.set_title('({:.0f} {:.0f} {:.0f})'.format(*hkl))
            ax1.minorticks_on()
            ax1.set_aspect(1)
            ax2.errorbar(mtd[scanfile.format(exp,scan)+'_int'], wkspIndex=0, marker='o', linestyle='', label='data')
            if 'Fit' in integration_method:
                bkg, A0, mu, sigma, _, A1, _, _, _  = mtd[scanfile.format(exp,scan)+'_Parameters'].column(1)
                xdim = mtd[scanfile.format(exp,scan)+'_int'].getXDimension()
                x = np.linspace(xdim.getMinimum(), xdim.getMaximum(),500)
                output = [bkg, A0 if 'up' else A1, mu, sigma, 0]
                y = gaussian(x, output)
                ax2.plot(x, y, label='calc')
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
        peak.setIntensity(I[0]/I[1])
        peak.setSigmaIntensity(I[0]/I[1]*np.sqrt((sig[0]/I[0])**2+(sig[1]/I[1])**2))
        mtd['ratio'].addPeak(peak)

SaveHKLCW('ratio', os.path.join(directory,outname+'_ratio.hkl'), DirectionCosines=False, Header=False)
