from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

import os
import sys

from mantid import config
config['Q.convention'] = 'Inelastic'

#print(sys.argv)
filename = sys.argv[1] #'/HFIR/HB3A/IPTS-30682/shared/zgf/test_peak_splitting/input.config'#

directory = os.path.dirname('/SNS/software/scd/reduction/inscape_dev/polarization/')
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

roi_file = dictionary.get('roi-file')
if roi_file is not None:
    roi_file = os.path.join(working_directory, dictionary['roi-file'])

    # scans, xs, ys = np.loadtxt(roi_file, delimiter=',', unpack=True)
    # roi_cntr = {}
    # for scan, x, y in zip(scans, xs, ys):
    #     roi_cntr[int(scan)] = [int(x), int(y)]
    scans, llxs, llys, urxs, urys, *splits = np.loadtxt(roi_file, delimiter=',', unpack=True)
    roi_lims = {}
    for scan, llx, lly, urx, ury in zip(scans, llxs, llys, urxs, urys):
        roi_lims[int(scan)] = [int(llx), int(lly), int(urx), int(ury)]
    split_peak = {}
    if len(splits) > 0:
        splits = splits[0]
        for scan, split in zip(scans, splits):
            split_peak[int(scan)] = True if split == 1 else False
    else:
        for scan in scans:
            split_peak[int(scan)] = False

integration_method = 'Fitted' # 'Fitted', 'Counts', 'CountsWithFitting'
number_of_backgroud_points = 3  # 'Counts' only

apply_lorentz = True
optimize_q_vector = True

scale_factor = dictionary['scale-factor']
min_signal_noise_ratio = dictionary['minimum-signal-noise-ratio']
max_chi_square = dictionary['maximum-chi-square']
common_background = False

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

split = False

for data_up, data_down in zip(up_data, down_data):

    for data in [data_up, data_down]:

        filename = working_directory+'autoreduce/'+data+'.nxs'

        if not mtd.doesExist(data):

            HB3AAdjustSampleNorm(Filename=filename,
                                 NormaliseBy='None',
                                 ScaleByMotorStep=False,
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

        try:
            peak = mtd['peaks'].createPeakHKL([*hkl])
        except:
            print('Skipping peak {} {} {}'.format(*hkl))
            continue

        if roi_file is not None:
            s = int(data.split('scan')[-1])
            # if roi_cntr.get(s) is not None:
            #     col, row = roi_cntr[s]
            split = split_peak.get(s)
            if roi_lims.get(s) is not None:
                 llx, lly, urx, ury = roi_lims[s]
                 col, x_pixels = (urx+llx) // 2, (urx-llx) // 2
                 row, y_pixels = (ury+lly) // 2, (ury-lly) // 2

                 if x_pixels <= 0:
                    col = peak.getCol()
                    x_pixels = dictionary['x-pixels'] 
                 if y_pixels <= 0:
                    row = peak.getRow()
                    y_pixels = dictionary['y-pixels'] 

                 print('Scan {} x-center {} size {}'.format(s, col, x_pixels))
                 print('        y-center {} size {}'.format(row, y_pixels))
            else:
                continue
        else:
            row, col = peak.getRow(), peak.getCol()
            x_pixels = dictionary['x-pixels'] 
            y_pixels = dictionary['y-pixels'] 

        try:
            peak = mtd[pws].createPeakHKL([*hkl])
        except:
            print('Skipping peak {} {} {}'.format(*hkl))
            continue

        mtd[pws].addPeak(peak)
        if split:
            peak.setPeakNumber(1)
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

        scan_log = 'omega' if np.isclose(run.getTimeAveragedStd('phi'), 0.0) else 'phi'
        scan_axis = run[scan_log].value
        scan_step = (scan_axis[-1] - scan_axis[0]) / (scan_axis.size - 1)

        mtd[iws].setX(0, scan_axis)

    # ScaleX(InputWorkspace=data_up+'_int', OutputWorkspace=data_up+'_int', Factor=scan_step)
    # ScaleX(InputWorkspace=data_down+'_int', OutputWorkspace=data_down+'_int', Factor=scan_step)

    x0 = mtd[data_up+'_int'].extractX().flatten()
    y0 = mtd[data_up+'_int'].extractY().flatten()

    x1 = mtd[data_down+'_int'].extractX().flatten()
    y1 = mtd[data_down+'_int'].extractY().flatten()

    #mtd[data_up+'_int'].setE(0,y0)
    #mtd[data_down+'_int'].setE(0,y1)

    if (y0 > 0).sum() > 5 and (y1 > 0).sum() > 5:

        print('Fitting peak {} {} {}'.format(*hkl))

        multi_domain_function = FunctionFactory.createInitializedMultiDomainFunction('name=CompositeFunction', 2)

        flat = FunctionFactory.createInitialized('name=FlatBackground')
        gauss = FunctionFactory.createInitialized('name=Gaussian')

        composite1 = multi_domain_function.getFunction(0)
        composite1.add(flat)
        composite1.add(gauss)

        composite2 = multi_domain_function.getFunction(1)
        composite2.add(flat)
        composite2.add(gauss)

        if split:
            gauss2 = FunctionFactory.createInitialized('name=Gaussian')
            composite1.add(gauss2)
            composite2.add(gauss2)

        mask0 = np.isfinite(y0)
        mask1 = np.isfinite(y1)

        x0min, x1min = np.min(x0[mask0]), np.min(x1[mask1])
        x0max, x1max = np.max(x0[mask0]), np.max(x1[mask1])

        y0min, y1min = np.min(y0[mask0]), np.min(y1[mask1])
        y0max, y1max = np.max(y0[mask0]), np.max(y1[mask1])

        xmin, ymin = np.min([x0min, x1min]), np.min([y0min, y1min])
        xmax, ymax = np.max([x0max, x1max]), np.max([y0max, y1max])

        A0 = (y0min+y1min)/2
        mu = (x0[mask0][np.argmax(y0[mask0])]+x1[mask1][np.argmax(y1[mask1])])/2
        sigma = (np.sqrt(np.average((x0[mask0]-mu)**2, weights=(y0[mask0]-A0)))+np.sqrt(np.average((x1[mask1]-mu)**2, weights=(y1[mask1]-A0))))/2

        h0 = y0max-y0min
        h1 = y1max-y1min

        composite1.setParameter(0, A0)
        composite2.setParameter(0, A0)

        if not split:
            composite1.setParameter(1, h0)
            composite2.setParameter(1, h1)

            composite1.setParameter(2, mu)
            composite2.setParameter(2, mu)

            composite1.setParameter(3, sigma)
            composite2.setParameter(3, sigma)
        else:
            composite1.setParameter(1, h0/2)
            composite2.setParameter(1, h1/2)

            composite1.setParameter(2, mu)
            composite2.setParameter(2, mu)

            composite1.setParameter(3, sigma/2)
            composite2.setParameter(3, sigma/2)

            composite1.setParameter(4, h0/2)
            composite2.setParameter(4, h1/2)

            composite1.setParameter(5, mu)
            composite2.setParameter(5, mu)

            composite1.setParameter(6, sigma/2)
            composite2.setParameter(6, sigma/2)

        function = str(multi_domain_function)
        function += ';ties=(f1.f1.PeakCentre=f0.f1.PeakCentre'
        if split:
            function += ',f1.f2.PeakCentre=f0.f2.PeakCentre,f1.f2.Sigma=f0.f2.Sigma=f1.f1.Sigma=f0.f1.Sigma,f1.f2.Height=f1.f1.Height/f0.f1.Height*f0.f2.Height' #
        else:
            function += ',f1.f1.Sigma=f0.f1.Sigma'
        if common_background:
            function += ',f1.f0.A0=f0.f0.A0'
            print('Using common backgrounds')
        else:
            print('Using different backgrounds')
        function += ')'

        function += ';constraints=(0<f1.f0.A0<{},0<f0.f0.A0<{},{}<f1.f1.PeakCentre<{},{}<f0.f1.PeakCentre<{},0<f1.f1.Height<{},0<f0.f1.Height<{}'.format(2*A0,2*A0,xmin,xmax,xmin,xmax,2*h1,2*h0)
        if split:
            function += ',{}<f1.f2.PeakCentre<{},{}<f0.f2.PeakCentre<{},0<f1.f2.Height<{},0<f0.f2.Height<{}'.format(xmin,xmax,xmin,xmax,2*h1,2*h0)
        function += ',0<f0.f1.Sigma<{}'.format((xmax-xmin)/3,(xmin-xmax)/10,(xmax-xmin)/10)
        function += ')'
        print(function)

        fit_result = ''
        fit_result = Fit(Function=function, 
                         InputWorkspace=data_up+'_int', 
                         InputWorkspace_1=data_down+'_int', 
                         Output='fit',
                         CostFunction='Least squares',
                         Minimizer='Levenberg-Marquardt',
                         IgnoreInvalidData=True,
                         EnableLogging=True)

        p = fit_result.OutputParameters.column(0)

        A0 = fit_result.OutputParameters.cell(p.index('f0.f1.Height'),1)
        A1 = fit_result.OutputParameters.cell(p.index('f1.f1.Height'),1)

        errA0 = fit_result.OutputParameters.cell(p.index('f0.f1.Height'),2)
        errA1 = fit_result.OutputParameters.cell(p.index('f1.f1.Height'),2)

        sigma = fit_result.OutputParameters.cell(p.index('f0.f1.Sigma'),1)
        errS = fit_result.OutputParameters.cell(p.index('f0.f1.Sigma'),2)

        I0 = A0*sigma*np.sqrt(2*np.pi)
        I1 = A1*sigma*np.sqrt(2*np.pi)

        c = fit_result.OutputNormalisedCovarianceMatrix.getColumnNames()
        r = fit_result.OutputNormalisedCovarianceMatrix.column(0)

        corr_A0S = np.abs(fit_result.OutputNormalisedCovarianceMatrix.cell(r.index('f0.f1.Height'),c.index('f0.f1.Sigma'))/100*np.sqrt(A0*sigma))
        corr_A1S = np.abs(fit_result.OutputNormalisedCovarianceMatrix.cell(r.index('f1.f1.Height'),c.index('f0.f1.Sigma'))/100*np.sqrt(A1*sigma))

        sig0 = np.sqrt(2*np.pi*(A0**2*errS**2+sigma**2*errA0**2+2*A0*sigma*corr_A0S))
        sig1 = np.sqrt(2*np.pi*(A1**2*errS**2+sigma**2*errA1**2+2*A1*sigma*corr_A1S))

        mtd[data_up+'_pk'].getPeak(0).setIntensity(I0)
        mtd[data_down+'_pk'].getPeak(0).setIntensity(I1)

        mtd[data_up+'_pk'].getPeak(0).setSigmaIntensity(sig0)
        mtd[data_down+'_pk'].getPeak(0).setSigmaIntensity(sig1)

        mtd[data_up+'_pk'].getPeak(0).setBinCount(errA0/A0)
        mtd[data_down+'_pk'].getPeak(0).setBinCount(errA1/A1)

        if split:

            A01 = fit_result.OutputParameters.cell(p.index('f0.f2.Height'),1)
            A11 = fit_result.OutputParameters.cell(p.index('f1.f2.Height'),1)

            errA01 = fit_result.OutputParameters.cell(p.index('f0.f2.Height'),2)
            errA11 = fit_result.OutputParameters.cell(p.index('f1.f2.Height'),2)

            I01 = A01*sigma*np.sqrt(2*np.pi)
            I11 = A11*sigma*np.sqrt(2*np.pi)

            corr_A01S = np.abs(fit_result.OutputNormalisedCovarianceMatrix.cell(r.index('f0.f2.Height'),c.index('f0.f1.Sigma'))/100*np.sqrt(A01*sigma))
            corr_A11S = np.abs(fit_result.OutputNormalisedCovarianceMatrix.cell(r.index('f0.f2.Height'),c.index('f0.f1.Sigma'))/100*np.sqrt(A11*sigma))

            sig01 = np.sqrt(2*np.pi*(A01**2*errS**2+sigma**2*errA01**2+2*A0*sigma*corr_A01S))
            sig11 = np.sqrt(2*np.pi*(A11**2*errS**2+sigma**2*errA11**2+2*A1*sigma*corr_A11S))

            mtd[data_up+'_pk'].getPeak(1).setIntensity(I01)
            mtd[data_down+'_pk'].getPeak(1).setIntensity(I11)

            mtd[data_up+'_pk'].getPeak(1).setSigmaIntensity(sig01)
            mtd[data_down+'_pk'].getPeak(1).setSigmaIntensity(sig11)

            mtd[data_up+'_pk'].getPeak(1).setBinCount(errA01/A01)
            mtd[data_down+'_pk'].getPeak(1).setBinCount(errA11/A11)

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

    else:

        print('Skipping peak {} {} {}'.format(*hkl))
        print((y0 > 0).sum() > 5, (y1 > 0).sum() > 5, y0.size, y1.size)

    DeleteWorkspace('peaks')

with PdfPages(os.path.join(directory,outname+'.pdf')) as pdf:

    N = mtd['up'].getNumberPeaks()
    for i in range(N):

        scan1 = mtd['up'].getPeak(i).getRunNumber()
        scan2 = mtd['down'].getPeak(i).getRunNumber()

        hkl1 = mtd['up'].getPeak(i).getIntHKL()
        hkl2 = mtd['down'].getPeak(i).getIntHKL()

        peak_no1 = mtd['up'].getPeak(i).getPeakNumber()
        peak_no2 = mtd['down'].getPeak(i).getPeakNumber()

        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(221, projection='mantid')
        ax2 = fig.add_subplot(222, projection='mantid')
        ax3 = fig.add_subplot(223, projection='mantid')
        ax4 = fig.add_subplot(224, projection='mantid')

        im1 = ax1.pcolormesh(mtd[scanfile.format(exp,scan1)+'_ROI'], transpose=True)
        im3 = ax3.pcolormesh(mtd[scanfile.format(exp,scan2)+'_ROI'], transpose=True)
        im1.set_edgecolor('face')
        im3.set_edgecolor('face')

        ax1.set_title('({:.0f} {:.0f} {:.0f})'.format(*hkl1))
        ax3.set_title('({:.0f} {:.0f} {:.0f})'.format(*hkl2))

        ax1.minorticks_on()
        ax2.minorticks_on()
        ax2.minorticks_on()
        ax3.minorticks_on()

        ax1.set_aspect(1)
        ax3.set_aspect(1)

        ax2.errorbar(mtd[scanfile.format(exp,scan1)+'_int'], wkspIndex=0, marker='o', linestyle='', label='up')
        ax4.errorbar(mtd[scanfile.format(exp,scan2)+'_int'], wkspIndex=0, marker='o', linestyle='', label='down')

        if 'Fit' in integration_method:
            p1 = mtd[scanfile.format(exp,scan1)+'_Parameters'].column(0)
            p2 = mtd[scanfile.format(exp,scan2)+'_Parameters'].column(0)

            bkg1 = mtd[scanfile.format(exp,scan1)+'_Parameters'].cell(p1.index('f0.f0.A0'),1)
            bkg2 = mtd[scanfile.format(exp,scan2)+'_Parameters'].cell(p2.index('f1.f0.A0'),1)

            if peak_no1 == 0:

                A1 = mtd[scanfile.format(exp,scan1)+'_Parameters'].cell(p1.index('f0.f1.Height'),1)
                A2 = mtd[scanfile.format(exp,scan2)+'_Parameters'].cell(p2.index('f1.f1.Height'),1)

                mu1 = mtd[scanfile.format(exp,scan1)+'_Parameters'].cell(p1.index('f0.f1.PeakCentre'),1)
                mu2 = mtd[scanfile.format(exp,scan2)+'_Parameters'].cell(p2.index('f1.f1.PeakCentre'),1)

            else:

                A1 = mtd[scanfile.format(exp,scan1)+'_Parameters'].cell(p1.index('f0.f2.Height'),1)
                A2 = mtd[scanfile.format(exp,scan2)+'_Parameters'].cell(p2.index('f1.f2.Height'),1)

                mu1 = mtd[scanfile.format(exp,scan1)+'_Parameters'].cell(p1.index('f0.f2.PeakCentre'),1)
                mu2 = mtd[scanfile.format(exp,scan2)+'_Parameters'].cell(p2.index('f1.f2.PeakCentre'),1)

            sigma1 = mtd[scanfile.format(exp,scan1)+'_Parameters'].cell(p1.index('f0.f1.Sigma'),1)
            sigma2 = mtd[scanfile.format(exp,scan2)+'_Parameters'].cell(p2.index('f1.f1.Sigma'),1)

            x1dim = mtd[scanfile.format(exp,scan1)+'_int'].getXDimension()
            x2dim = mtd[scanfile.format(exp,scan2)+'_int'].getXDimension()

            x1 = np.linspace(x1dim.getMinimum(), x1dim.getMaximum(), 500)
            x2 = np.linspace(x2dim.getMinimum(), x2dim.getMaximum(), 500)

            y1 = gaussian(x1, [bkg1, A1, mu1, sigma1, 0])
            y2 = gaussian(x2, [bkg2, A2, mu2, sigma2, 0])                

            ax2.plot(x1, y1, label='calc')
            ax4.plot(x2, y2, label='calc')

        ax2.legend()
        ax4.legend()

        ax2.set_title('Exp #{}, Scan #{}'.format(exp,scan1))
        ax4.set_title('Exp #{}, Scan #{}'.format(exp,scan2))

        pdf.savefig()
        plt.close()

    #SaveHKLCW(pk, os.path.join(directory,outname+'_'+pk+'_SHELX_dir_cos.hkl'), DirectionCosines=True)
    #SaveHKLCW(pk, os.path.join(directory,outname+'_'+pk+'_SHELX.hkl'), DirectionCosines=False)
    #SaveReflections(pk, os.path.join(directory,outname+'_'+pk+'_FullProf.int'), Format='Fullprof')

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
            data[key] = [[hkl],[peak.getIntensity()],[peak.getBinCount()]]
        else:
            HKL, I, sig = data[key]
            HKL.append(peak.getHKL())
            I.append(peak.getIntensity())
            sig.append(peak.getBinCount())
            data[key] = [HKL,I,sig]

for key in data.keys():

    HKL, I, sig = data[key]

    if len(HKL) == 2:
        peak = mtd['ratio'].createPeakHKL(HKL[0])
        peak.setIntensity(I[0]/I[1])
        peak.setSigmaIntensity(I[0]/I[1]*np.sqrt(sig[0]**2+sig[1]**2))
        mtd['ratio'].addPeak(peak)
    elif len(HKL) == 4:
        peak = mtd['ratio'].createPeakHKL(HKL[0])
        peak.setIntensity(I[0]/I[2])
        peak.setSigmaIntensity(I[0]/I[2]*np.sqrt(sig[0]**2+sig[2]**2))
        if peak.getIntensityOverSigma() > 1:
            mtd['ratio'].addPeak(peak)
        peak = mtd['ratio'].createPeakHKL(HKL[1])
        peak.setIntensity(I[1]/I[3])
        peak.setSigmaIntensity(I[1]/I[3]*np.sqrt(sig[1]**2+sig[3]**2))
        if peak.getIntensityOverSigma() > 1:
            mtd['ratio'].addPeak(peak)

SaveHKLCW('ratio', os.path.join(directory,outname+'_ratio.hkl'), DirectionCosines=False, Header=False)

with open(os.path.join(directory,outname+'_ratio.hkl'), 'w') as f:
    hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:8.4f}{:8.4f}\n'
    for pk in mtd['ratio']:
        h, k, l = pk.getIntHKL()
        I = pk.getIntensity()
        sig = pk.getSigmaIntensity()
        f.write(hkl_format.format(h,k,l,I,sig))
