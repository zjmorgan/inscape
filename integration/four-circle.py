from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import itertools
from mantid.kernel import V3D

from matplotlib.backends.backend_pdf import PdfPages

import os
import sys

from mantid import config
config['Q.convention'] = 'Inelastic'

filename = sys.argv[1] #/HFIR/HB3A/IPTS-29609/shared/zgf/NaMn6Bi5_60K_test.inp

directory = os.path.dirname(os.path.realpath(__file__)) #/SNS/software/scd/dev/inscape/integration/
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

fullpath = os.path.abspath(filename)
tutorial = '' if not 'shared/examples/IPTS' in fullpath else '/shared/examples' 

import imp
import parameters
import scipy.optimize

imp.reload(parameters)

dictionary = parameters.load_input_file(filename)

run_nos = dictionary['scans'] if type(dictionary['scans']) is list else [dictionary['scans']]

if np.any([type(run) is list for run in run_nos]):  
    run_nos = [run for run_no in run_nos for run in run_no]

facility, instrument = 'HFIR', 'HB3A'
ipts = dictionary['ipts']
exp = dictionary['experiment']

working_directory = '/{}/{}/IPTS-{}/shared/'.format(facility,instrument,ipts)

if dictionary.get('ub-file') is not None:
    ub_file = os.path.join(working_directory, dictionary['ub-file'])
    if '*' in ub_file:
        ub_file = [ub_file.replace('*', str(run)) for run in run_nos]
else:
    ub_file = None

working_directory = '/{}/{}/IPTS-{}/shared/'.format(facility,instrument+tutorial,ipts)
shared_directory = '/{}/{}/shared/'.format(facility,instrument)

directory = os.path.dirname(os.path.abspath(filename))
outname = dictionary['name']

parameters.output_input_file(filename, directory, outname)

# data normalization -----------------------------------------------------------
normalization = dictionary['normalization'].lower()

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

roi_file = dictionary.get('roi-file')
if roi_file is not None:
    roi_file = os.path.join(working_directory, dictionary['roi-file'])
    scans, xs, ys = np.loadtxt(roi_file, delimiter=',', unpack=True)
    roi_cntr = {}
    for scan, x, y in zip(scans, xs, ys):
        roi_cntr[int(scan)] = [int(x), int(y)]

method = dictionary['integration-method'].lower()

if method == 'fitted':
    integration_method = 'Fitted'
elif method == 'counts':
    integration_method = 'Counts'
else:
    integration_method = 'CountsWithFitting'

number_of_background_points = 3  # 'Counts' only
if dictionary.get('background-points') is not None:
    number_of_background_points = dictionary['background-points']

apply_lorentz = True
optimize_q_vector = False

scale_factor = dictionary['scale-factor']
min_signal_noise_ratio = dictionary['minimum-signal-noise-ratio']
max_chi_square = dictionary['maximum-chi-square']

reflection_condition = dictionary['reflection-condition']

if reflection_condition == 'P':
    reflection_condition = 'Primitive'
elif reflection_condition == 'F':
    reflection_condition = 'All-face centred'
elif reflection_condition == 'I':
    reflection_condition = 'Body centred'
elif reflection_condition == 'A':
    reflection_condition = 'A-face centred'
elif reflection_condition == 'B':
    reflection_condition = 'B-face centred'
elif reflection_condition == 'C':
    reflection_condition = 'C-face centred'
elif reflection_condition == 'R' or reflection_condition == 'Robv':
    reflection_condition = 'Rhombohedrally centred, obverse'
elif reflection_condition == 'Rrev':
    reflection_condition = 'Rhombohedrally centred, reverse'
elif reflection_condition == 'H':
     reflection_condition = 'Hexagonally centred, reverse'

cell_type = dictionary.get('cell-type').lower()

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

mod_vector_1 = dictionary['modulation-vector-1']
mod_vector_2 = dictionary['modulation-vector-2']
mod_vector_3 = dictionary['modulation-vector-3']
max_order = dictionary['max-order']
cross_terms = dictionary['cross-terms']

# ---

def gaussian(x, parameters):
    bkg, amp, mu, sigma, _ = parameters
    return bkg+amp*np.exp(-0.5*(x-mu)**2/sigma**2)

rootname = '/HFIR/HB3A{}/IPTS-{}/'.format(tutorial,ipts)
scanfile = 'HB3A_exp{:04}_scan{:04}'

CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace='integrated')

for i, s in enumerate(run_nos[::-1]):

    data = scanfile.format(exp,s)

    filename = rootname+'shared/autoreduce/'+data+'.nxs'

    HB3AAdjustSampleNorm(Filename=filename,
                         NormaliseBy='None',
                         ScaleByMotorStep=False,
                         OutputType='Detector',
                         OutputWorkspace=data)

    CreatePeaksWorkspace(data, NumberOfPeaks=0, OutputType='Peak', OutputWorkspace='tmp')

    if type(ub_file) is list:
        LoadIsawUB(InputWorkspace=data, Filename=ub_file[i])
    elif type(ub_file) is str:
        LoadIsawUB(InputWorkspace=data, Filename=ub_file)
    else:
        UB = mtd[data].getExperimentInfo(0).sample().getOrientedLattice().getUB()
        SetUB(Workspace=data, UB=UB)

    if i == 0:
        CopySample(InputWorkspace=data,
                   OutputWorkspace='integrated',
                   CopyName=False,
                   CopyMaterial=False,
                   CopyEnvironment=False,
                   CopyShape=False)
        SaveIsawUB(InputWorkspace='integrated', Filename=os.path.join(directory,outname+'.mat'))

    if mtd[data].getDimension(2).getNBins() > 5: # bkg, amp, mu, sigma

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

        try:
            peak = mtd['tmp'].createPeakHKL([*hkl])
        except:
            continue

        if roi_file is not None:
            if roi_cntr.get(s) is not None:
                col, row = roi_cntr[s]
            else:
                continue
        else:
            row, col = peak.getRow(), peak.getCol()

        if 'Fit' not in integration_method:
            HB3AIntegrateDetectorPeaks(InputWorkspace=data,
                                       Method='CountsWithFitting',
                                       NumBackgroundPts=number_of_background_points,
                                       LowerLeft=[col-x_pixels,row-y_pixels],
                                       UpperRight=[col+x_pixels,row+y_pixels],
                                       ChiSqMax=np.inf,
                                       SignalNoiseMin=min_signal_noise_ratio,
                                       ScaleFactor=scale_factor,
                                       ApplyLorentz=apply_lorentz,
                                       OptimizeQVector=optimize_q_vector,
                                       OutputFitResults=True,
                                       OutputWorkspace='peaks')
            if mtd['peaks'].getNumberPeaks() == 0:
                continue
       
        HB3AIntegrateDetectorPeaks(InputWorkspace=data,
                                   Method=integration_method,
                                   NumBackgroundPts=number_of_background_points,
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

            ws = mtd['peaks_'+data+'_ROI']

            weights = ws.getSignalArray()

            x = np.linspace(ws.getXDimension().getMinimum(),
                            ws.getXDimension().getMaximum(),
                            ws.getXDimension().getNBoundaries())

            y = np.linspace(ws.getYDimension().getMinimum(),
                            ws.getYDimension().getMaximum(),
                            ws.getYDimension().getNBoundaries())

            x, y = 0.5*(x[1:]+x[:-1]), 0.5*(y[1:]+y[:-1])

            x, y = np.meshgrid(x, y)

            mux = np.average(x.flatten(), weights=weights.flatten())
            muy = np.average(y.flatten(), weights=weights.flatten())

            col, row = int(mux), int(muy)

            ws = mtd['peaks_'+data+'_Parameters']

            ind = int(ws.toDict().get('Value')[2])

            if ind >= run.getNumGoniometers() or ind < 0:
                ind = run.getNumGoniometers()//2

            R = mtd[data].getExperimentInfo(0).run().getGoniometer(ind).getR()
            wl = float(mtd[data].getExperimentInfo(0).run().getProperty('wavelength').value)

            pos = np.array(mtd['tmp'].getInstrument().getComponentByName('panel({},{})'.format(col,row)).getPos())

            vec = pos/np.linalg.norm(pos)

            Qlab = 2*np.pi/wl*(np.array([0,0,1])-vec)

            pk = mtd['peaks'].getPeak(0)
            pk.setQLabFrame(V3D(*Qlab))
            pk.setGoniometerMatrix(R)

            if max_order > 0:

                ol = mtd['peaks'].sample().getOrientedLattice()
                ol.setMaxOrder(max_order)

                ol.setModVec1(V3D(*mod_vector_1))
                ol.setModVec2(V3D(*mod_vector_2))
                ol.setModVec3(V3D(*mod_vector_3))

                UB = ol.getUB()

                mod_HKL = np.column_stack((mod_vector_1,mod_vector_2,mod_vector_3))
                mod_UB = np.dot(UB, mod_HKL)

                ol.setModUB(mod_UB)

                mod_1 = np.linalg.norm(mod_vector_1) > 0
                mod_2 = np.linalg.norm(mod_vector_2) > 0
                mod_3 = np.linalg.norm(mod_vector_3) > 0

                ind_1 = np.arange(-max_order*mod_1,max_order*mod_1+1).tolist()
                ind_2 = np.arange(-max_order*mod_2,max_order*mod_2+1).tolist()
                ind_3 = np.arange(-max_order*mod_3,max_order*mod_3+1).tolist()

                if cross_terms:
                    iter_mnp = list(itertools.product(ind_1,ind_2,ind_3))
                else:
                    iter_mnp = list(set(list(itertools.product(ind_1,[0],[0]))\
                                      + list(itertools.product([0],ind_2,[0]))\
                                      + list(itertools.product([0],[0],ind_3))))

                iter_mnp = [iter_mnp[s] for s in np.lexsort(np.array(iter_mnp).T, axis=0)]

                for pn in range(mtd['peaks'].getNumberPeaks()):
                    pk = mtd['peaks'].getPeak(pn)
                    hkl = pk.getHKL()
                    for m, n, p in iter_mnp:
                        d_hkl = m*np.array(mod_vector_1)\
                              + n*np.array(mod_vector_2)\
                              + p*np.array(mod_vector_3)
                        HKL = np.round(hkl-d_hkl,2)
                        mnp = [m,n,p]
                        H, K, L = HKL
                        h, k, l = int(H), int(K), int(L)
                        if reflection_condition == 'Primitive':
                            allowed = True
                        elif reflection_condition == 'C-face centred':
                            allowed (h + l) % 2 == 0
                        elif reflection_condition == 'Body centred':
                            allowed = (h + k + l) % 2 == 0
                        elif reflection_condition == 'All-face centred':
                            allowed = (h + l) % 2 == 0 and (k + l) % 2 == 0 and (h + k) % 2 == 0
                        elif reflection_condition == 'Rhombohedrally centred, obverse':
                            allowed = (-h + k + l) % 3 == 0
                        elif reflection_condition == 'Rhombohedrally centred, reverse':
                            allowed = (h - k + l) % 3 == 0
                        elif reflection_condition == 'Hexagonally centred, reverse':
                            allowed = (h - k) % 3 == 0
                        if np.isclose(np.linalg.norm(np.mod(HKL,1)), 0) and allowed:
                            HKL = HKL.astype(int).tolist()
                            pk.setIntMNP(V3D(*mnp))
                            pk.setIntHKL(V3D(*HKL))

            CombinePeaksWorkspaces(LHSWorkspace='peaks', RHSWorkspace='integrated', OutputWorkspace='integrated')

            DeleteWorkspace('peaks')
            DeleteWorkspace('tmp')

            if mtd['peaks_fit_results'].size() > 0:
                RenameWorkspace(InputWorkspace='peaks_fit_results', OutputWorkspace=data+'_fit_results')
            else:
                DeleteWorkspace('peaks_fit_results')

with PdfPages(os.path.join(directory,outname+'.pdf')) as pdf:

    N = mtd['integrated'].getNumberPeaks()
    for i in range(N):
        scan = mtd['integrated'].getPeak(i).getRunNumber()
        hkl = mtd['integrated'].getPeak(i).getHKL()

        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121, projection='mantid')
        ax2 = fig.add_subplot(122, projection='mantid')
        im = ax1.pcolormesh(mtd['peaks_'+scanfile.format(exp,scan)+'_ROI'], transpose=True)
        im.set_edgecolor('face')
        ax1.set_title('({:.4f} {:.4f} {:.4f})'.format(*hkl))
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

if max_order > 0:

    ol = mtd['integrated'].sample().getOrientedLattice()
    ol.setMaxOrder(max_order)

    ol.setModVec1(V3D(*mod_vector_1))
    ol.setModVec2(V3D(*mod_vector_2))
    ol.setModVec3(V3D(*mod_vector_3))

    UB = ol.getUB()

    mod_HKL = np.column_stack((mod_vector_1,mod_vector_2,mod_vector_3))
    mod_UB = np.dot(UB, mod_HKL)

    ol.setModUB(mod_UB)

    mod_1 = np.linalg.norm(mod_vector_1) > 0
    mod_2 = np.linalg.norm(mod_vector_2) > 0
    mod_3 = np.linalg.norm(mod_vector_3) > 0

    ind_1 = np.arange(-max_order*mod_1,max_order*mod_1+1).tolist()
    ind_2 = np.arange(-max_order*mod_2,max_order*mod_2+1).tolist()
    ind_3 = np.arange(-max_order*mod_3,max_order*mod_3+1).tolist()

    if cross_terms:
        iter_mnp = list(itertools.product(ind_1,ind_2,ind_3))
    else:
        iter_mnp = list(set(list(itertools.product(ind_1,[0],[0]))\
                          + list(itertools.product([0],ind_2,[0]))\
                          + list(itertools.product([0],[0],ind_3))))

    iter_mnp = [iter_mnp[s] for s in np.lexsort(np.array(iter_mnp).T, axis=0)]

    for pn in range(mtd['integrated'].getNumberPeaks()):
        pk = mtd['integrated'].getPeak(pn)
        hkl = pk.getHKL()
        for m, n, p in iter_mnp:
            d_hkl = m*np.array(mod_vector_1)\
                  + n*np.array(mod_vector_2)\
                  + p*np.array(mod_vector_3)
            HKL = np.round(hkl-d_hkl,2)
            mnp = [m,n,p]
            H, K, L = HKL
            h, k, l = int(H), int(K), int(L)
            if reflection_condition == 'Primitive':
                allowed = True
            elif reflection_condition == 'C-face centred':
                allowed (h + l) % 2 == 0
            elif reflection_condition == 'Body centred':
                allowed = (h + k + l) % 2 == 0
            elif reflection_condition == 'All-face centred':
                allowed = (h + l) % 2 == 0 and (k + l) % 2 == 0 and (h + k) % 2 == 0
            elif reflection_condition == 'Rhombohedrally centred, obverse':
                allowed = (-h + k + l) % 3 == 0
            elif reflection_condition == 'Rhombohedrally centred, reverse':
                allowed = (h - k + l) % 3 == 0
            elif reflection_condition == 'Hexagonally centred, reverse':
                allowed = (h - k) % 3 == 0
            if np.isclose(np.linalg.norm(np.mod(HKL,1)), 0) and allowed:
                HKL = HKL.astype(int).tolist()
                pk.setIntMNP(V3D(*mnp))
                pk.setIntHKL(V3D(*HKL))

n = mtd['integrated'].getNumberPeaks()

if n > 0:

    #SaveHKLCW('integrated', os.path.join(directory,outname+'_SHELX_dir_cos.hkl'), DirectionCosines=True)
    #SaveHKLCW('integrated', os.path.join(directory,outname+'_SHELX.hkl'), DirectionCosines=False)
    SaveReflections('integrated', os.path.join(directory,outname+'_FullProf.int'), Format='Fullprof')

    for appname in ['_SHELX.hkl', '_SHELX_dir_cos.hkl']:

        dir_cos = 'dir_cos' in appname

        with open(os.path.join(directory,outname+appname), 'w') as f:

            if dir_cos:
                UB = mtd['integrated'].sample().getOrientedLattice().getUB()
                for p in mtd['integrated']:
                    R = p.getGoniometerMatrix()

                    two_theta = p.getScattering()
                    az_phi = p.getAzimuthal()

                    t1 = UB[:,0].copy()
                    t2 = UB[:,1].copy()
                    t3 = UB[:,2].copy()

                    t1 /= np.linalg.norm(t1)
                    t2 /= np.linalg.norm(t2)
                    t3 /= np.linalg.norm(t3)

                    up = np.dot(R.T, [0,0,-1])
                    us = np.dot(R.T, [np.sin(two_theta)*np.cos(az_phi),np.sin(two_theta)*np.sin(az_phi),np.cos(two_theta)])

                    dir_cos_1 = np.dot(up,t1), np.dot(up,t2), np.dot(up,t3)
                    dir_cos_2 = np.dot(us,t1), np.dot(us,t2), np.dot(us,t3)

                    f.write(
                        "{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:4d}{:8.5f}{:8.5f}{:8.5f}{:8.5f}{:8.5f}{:8.5f}\n"
                        .format(p.getH(), p.getK(), p.getL(), p.getIntensity(),
                                p.getSigmaIntensity(), 1, dir_cos_1[0], dir_cos_2[0], dir_cos_1[1],
                                dir_cos_2[1], dir_cos_1[2], dir_cos_2[2]))
            else:
                for p in mtd['integrated']:
                    f.write("{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:4d}\n".format(
                        p.getH(), p.getK(), p.getL(), p.getIntensity(), p.getSigmaIntensity(), 1))

def __U_matrix(phi, theta, omega):

    ux = np.cos(phi)*np.sin(theta)
    uy = np.sin(phi)*np.sin(theta)
    uz = np.cos(theta)

    U = np.array([[np.cos(omega)+ux**2*(1-np.cos(omega)), ux*uy*(1-np.cos(omega))-uz*np.sin(omega), ux*uz*(1-np.cos(omega))+uy*np.sin(omega)],
                  [uy*ux*(1-np.cos(omega))+uz*np.sin(omega), np.cos(omega)+uy**2*(1-np.cos(omega)), uy*uz*(1-np.cos(omega))-ux*np.sin(omega)],
                  [uz*ux*(1-np.cos(omega))-uy*np.sin(omega), uz*uy*(1-np.cos(omega))+ux*np.sin(omega), np.cos(omega)+uz**2*(1-np.cos(omega))]])

    return U

def __B_matrix(a, b, c, alpha, beta, gamma):

    G = np.array([[a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
                  [b*a*np.cos(gamma), b**2, b*c*np.cos(alpha)],
                  [c*a*np.cos(beta), c*b*np.cos(alpha), c**2]])

    B = scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

    return B

def __cub(x):

    a, *params = x

    return (a, a, a, np.pi/2, np.pi/2, np.pi/2, *params)

def __rhom(x):

    a, alpha, *params = x

    return (a, a, a, alpha, alpha, alpha, *params)

def __tet(x):

    a, c, *params = x

    return (a, a, c, np.pi/2, np.pi/2, np.pi/2, *params)

def __hex(x):

    a, c, *params = x

    return (a, a, c, np.pi/2, np.pi/2, 2*np.pi/3, *params)

def __ortho(x):

    a, b, c, *params = x

    return (a, b, c, np.pi/2, np.pi/2, np.pi/2, *params)

def __mono1(x):

    a, b, c, gamma, *params = x

    return (a, b, c, np.pi/2, np.pi/2, gamma, *params)

def __mono2(x):

    a, b, c, beta, *params = x

    return (a, b, c, np.pi/2, beta, np.pi/2, *params)

def __tri(x):

    a, b, c, alpha, beta, gamma, *params = x

    return (a, b, c, alpha, beta, gamma, *params)

def __res(x, hkl, Q, fun):

    a, b, c, alpha, beta, gamma, phi, theta, omega = fun(x)

    B = __B_matrix(a, b, c, alpha, beta, gamma)
    U = __U_matrix(phi, theta, omega)

    UB = np.dot(U,B)

    return (np.einsum('ij,lj->li', UB, hkl)*2*np.pi-Q).flatten()

CloneWorkspace('integrated', OutputWorkspace='cal')

FilterPeaks(InputWorkspace='cal',
            FilterVariable='Signal/Noise',
            FilterValue=min_signal_noise_ratio,
            Operator='>',
            OutputWorkspace='cal')

cal = mtd['cal']

n = cal.getNumberPeaks()

print('\n')
print('Number of peaks to calculate UB = {}'.format(n))
print('\n')

if n >= 10:

    ol = cal.sample().getOrientedLattice()

    mod_vec_1 = ol.getModVec(0)
    mod_vec_2 = ol.getModVec(1)
    mod_vec_3 = ol.getModVec(2)

    max_order = ol.getMaxOrder()

    Q, hkl = [], []

    for pn in range(n):
        pk = cal.getPeak(pn)
        hkl.append(pk.getHKL())
        Q.append(pk.getQSampleFrame())

    Q, hkl = np.array(Q), np.array(hkl)

    a, b, c, alpha, beta, gamma = ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma()

    if (np.allclose([a, b], c) and np.allclose([alpha, beta, gamma], 90)) or cell_type == 'Cubic':
        fun = __cub
        x0 = (a, )
    elif (np.allclose([a, b], c) and np.allclose([alpha, beta], gamma)) or cell_type == 'Rhombohedral':
        fun = __rhom
        x0 = (a, np.deg2rad(alpha))
    elif (np.isclose(a, b) and np.allclose([alpha, beta, gamma], 90)) or cell_type == 'Tetragonal':
        fun = __tet
        x0 = (a, c)
    elif (np.isclose(a, b) and np.allclose([alpha, beta], 90) and np.isclose(gamma, 120)) or cell_type == 'Hexagonal' or cell_type == 'Trigonal':
        fun = __hex
        x0 = (a, c)
    elif (np.allclose([alpha, beta, gamma], 90)) or cell_type == 'Orthorhombic':
        fun = __ortho
        x0 = (a, b, c)
    elif np.allclose([alpha, gamma], 90) or cell_type == 'Monoclinic':
        fun = __mono2
        x0 = (a, b, c, np.deg2rad(beta))
    elif np.allclose([alpha, beta], 90) or cell_type == 'Monoclinic2':
        fun = __mono1
        x0 = (a, b, c, np.deg2rad(gamma))
    else:
        fun = __tri
        x0 = (a, b, c, np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma))

    U = ol.getU()

    omega = np.arccos((np.trace(U)-1)/2)

    val, vec = np.linalg.eig(U)

    ux, uy, uz = vec[:,np.argwhere(np.isclose(val, 1))[0][0]].real

    theta = np.arccos(uz)
    phi = np.arctan2(uy,ux)

    sol = scipy.optimize.least_squares(__res, x0=x0+(phi,theta,omega), args=(hkl,Q,fun))

    a, b, c, alpha, beta, gamma, phi, theta, omega = fun(sol.x)

    B = __B_matrix(a, b, c, alpha, beta, gamma)
    U = __U_matrix(phi, theta, omega)

    UB = np.dot(U,B)

    SetUB(Workspace='cal', UB=UB)

    ol.setMaxOrder(max_order)

    ol.setModVec1(V3D(*mod_vector_1))
    ol.setModVec2(V3D(*mod_vector_2))
    ol.setModVec3(V3D(*mod_vector_3))

    mod_HKL = np.column_stack((mod_vector_1,mod_vector_2,mod_vector_3))
    mod_UB = np.dot(UB, mod_HKL)

    ol.setModUB(mod_UB)

    SaveIsawUB(InputWorkspace='cal', Filename=os.path.join(directory,outname+'_cal.mat'))
