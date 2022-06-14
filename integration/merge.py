from mantid.simpleapi import *

from mantid.kernel import V3D, FloatTimeSeriesProperty
from mantid.geometry import Goniometer

import os
import re
import psutil
import itertools

import numpy as np

from scipy.ndimage import gaussian_filter

import peak
from peak import PeakEnvelope, PeakDictionary

import fitting
from fitting import Ellipsoid, Profile, Projection

def box_integrator(facility, instrument, runs, banks, indices, Q0, key, binsize=0.01, D=np.diag([1,1,1]), W=np.eye(3), exp=None):

    for j, (r, b, i) in enumerate(zip(runs, banks, indices)):

        if exp is None:
            ows = '{}_{}_{}'.format(instrument,r,b)
        else:
            ows = '{}_{}_{}_{}'.format(instrument,exp,r,i)

        omd = ows+'_md'

        if j == 0:

            dQ = 1/np.sqrt(np.diag(D))

            Q_rot = np.dot(W.T,Q0)

            dQp = np.array([binsize,binsize,binsize])

            extents = [[Q_rot[0]-dQ[0],Q_rot[0]+dQ[0]],
                       [Q_rot[1]-dQ[1],Q_rot[1]+dQ[1]],
                       [Q_rot[2]-dQ[2],Q_rot[2]+dQ[2]]]

            bins = [int(round(2*dQ[0]/dQp[0]))+1,
                    int(round(2*dQ[1]/dQp[1]))+1,
                    int(round(2*dQ[2]/dQp[2]))+1]

            steps = [(extents[0][1]-extents[0][0])/bins[0],
                     (extents[1][1]-extents[1][0])/bins[1],
                     (extents[2][1]-extents[2][0])/bins[2]]

            Q0_bin = [extents[0][0],steps[0],extents[0][1]]
            Q1_bin = [extents[1][0],steps[1],extents[1][1]]
            Q2_bin = [extents[2][0],steps[2],extents[2][1]]
        
        ws = omd if facility == 'SNS' else ows

        SetUB(Workspace=ws, UB=np.eye(3)/(2*np.pi)) # hack to transform axes

        if facility == 'SNS':

            MDNorm(InputWorkspace=omd,
                   SolidAngleWorkspace='sa',
                   FluxWorkspace='flux',
                   RLU=True, # not actually HKL
                   QDimension0='{},{},{}'.format(*W[:,0]),
                   QDimension1='{},{},{}'.format(*W[:,1]),
                   QDimension2='{},{},{}'.format(*W[:,2]),
                   Dimension0Name='QDimension0',
                   Dimension1Name='QDimension1',
                   Dimension2Name='QDimension2',
                   Dimension0Binning='{},{},{}'.format(*Q0_bin),
                   Dimension1Binning='{},{},{}'.format(*Q1_bin),
                   Dimension2Binning='{},{},{}'.format(*Q2_bin),
                   OutputWorkspace='__normDataMD',
                   OutputDataWorkspace='__tmpDataMD',
                   OutputNormalizationWorkspace='__tmpNormMD')

            DeleteWorkspace('__normDataMD')

        else:

            lamda = float(mtd[ows].getExperimentInfo(0).run().getProperty('wavelength').value)

            ConvertWANDSCDtoQ(InputWorkspace=ows,
                              NormalisationWorkspace='van_{}'.format(b),
                              UBWorkspace=ows,
                              OutputWorkspace='__tmp',
                              Wavelength=lamda,
                              NormaliseBy='Monitor',
                              Frame='HKL', # not actually HKL,
                              KeepTemporaryWorkspaces=True,
                              Uproj='{},{},{}'.format(*W[:,0]),
                              Vproj='{},{},{}'.format(*W[:,1]),
                              Wproj='{},{},{}'.format(*W[:,2]),
                              BinningDim0='{},{},{}'.format(Q0_bin[0],Q0_bin[2],bins[0]),
                              BinningDim1='{},{},{}'.format(Q1_bin[0],Q1_bin[2],bins[1]),
                              BinningDim2='{},{},{}'.format(Q2_bin[0],Q2_bin[2],bins[2]))
                              
            DeleteWorkspace('__tmp')

            RenameWorkspace(InputWorkspace='__tmp_data', OutputWorkspace='__tmpDataMD')
            RenameWorkspace(InputWorkspace='__tmp_normalization', OutputWorkspace='__tmpNormMD')

            scale = float(mtd['van_{}'.format(b)].getExperimentInfo(0).run().getProperty('monitor').value)

            mtd['__tmpDataMD'] /= scale 
            mtd['__tmpNormMD'] /= scale 

        if j == 0:
            CloneMDWorkspace(InputWorkspace='__tmpDataMD', OutputWorkspace='dataMD')
            CloneMDWorkspace(InputWorkspace='__tmpNormMD', OutputWorkspace='normMD')
        else:
            PlusMD(LHSWorkspace='dataMD', RHSWorkspace='__tmpDataMD', OutputWorkspace='dataMD')
            PlusMD(LHSWorkspace='normMD', RHSWorkspace='__tmpNormMD', OutputWorkspace='normMD')

    DivideMD(LHSWorkspace='dataMD', RHSWorkspace='normMD', OutputWorkspace='normDataMD')

    SetMDFrame(InputWorkspace='dataMD', MDFrame='QSample', Axes=[0,1,2])
    SetMDFrame(InputWorkspace='normMD', MDFrame='QSample', Axes=[0,1,2])
    SetMDFrame(InputWorkspace='normDataMD', MDFrame='QSample', Axes=[0,1,2])

    mtd['dataMD'].clearOriginalWorkspaces()
    mtd['normMD'].clearOriginalWorkspaces()
    mtd['normDataMD'].clearOriginalWorkspaces()
    
    # SaveMD(InputWorkspace='dataMD', Filename='/tmp/dataMD_'+str(key)+'.nxs')
    # SaveMD(InputWorkspace='normMD', Filename='/tmp/normMD_'+str(key)+'.nxs')
    # SaveMD(InputWorkspace='normDataMD', Filename='/tmp/normDataMD_'+str(key)+'.nxs')

    Q0axis = mtd['normDataMD'].getXDimension()
    Q1axis = mtd['normDataMD'].getYDimension()
    Q2axis = mtd['normDataMD'].getZDimension()

    Q0 = np.linspace(Q0axis.getMinimum(), Q0axis.getMaximum(), Q0axis.getNBins())
    Q1 = np.linspace(Q1axis.getMinimum(), Q1axis.getMaximum(), Q1axis.getNBins())
    Q2 = np.linspace(Q2axis.getMinimum(), Q2axis.getMaximum(), Q2axis.getNBins())

    Q0, Q1, Q2 = np.meshgrid(Q0, Q1, Q2, indexing='ij', copy=False)

    Qx = W[0,0]*Q0+W[0,1]*Q1+W[0,2]*Q2
    Qy = W[1,0]*Q0+W[1,1]*Q1+W[1,2]*Q2
    Qz = W[2,0]*Q0+W[2,1]*Q1+W[2,2]*Q2

    mask = mtd['normMD'].getSignalArray() > 0

    Q = np.sqrt(Qx**2+Qy**2+Qz**2)

    Q, Qx, Qy, Qz = Q[mask], Qx[mask], Qy[mask], Qz[mask]

    data = mtd['dataMD'].getSignalArray().copy()[mask]
    norm = mtd['normMD'].getSignalArray().copy()[mask]

    return Q, Qx, Qy, Qz, data, norm

def partial_integration(signal, Q0, Q1, Q2, Q_rot, D_pk, D_bkg_in, D_bkg_out):

    mask = D_pk[0,0]*(Q0-Q_rot[0])**2\
         + D_pk[1,1]*(Q1-Q_rot[1])**2\
         + D_pk[2,2]*(Q2-Q_rot[2])**2 <= 1

    pk = signal[mask].astype(float)

    pk_Q0, pk_Q1, pk_Q2 = Q0[mask], Q1[mask], Q2[mask]

    mask = (D_bkg_in[0,0]*(Q0-Q_rot[0])**2\
           +D_bkg_in[1,1]*(Q1-Q_rot[1])**2\
           +D_bkg_in[2,2]*(Q2-Q_rot[2])**2 > 1)\
         & (D_bkg_out[0,0]*(Q0-Q_rot[0])**2\
           +D_bkg_out[1,1]*(Q1-Q_rot[1])**2\
           +D_bkg_out[2,2]*(Q2-Q_rot[2])**2 <= 1)

    bkg = signal[mask].astype(float)

    bkg_Q0, bkg_Q1, bkg_Q2 = Q0[mask], Q1[mask], Q2[mask]

    return pk, pk_Q0, pk_Q1, pk_Q2, bkg, bkg_Q0, bkg_Q1, bkg_Q2

def norm_integrator(facility, instrument, runs, banks, indices, Q0, D, W, bin_size=0.013, box_size=1.65,
                    peak_ellipsoid=1.1, inner_bkg_ellipsoid=1.3, outer_bkg_ellipsoid=1.5, exp=None):

    principal_radii = 1/np.sqrt(D.diagonal())

    dQ = box_size*principal_radii

    dQp = np.array([bin_size,bin_size,bin_size])

    D_pk = D/peak_ellipsoid**2
    D_bkg_in = D/inner_bkg_ellipsoid**2
    D_bkg_out = D/outer_bkg_ellipsoid**2

    Q_rot = np.dot(W.T,Q0)

    _, Q0_bin_size = np.linspace(Q_rot[0]-dQ[0],Q_rot[0]+dQ[0], 11, retstep=True)
    _, Q1_bin_size = np.linspace(Q_rot[1]-dQ[1],Q_rot[1]+dQ[1], 11, retstep=True)
    _, Q2_bin_size = np.linspace(Q_rot[2]-dQ[2],Q_rot[2]+dQ[2], 27, retstep=True)

    if not np.isclose(Q0_bin_size, 0):
        dQp[0] = np.min([Q0_bin_size,bin_size])
    if not np.isclose(Q1_bin_size, 0):
        dQp[1] = np.min([Q1_bin_size,bin_size])
    dQp[2] = Q2_bin_size

    # dQp = [0.02,0.02,0.02]

    pk_data, pk_norm = [], []
    bkg_data, bkg_norm = [], []

    if mtd.doesExist('dataMD'): DeleteWorkspace('dataMD')
    if mtd.doesExist('normMD'): DeleteWorkspace('normMD')

    for j, (r, b, i) in enumerate(zip(runs, banks, indices)):

        if facility == 'SNS':
            ows = '{}_{}_{}'.format(instrument,r,b)
        else:
            ows = '{}_{}_{}_{}'.format(instrument,exp,r,i)

        omd = ows+'_md'

        if mtd.doesExist('tmpDataMD'): DeleteWorkspace('tmpDataMD')
        if mtd.doesExist('tmpNormMD'): DeleteWorkspace('tmpNormMD')

        ws = omd if facility == 'SNS' else ows

        SetUB(Workspace=ws, UB=np.eye(3)/(2*np.pi)) # hack to transform axes

        if j == 0:

            extents = [[Q_rot[0]-dQ[0],Q_rot[0]+dQ[0]],
                       [Q_rot[1]-dQ[1],Q_rot[1]+dQ[1]],
                       [Q_rot[2]-dQ[2],Q_rot[2]+dQ[2]]]

            bins = [int(round(2*dQ[0]/dQp[0]))+1,
                    int(round(2*dQ[1]/dQp[1]))+1,
                    int(round(2*dQ[2]/dQp[2]))+1]

            steps = [(extents[0][1]-extents[0][0])/bins[0],
                     (extents[1][1]-extents[1][0])/bins[1],
                     (extents[2][1]-extents[2][0])/bins[2]]

            Q0_bin = [extents[0][0],steps[0],extents[0][1]]
            Q1_bin = [extents[1][0],steps[1],extents[1][1]]
            Q2_bin = [extents[2][0],steps[2],extents[2][1]]

            print('\tdQp = ', dQp)
            print('\tPeak radius = ', 1/np.sqrt(D_pk.diagonal()))
            print('\tInner radius = ', 1/np.sqrt(D_bkg_in.diagonal()))
            print('\tOuter radius = ', 1/np.sqrt(D_bkg_out.diagonal()))

            print('\tQ0_bin', Q0_bin)
            print('\tQ1_bin', Q1_bin)
            print('\tQ2_bin', Q2_bin)

        if facility == 'SNS':

            MDNorm(InputWorkspace=omd,
                   SolidAngleWorkspace='sa',
                   FluxWorkspace='flux',
                   RLU=True, # not actually HKL
                   QDimension0='{},{},{}'.format(*W[:,0]),
                   QDimension1='{},{},{}'.format(*W[:,1]),
                   QDimension2='{},{},{}'.format(*W[:,2]),
                   Dimension0Name='QDimension0',
                   Dimension1Name='QDimension1',
                   Dimension2Name='QDimension2',
                   Dimension0Binning='{},{},{}'.format(*Q0_bin),
                   Dimension1Binning='{},{},{}'.format(*Q1_bin),
                   Dimension2Binning='{},{},{}'.format(*Q2_bin),
                   OutputWorkspace='__normDataMD',
                   OutputDataWorkspace='__tmpDataMD',
                   OutputNormalizationWorkspace='__tmpNormMD')

        else:

            lamda = float(mtd[ows].getExperimentInfo(0).run().getProperty('wavelength').value)

            #monitor = float(mtd[ows].getExperimentInfo(0).run().getProperty('monitor').value)

            #CloneMDWorkspace(InputWorkspace=ows, OutputWorkspace='tmp')
            #CloneMDWorkspace(InputWorkspace=ows, OutputWorkspace='tmp')

            ConvertWANDSCDtoQ(InputWorkspace=ows,
                              NormalisationWorkspace='van_{}'.format(b),
                              UBWorkspace=ows,
                              OutputWorkspace='__normDataMD',
                              Wavelength=lamda,
                              NormaliseBy='Monitor',
                              Frame='HKL', # not actually HKL,
                              KeepTemporaryWorkspaces=True,
                              Uproj='{},{},{}'.format(*W[:,0]),
                              Vproj='{},{},{}'.format(*W[:,1]),
                              Wproj='{},{},{}'.format(*W[:,2]),
                              BinningDim0='{},{},{}'.format(Q0_bin[0],Q0_bin[2],bins[0]),
                              BinningDim1='{},{},{}'.format(Q1_bin[0],Q1_bin[2],bins[1]),
                              BinningDim2='{},{},{}'.format(Q2_bin[0],Q2_bin[2],bins[2]))

            RenameWorkspace(InputWorkspace='__normDataMD_data', OutputWorkspace='tmpDataMD')
            RenameWorkspace(InputWorkspace='__normDataMD_normalization', OutputWorkspace='tmpNormMD')

            scale = float(mtd['van_{}'.format(b)].getExperimentInfo(0).run().getProperty('monitor').value)

            mtd['tmpDataMD'] /= scale 
            mtd['tmpNormMD'] /= scale 

        if j == 0:

            Q0axis = mtd['__normDataMD'].getXDimension()
            Q1axis = mtd['__normDataMD'].getYDimension()
            Q2axis = mtd['__normDataMD'].getZDimension()

            Q0, Q0h = np.linspace(Q0axis.getMinimum(), Q0axis.getMaximum(), Q0axis.getNBins()+1, retstep=True)
            Q1, Q1h = np.linspace(Q1axis.getMinimum(), Q1axis.getMaximum(), Q1axis.getNBins()+1, retstep=True)
            Q2, Q2h = np.linspace(Q2axis.getMinimum(), Q2axis.getMaximum(), Q2axis.getNBins()+1, retstep=True)

            Q0, Q1, Q2 = 0.5*(Q0[:-1]+Q0[1:]), 0.5*(Q1[:-1]+Q1[1:]), 0.5*(Q2[:-1]+Q2[1:])

            dQp_bin= np.array([Q0h,Q1h,Q2h])

            Q0, Q1, Q2 = np.meshgrid(Q0, Q1, Q2, indexing='ij', copy=False)

        if facility == 'SNS':

            RenameWorkspace(InputWorkspace='__tmpDataMD', OutputWorkspace='tmpDataMD')
            RenameWorkspace(InputWorkspace='__tmpNormMD', OutputWorkspace='tmpNormMD')
            RenameWorkspace(InputWorkspace='__normDataMD', OutputWorkspace='normDataMD')

        signal = mtd['tmpDataMD'].getSignalArray().copy()

        pk, pk_Q0, pk_Q1, pk_Q2, bkg, bkg_Q0, bkg_Q1, bkg_Q2 = partial_integration(signal, Q0, Q1, Q2, Q_rot, D_pk, D_bkg_in, D_bkg_out)

        pk_data.append(pk)
        bkg_data.append(bkg)

        signal = mtd['tmpNormMD'].getSignalArray().copy()

        pk, pk_Q0, pk_Q1, pk_Q2, bkg, bkg_Q0, bkg_Q1, bkg_Q2 = partial_integration(signal, Q0, Q1, Q2, Q_rot, D_pk, D_bkg_in, D_bkg_out)

        pk_norm.append(pk)
        bkg_norm.append(bkg)

        if j == 0:
            CloneMDWorkspace(InputWorkspace='tmpDataMD', OutputWorkspace='dataMD')
            CloneMDWorkspace(InputWorkspace='tmpNormMD', OutputWorkspace='normMD')
        else:
            PlusMD(LHSWorkspace='dataMD', RHSWorkspace='tmpDataMD', OutputWorkspace='dataMD')
            PlusMD(LHSWorkspace='normMD', RHSWorkspace='tmpNormMD', OutputWorkspace='normMD')

    DivideMD(LHSWorkspace='dataMD', RHSWorkspace='normMD', OutputWorkspace='normDataMD')

    signal = mtd['normDataMD'].getSignalArray().copy()

    data = (pk_data, pk_norm, bkg_data, bkg_norm, dQp_bin, pk_Q0, pk_Q1, pk_Q2, bkg_Q0, bkg_Q1, bkg_Q2)

    envelope_scales = np.array([peak_ellipsoid, inner_bkg_ellipsoid, outer_bkg_ellipsoid])

    Q_bin = (Q0_bin, Q1_bin, Q2_bin)

    return signal, Q_bin, Q_rot, principal_radii, envelope_scales, data

def load_normalization_calibration(facility, instrument, spectrum_file, counts_file,
                                   tube_calibration, detector_calibration):

    if instrument == 'CORELLI':
        k_min, k_max = 2.5, 10
    elif instrument == 'TOPAZ':
        k_min, k_max = 1.8, 12.5
    elif instrument == 'MANDI':
        k_min, k_max = 1.5, 6.3
    elif instrument == 'SNAP':
        k_min, k_max = 1.8, 12.5

    if not mtd.doesExist('tube_table') and tube_calibration is not None:
        LoadNexus(Filename=tube_calibration, OutputWorkspace='tube_table')

    if not mtd.doesExist('sa') and counts_file is not None:
        if facility == 'SNS':
            LoadNexus(Filename=counts_file, OutputWorkspace='sa')
            if not mtd['sa'].run().hasProperty('NormalizationFactor'):
                NormaliseByCurrent('sa', OutputWorkspace='sa')
        elif not mtd.doesExist('van'):
            LoadMD(Filename=counts_file, OutputWorkspace='van')

    elif not mtd.doesExist('sa') and facility == 'SNS':
        CreateSimulationWorkspace(Instrument=instrument, BinParams='{},0.01,{}'.format(k_min,k_max), OutputWorkspace='tmp', UnitX='Momentum')
        mtd['tmp'] /= mtd['tmp'].getMaxNumberBins()
        Rebin(InputWorkspace='tmp', Params='{},{},{}'.format(k_min,k_max,k_max), PreserveEvents=False, OutputWorkspace='sa')
        mtd['tmp'] /= mtd['tmp'].getNumberHistograms()
        CreateGroupingWorkspace(InputWorkspace='tmp', GroupDetectorsBy='bank', OutputWorkspace='group')
        GroupDetectors(InputWorkspace='tmp', OutputWorkspace='tmp', CopyGroupingFromWorkspace='group')
        mtd['tmp'] *= mtd['tmp'].getNumberHistograms()
        Rebin(InputWorkspace='tmp', Params='{},0.001,{}'.format(k_min,k_max), OutputWorkspace='tmp')
        IntegrateFlux(InputWorkspace='tmp', OutputWorkspace='flux')
        DeleteWorkspace('tmp')
        DeleteWorkspace('group')

    if not mtd.doesExist('flux') and spectrum_file is not None:
        LoadNexus(Filename=spectrum_file, OutputWorkspace='flux')

    if mtd.doesExist('tube_table'):
        if mtd.doesExist('sa'):
            ApplyCalibration(Workspace='sa', CalibrationTable='tube_table')
        if mtd.doesExist('flux'):
            ApplyCalibration(Workspace='flux', CalibrationTable='tube_table')
        DeleteWorkspace('tube_table')

    if detector_calibration is not None:
        ext = os.path.splitext(detector_calibration)[1]
        if mtd.doesExist('sa'):
            if ext == '.xml':
                LoadParameterFile(Workspace='sa', Filename=detector_calibration)
            else:
                LoadIsawDetCal(InputWorkspace='sa', Filename=detector_calibration)
        if mtd.doesExist('flux'):
            if ext == '.xml':
                LoadParameterFile(Workspace='flux', Filename=detector_calibration)
            else:
                LoadIsawDetCal(InputWorkspace='flux', Filename=detector_calibration)

def pre_integration(runs, outname, directory, facility, instrument, ipts, ub_file, reflection_condition, min_d,
                    spectrum_file, counts_file, tube_calibration, detector_calibration,
                    mod_vector_1=[0,0,0], mod_vector_2=[0,0,0], mod_vector_3=[0,0,0],
                    max_order=0, cross_terms=False, exp=None, tmp=None):

    min_d_spacing = min_d
    max_d_spacing= 100

    # peak centroid radius ---------------------------------------------------------
    centroid_radius = 0.125

    load_normalization_calibration(facility, instrument, spectrum_file, counts_file,
                                   tube_calibration, detector_calibration)

    if mtd.doesExist('sa'):
        CreatePeaksWorkspace(InstrumentWorkspace='sa', NumberOfPeaks=0, OutputType='Peak', OutputWorkspace='tmp')
    else:
        CreatePeaksWorkspace(InstrumentWorkspace='van', NumberOfPeaks=0, OutputType='Peak', OutputWorkspace='tmp')
    
    CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace='tmp_lean')

    if facility == 'HFIR':
        LoadEmptyInstrument(InstrumentName='HB3A', OutputWorkspace='rws')

    for i, r in enumerate(runs):
        print('\tProcessing run : {}'.format(r))
        if facility == 'SNS':
            ows = '{}_{}'.format(instrument,r)
        else:
            ows = '{}_{}_{}'.format(instrument,exp,r)

        omd = ows+'_md'
        opk = ows+'_pk'

        if not mtd.doesExist(opk):

            if facility == 'SNS':
                filename = '/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(instrument,ipts,instrument,r)
                LoadEventNexus(Filename=filename, OutputWorkspace=ows)
                
                if mtd.doesExist('sa'):
                    MaskDetectors(Workspace=ows, MaskedWorkspace='sa')
                    
                    if instrument == 'SNAP':
                        if not mtd['sa'].run().hasProperty('det_arc1') or \
                           not mtd['sa'].run().hasProperty('det_arc2') or \
                           not mtd['sa'].run().hasProperty('det_lin1') or \
                           not mtd['sa'].run().hasProperty('det_lin2'):
                            LoadNexusLogs(Workspace='sa', Filename=filename, OverwriteLogs=False)
                            LoadInstrument(Workspace='sa', InstrumentName='SNAP', RewriteSpectraMap=False)
                            
                            load_normalization_calibration(facility, instrument, spectrum_file, counts_file,
                                                           tube_calibration, detector_calibration)
                                   
                    CopyInstrumentParameters(InputWorkspace='sa', OutputWorkspace=ows)

                if instrument == 'CORELLI':
                    gon_axis = 'BL9:Mot:Sample:Axis3.RBV'
                    possible_axes = ['BL9:Mot:Sample:Axis1', 'BL9:Mot:Sample:Axis2', 'BL9:Mot:Sample:Axis3', 
                                     'BL9:Mot:Sample:Axis1.RBV', 'BL9:Mot:Sample:Axis2.RBV', 'BL9:Mot:Sample:Axis3.RBV']
                    for possible_axis in possible_axes:
                        if mtd[ows].run().hasProperty(possible_axis):
                            angle = np.mean(mtd[ows].run().getProperty(possible_axis).value)
                            if not np.isclose(angle,0):
                                gon_axis = possible_axis
                    SetGoniometer(Workspace=ows, Axis0='{},0,1,0,1'.format(gon_axis))
                elif instrument == 'SNAP':
                    SetGoniometer(Workspace=ows, Axis0='omega,0,1,0,1')
                else:
                    SetGoniometer(Workspace=ows, Goniometers='Universal')

                if instrument == 'CORELLI':
                    k_max, two_theta_max = 10, 148.2
                elif instrument == 'TOPAZ':
                    k_max, two_theta_max = 12.5, 160
                elif instrument == 'MANDI':
                    k_max, two_theta_max = 6.3, 160
                elif instrument == 'SNAP':
                    k_max, two_theta_max = 12.5, 138

                lamda_min = 2*np.pi/k_max  

                Qmax = 4*np.pi/lamda_min*np.sin(np.deg2rad(two_theta_max)/2)

                # min_vals, max_vals = ConvertToMDMinMaxGlobal(InputWorkspace=ows,
                #                                              QDimensions='Q3D',
                #                                              dEAnalysisMode='Elastic',
                #                                              Q3DFrames='Q')

                ConvertToMD(InputWorkspace=ows,
                            OutputWorkspace=omd,
                            QDimensions='Q3D',
                            dEAnalysisMode='Elastic',
                            Q3DFrames='Q_sample',
                            LorentzCorrection=False,
                            MinValues='{},{},{}'.format(-Qmax,-Qmax,-Qmax),
                            MaxValues='{},{},{}'.format(Qmax,Qmax,Qmax),
                            Uproj='1,0,0',
                            Vproj='0,1,0',
                            Wproj='0,0,1')

            else:
                filename = '/HFIR/{}/IPTS-{}/shared/autoreduce/{}_exp{:04}_scan{:04}.nxs'.format(instrument,ipts,instrument,exp,r)
                LoadMD(Filename=filename, OutputWorkspace=ows)
                
                if instrument == 'HB3A':
                    component = mtd[ows].getExperimentInfo(0).componentInfo()
                    for bank in [1,2,3]:
                        height_offset = 0.0     # unit meter
                        distance_offset = 0.0   # unit meter

                        index = component.indexOfAny('bank{}'.format(bank))

                        offset = V3D(0, height_offset, 0)
                        pos = component.position(index)
                        offset += pos
                        component.setPosition(index, offset)

                        panel_index = int(component.children(index)[0])
                        panel_pos = component.position(panel_index)
                        panel_rel_pos = component.relativePosition(panel_index)

                        panel_offset = panel_rel_pos*(distance_offset/panel_rel_pos.norm())
                        panel_offset += panel_pos
                        component.setPosition(panel_index, panel_offset)

                if instrument == 'HB3A':
                    SetGoniometer(Workspace=ows,
                                  Axis0='omega,0,1,0,-1',
                                  Axis1='chi,0,0,1,-1',
                                  Axis2='phi,0,1,0,-1',
                                  Average=False)
                elif instrument == 'HB2C':
                    SetGoniometer(Workspace=ows,
                                  Axis0='s1,0,1,0,',
                                  Average=False)
                                  
                lamda = float(mtd[ows].getExperimentInfo(0).run().getProperty('wavelength').value)

                if instrument == 'HB3A':
                    two_theta_max = 155
                elif instrument == 'HB2C':
                    two_theta_max = 156

                Qmax = 4*np.pi/lamda*np.sin(np.deg2rad(two_theta_max)/2)

                ConvertHFIRSCDtoMDE(InputWorkspace=ows,
                                    Wavelength=lamda,
                                    MinValues='{},{},{}'.format(-Qmax,-Qmax,-Qmax),
                                    MaxValues='{},{},{}'.format(Qmax,Qmax,Qmax),
                                    SplitInto=5,
                                    SplitThreshold=1000,
                                    MaxRecursionDepth=13,
                                    OutputWorkspace=omd)

            if type(ub_file) is list:
                LoadIsawUB(InputWorkspace=omd, Filename=ub_file[i])
            elif type(ub_file) is str:
                LoadIsawUB(InputWorkspace=omd, Filename=ub_file)
            else:
                # UB = mtd[ows].getExperimentInfo(0).run().getProperty('ubmatrix').value
                # UB = np.array([float(ub) for ub in UB.split(' ')]).reshape(3,3)
                UB = mtd[ows].getExperimentInfo(0).sample().getOrientedLattice().getUB()
                SetUB(Workspace=omd, UB=UB)

        if not mtd.doesExist(opk):

            if facility == 'SNS':

                if instrument == 'CORELLI':
                    k_min, k_max = 2.5, 10
                elif instrument == 'TOPAZ':
                    k_min, k_max = 1.8, 12.5
                elif instrument == 'MANDI':
                    k_min, k_max = 1.5, 6.3
                elif instrument == 'SNAP':
                    k_min, k_max = 1.8, 12.5

                lamda_min, lamda_max = 2*np.pi/k_max, 2*np.pi/k_min

                PredictPeaks(InputWorkspace=omd,
                             WavelengthMin=lamda_min,
                             WavelengthMax=lamda_max,
                             MinDSpacing=min_d_spacing,
                             MaxDSpacing=max_d_spacing,
                             OutputType='Peak',
                             ReflectionCondition=reflection_condition if reflection_condition is not None else 'Primitive',
                             OutputWorkspace=opk)

            else:

                run = mtd[omd].getExperimentInfo(0).run()

                lamda = float(run.getProperty('wavelength').value)

                lamda_min, lamda_max = 0.95*lamda, 1.05*lamda

                gon = run.getGoniometer().getEulerAngles('YZY')

                use_inner = True if np.mean(np.diff(run.getLogData('phi').value)) > np.mean(np.diff(run.getLogData('omega').value)) else False

                if use_inner:
                    min_angle = -run.getLogData('phi').value.max()
                    max_angle = -run.getLogData('phi').value.min()

                    phi_log = -run.getLogData('phi').value[0]
                    if np.isclose(phi_log+180, gon[2]):
                        min_angle += 180
                        max_angle += 180
                    elif np.isclose(phi_log-180, gon[2]):
                        min_angle -= 180
                        max_angle -= 180
                else:
                    min_angle = -run.getLogData('omega').value.max()
                    max_angle = -run.getLogData('omega').value.min()

                    omega_log = -run.getLogData('omega').value[0]
                    if np.isclose(omega_log+180, gon[0]):
                        min_angle += 180
                        max_angle += 180
                    elif np.isclose(omega_log-180, gon[0]):
                        min_angle -= 180
                        max_angle -= 180

                PredictPeaks(InputWorkspace=omd,
                             WavelengthMin=lamda_min,
                             WavelengthMax=lamda_max,
                             MinDSpacing=min_d_spacing,
                             MaxDSpacing=max_d_spacing,
                             ReflectionCondition=reflection_condition,
                             CalculateGoniometerForCW=True,
                             CalculateWavelength=False,
                             Wavelength=lamda,
                             InnerGoniometer=use_inner,
                             MinAngle=min_angle,
                             MaxAngle=max_angle,
                             FlipX=True if instrument == 'HB3A' else False,
                             OutputType='Peak',
                             OutputWorkspace=opk)

                for pn in range(mtd[opk].getNumberPeaks()):
                    pk = mtd[opk].getPeak(pn)
                    pk.setRunNumber(r)

            if max_order > 0:

                if facility == 'HFIR':

                    PredictPeaks(InputWorkspace=omd,
                                 WavelengthMin=lamda_min,
                                 WavelengthMax=lamda_max,
                                 MinDSpacing=min_d_spacing,
                                 MaxDSpacing=max_d_spacing,
                                 ReflectionCondition=reflection_condition,
                                 CalculateGoniometerForCW=True,
                                 CalculateWavelength=False,
                                 Wavelength=lamda,
                                 InnerGoniometer=use_inner,
                                 MinAngle=min_angle,
                                 MaxAngle=max_angle,
                                 FlipX=True if instrument == 'HB3A' else False,
                                 OutputType='LeanElasticPeak',
                                 OutputWorkspace='main')

                    for pn in range(mtd['main'].getNumberPeaks()-1,0,-1):
                        pk_1, pk_2 = mtd['main'].getPeak(pn), mtd['main'].getPeak(pn-1)
                        if np.allclose([list(pk_1.getHKL())],
                                       [list(pk_2.getHKL())]):
                            mtd['main'].removePeak(pn)

                    PredictSatellitePeaks(Peaks='main',
                                          SatellitePeaks='sat',
                                          ModVector1=mod_vector_1,
                                          ModVector2=mod_vector_2,
                                          ModVector3=mod_vector_3,
                                          MaxOrder=max_order,
                                          CrossTerms=cross_terms,
                                          IncludeIntegerHKL=False,
                                          IncludeAllPeaksInRange=False)

                    DeleteWorkspace('main')

                    HFIRCalculateGoniometer(Workspace='sat',
                                            Wavelength=lamda,
                                            OverrideProperty=True,
                                            InnerGoniometer=use_inner,
                                            FlipX=True if instrument == 'HB3A' else False)

                    ns = mtd[opk].getNumberPeaks()
                    for pn in range(mtd['sat'].getNumberPeaks()):
                        pk = mtd['sat'].getPeak(pn)
                        pk.setRunNumber(r)
                        pk.setPeakNumber(pk.getPeakNumber()+ns+1)

                    ConvertPeaksWorkspace(PeakWorkspace='sat',
                                          InstrumentWorkspace=opk,
                                          OutputWorkspace='sat')

                else:

                    PredictSatellitePeaks(Peaks=opk,
                                          SatellitePeaks='sat',
                                          WavelengthMin=lamda_min,
                                          WavelengthMax=lamda_max,
                                          MinDSpacing=min_d_spacing,
                                          MaxDSpacing=max_d_spacing,
                                          ModVector1=mod_vector_1,
                                          ModVector2=mod_vector_2,
                                          ModVector3=mod_vector_3,
                                          MaxOrder=max_order,
                                          CrossTerms=cross_terms,
                                          IncludeIntegerHKL=False,
                                          IncludeAllPeaksInRange=True)

                for pn in range(mtd['sat'].getNumberPeaks()-1,-1,-1):
                    h, k, l =  mtd['sat'].getPeak(pn).getIntHKL()
                    h, k, l = int(h), int(k), int(l)
                    if reflection_condition == 'Primitive':
                        allowed = True
                    elif reflection_condition == 'C-face centred':
                        allowed = (h + k) % 2 == 0
                    elif reflection_condition == 'A-face centred':
                        allowed = (k + l) % 2 == 0
                    elif reflection_condition == 'B-face centred':
                        allowed = (h + l) % 2 == 0
                    elif reflection_condition == 'Body centred':
                        allowed = (h + k + l) % 2 == 0
                    elif reflection_condition == 'Rhombohedrally centred, obverse':
                        allowed = (-h + k + l) % 3 == 0
                    elif reflection_condition == 'Rhombohedrally centred, reverse':
                        allowed = (h - k + l) % 3 == 0
                    elif reflection_condition == 'Hexagonally centred, reverse':
                        allowed = (h - k) % 3 == 0
                    if not allowed:
                        mtd['sat'].removePeak(pn)

                CombinePeaksWorkspaces(LHSWorkspace=opk,
                                       RHSWorkspace='sat',
                                       CombineMatchingPeaks=True,
                                       Tolerance=1e-3,
                                       OutputWorkspace=opk)

                DeleteWorkspace('sat')

            IntegratePeaksMD(InputWorkspace=omd,
                             PeakRadius=centroid_radius,
                             BackgroundInnerRadius=centroid_radius+0.01,
                             BackgroundOuterRadius=centroid_radius+0.02,
                             PeaksWorkspace=opk,
                             OutputWorkspace=opk)

            FilterPeaks(InputWorkspace=opk,
                        FilterVariable='QMod',
                        FilterValue=0,
                        Operator='>',
                        OutputWorkspace=opk)

            FilterPeaks(InputWorkspace=opk,
                        FilterVariable='Intensity',
                        FilterValue=0,
                        Operator='>',
                        OutputWorkspace=opk)

            FilterPeaks(InputWorkspace=opk,
                        FilterVariable='h^2+k^2+l^2',
                        FilterValue=0,
                        Operator='>',
                        OutputWorkspace=opk)
            
            det_IDs = mtd[opk].column(1)

            for pn in range(mtd[opk].getNumberPeaks()-1,-1,-1):
                if det_IDs[pn] == -1:
                    mtd[opk].removePeak(pn)

            SortPeaksWorkspace(InputWorkspace=opk, 
                               ColumnNameToSortBy='DSpacing',
                               SortAscending=False,
                               OutputWorkspace=opk)

            for pn in range(mtd[opk].getNumberPeaks()-1,0,-1):
                pk_1, pk_2 = mtd[opk].getPeak(pn), mtd[opk].getPeak(pn-1)
                if np.allclose([pk_1.getRunNumber(), *list(pk_1.getHKL())],
                               [pk_2.getRunNumber(), *list(pk_2.getHKL())]): 
                    mtd[opk].removePeak(pn)

        if facility == 'HFIR':

            det_IDs = mtd[opk].column(1)

            run = mtd[omd].getExperimentInfo(0).run()

            scans = run.getNumGoniometers()

            rotation, _, _ = np.array([run.getGoniometer(i).getEulerAngles('YZY') for i in range(scans)]).T

            rotation_start, rotation_stop = rotation[0], rotation[-1]

            rotation_diff = np.diff(rotation).mean()

            i_diff = abs(int(round(10/rotation_diff)))

            two_theta = np.mean(run.getProperty('2theta').value)
            det_trans = np.mean(run.getProperty('det_trans').value)

            AddSampleLog(Workspace='rws',
                         LogName='2theta', 
                         LogText=str(two_theta),
                         LogType='Number Series',
                         LogUnit='degree',
                         NumberType='Double')

            AddSampleLog(Workspace='rws',
                         LogName='det_trans', 
                         LogText=str(det_trans),
                         LogType='Number Series',
                         LogUnit='millimeter',
                         NumberType='Double')

            LoadInstrument(Workspace='rws', InstrumentName='HB3A', RewriteSpectraMap=False)

            component = mtd['rws'].componentInfo()
            for bank in [1,2,3]:
                height_offset = 0    # unit meter
                distance_offset = 0  # unit meter

                index = component.indexOfAny('bank{}'.format(bank))

                offset = V3D(0, height_offset, 0)
                pos = component.position(index)
                offset += pos
                component.setPosition(index, offset)

                panel_index = int(component.children(index)[0])
                panel_pos = component.position(panel_index)
                panel_rel_pos = component.relativePosition(panel_index)

                panel_offset = panel_rel_pos*(distance_offset/panel_rel_pos.norm())
                panel_offset += panel_pos
                component.setPosition(panel_index, panel_offset)

            PreprocessDetectorsToMD(InputWorkspace='rws', OutputWorkspace='preproc')

            twotheta = np.array(mtd['preproc'].column(2)).reshape(3,512,512)
            azimuthal = np.array(mtd['preproc'].column(3)).reshape(3,512,512)

            logs = ['omega', 'chi', 'phi', 'time', 'monitor', 'monitor2']

            log_dict = {}

            for log in logs:
                prop = run.getProperty(log)
                log_dict[log] = prop.times.copy(), prop.value.copy()

            for log in run.keys():
                if log not in logs:
                    prop = run.getProperty(log)
                    if type(prop.value) is np.ndarray:
                        run.addProperty(log, 0, True)

            for pn in range(mtd[opk].getNumberPeaks()):
                pk = mtd[opk].getPeak(pn)

                det_ID = det_IDs[pn]
                no = pk.getPeakNumber()

                comp_path = mtd['rws'].getInstrument().getDetector(det_ID).getFullName()
                bank = int(re.search('bank[0-9]+', comp_path).group().lstrip('bank'))

                start, stop = 512*(bank-1), 512*bank

                pk.setBinCount(bank)

                R = pk.getGoniometerMatrix()

                g = Goniometer()
                g.setR(R)

                rot, _, _ = g.getEulerAngles('YZY')

                i_ind = int(round((rot-rotation_start)/(rotation_stop-rotation_start)*(scans-1))) 

                i_start = i_ind-i_diff if i_ind-i_diff > 0 else 0
                i_stop = i_ind+i_diff if i_ind+i_diff < scans else scans

                SliceMDHisto(InputWorkspace=ows,
                             Start='{},0,{}'.format(start,i_start),
                             End='{},512,{}'.format(stop,i_stop),
                             OutputWorkspace='crop_data')

                crop_run = mtd['crop_data'].getExperimentInfo(0).run()

                for log in logs:
                    prop_times, prop_values = log_dict[log]
                    times, values = prop_times[i_start:i_stop], prop_values[i_start:i_stop]
                    new_log = FloatTimeSeriesProperty(log)
                    for t, v in zip(times, values):
                        new_log.addValue(t, v)
                    crop_run[log] = new_log

                crop_run.addProperty('twotheta', twotheta[bank-1].T.flatten().tolist(), True)
                crop_run.addProperty('azimuthal', azimuthal[bank-1].T.flatten().tolist(), True)

                tmp_directory = '{}/{}/'.format(directory, tmp)
                tmp_outname = '{}_{}_{}_{}.nxs'.format(instrument,exp,r,no)

                SaveMD(InputWorkspace='crop_data', Filename=os.path.join(tmp_directory, tmp_outname))
                DeleteWorkspace('crop_data')

        ConvertPeaksWorkspace(PeakWorkspace=opk, OutputWorkspace=opk+'_lean')

        CombinePeaksWorkspaces(LHSWorkspace='tmp_lean', RHSWorkspace=opk+'_lean', OutputWorkspace='tmp_lean')
        SetUB(Workspace='tmp_lean', UB=mtd[opk+'_lean'].sample().getOrientedLattice().getUB())

        CombinePeaksWorkspaces(LHSWorkspace='tmp', RHSWorkspace=opk, OutputWorkspace='tmp')
        SetUB(Workspace='tmp', UB=mtd[opk].sample().getOrientedLattice().getUB())

        if max_order > 0:

            ol = mtd['tmp'].sample().getOrientedLattice()
            ol.setMaxOrder(max_order)

            ol.setModVec1(V3D(*mod_vector_1))
            ol.setModVec2(V3D(*mod_vector_2))
            ol.setModVec3(V3D(*mod_vector_3))

            UB = ol.getUB()

            mod_HKL = np.column_stack((mod_vector_1,mod_vector_2,mod_vector_3))
            mod_UB = np.dot(UB, mod_HKL)

            ol.setModUB(mod_UB)

        if mtd.doesExist(ows):
            DeleteWorkspace(ows)
        if mtd.doesExist(opk):
            DeleteWorkspace(opk)
        if mtd.doesExist(omd):
            DeleteWorkspace(omd)
        if mtd.doesExist(opk+'_lean'):
            DeleteWorkspace(opk+'_lean')

    SaveNexus(InputWorkspace='tmp_lean', Filename=os.path.join(directory, outname+'_pk_lean.nxs'))
    SaveNexus(InputWorkspace='tmp', Filename=os.path.join(directory, outname+'_pk.nxs'))
    SaveIsawUB(InputWorkspace='tmp', Filename=os.path.join(directory, outname+'.mat'))

def partial_load(facility, instrument, runs, banks, indices, phi, chi, omega, norm_scale,
                 directory, ipts, outname, exp=None, tmp=None):

    for r, b, i, p, c, o in zip(runs, banks, indices, phi, chi, omega):

        if facility == 'SNS':
            ows = '{}_{}_{}'.format(instrument,r,b)
        else:
            ows = '{}_{}_{}_{}'.format(instrument,exp,r,i)

        omd = ows+'_md'

        if facility == 'SNS':

            if not mtd.doesExist(omd):

                filename = '/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(instrument,ipts,instrument,r)
                
                if instrument == 'CORELLI':
                    if b == 1 or b == 91:
                        banks = 'bank{}'.format(b)
                    else:
                        banks = ','.join(['bank{}'.format(bank) for bank in [b-1,b,b+1]])
                else:
                    banks = 'bank{}'.format(b)
                    
                LoadEventNexus(Filename=filename, 
                               BankName=banks, 
                               SingleBankPixelsOnly=True,
                               LoadLogs=False,
                               LoadNexusInstrumentXML=False,
                               OutputWorkspace=ows)

                pc = norm_scale[r]

                AddSampleLog(Workspace=ows,
                             LogName='gd_prtn_chrg', 
                             LogText=str(pc),
                             LogType='Number',
                             LogUnit='uA.hour',
                             NumberType='Double')

                AddSampleLog(Workspace=ows,
                             LogName='phi', 
                             LogText=str(p),
                             LogType='Number Series',
                             LogUnit='degree',
                             NumberType='Double')

                AddSampleLog(Workspace=ows,
                             LogName='chi', 
                             LogText=str(c),
                             LogType='Number Series',
                             LogUnit='degree',
                             NumberType='Double')

                AddSampleLog(Workspace=ows,
                             LogName='omega', 
                             LogText=str(o),
                             LogType='Number Series',
                             LogUnit='degree',
                             NumberType='Double')

                if mtd.doesExist('sa'):

                    CopyInstrumentParameters(InputWorkspace='sa', OutputWorkspace=ows)

                SetGoniometer(Workspace=ows, Goniometers='Universal')

                if mtd.doesExist('flux'):

                    ConvertUnits(InputWorkspace=ows, OutputWorkspace=ows, EMode='Elastic', Target='Momentum')

                    CropWorkspaceForMDNorm(InputWorkspace=ows,
                                           XMin=mtd['flux'].dataX(0).min(),
                                           XMax=mtd['flux'].dataX(0).max(),
                                           OutputWorkspace=ows)

                ConvertToMD(InputWorkspace=ows,
                            OutputWorkspace=omd,
                            QDimensions='Q3D',
                            dEAnalysisMode='Elastic',
                            Q3DFrames='Q_sample',
                            LorentzCorrection=False,
                            PreprocDetectorsWS='-',
                            MinValues='-20,-20,-20',
                            MaxValues='20,20,20',
                            Uproj='1,0,0',
                            Vproj='0,1,0',
                            Wproj='0,0,1')

                DeleteWorkspace(ows)

        else:

            if not mtd.doesExist(ows):

                filename = '{}/{}/{}.nxs'.format(directory, tmp, ows)
                LoadMD(Filename=filename, OutputWorkspace=ows)

                SetGoniometer(Workspace=ows,
                              Axis0='omega,0,1,0,-1',
                              Axis1='chi,0,0,1,-1',
                              Axis2='phi,0,1,0,-1',
                              Average=False)

                norm_scale[(r,i)] = np.sum(mtd[ows].getExperimentInfo(0).run().getProperty('monitor').value)

def partial_cleanup(runs, banks, indices, facility, instrument, runs_banks, bank_keys, bank_group, key, exp=None):

    for r, b, i in zip(runs, banks, indices):

        if facility == 'SNS':
            ows = '{}_{}_{}'.format(instrument,r,b)
        else:
            ows = '{}_{}_{}_{}'.format(instrument,exp,r,i)

        omd = ows+'_md'

        peak_keys = runs_banks[(r,b)]
        peak_keys.remove(key)
        runs_banks[(r,b)] = peak_keys

        key_list = bank_keys[b]
        key_list.remove(key)
        bank_keys[b] = key_list

        if facility == 'SNS':

            if len(peak_keys) == 0 or psutil.virtual_memory().percent > 85:
                if mtd.doesExist(omd):
                    DeleteWorkspace(omd)

            # if len(key_list) == 0:
            #     MaskBTP(Workspace='sa', Bank=b)
            #     if bank_group.get(b) is not None:
            #         MaskSpectra(InputWorkspace='flux', 
            #                     InputWorkspaceIndexType='SpectrumNumber',
            #                     InputWorkspaceIndexSet=bank_group[b],
            #                     OutputWorkspace='flux')

        else:
            
            if mtd.doesExist(ows):
                DeleteWorkspace(ows)

    return runs_banks, bank_keys

def set_instrument(instrument):

    tof_instruments = ['CORELLI', 'MANDI', 'TOPAZ', 'SNAP']

    instrument = instrument.upper()

    if instrument == 'BL9':
        instrument = 'CORELLI'
    if instrument == 'BL11B':
        instrument = 'MANDI'
    if instrument == 'BL12':
        instrument = 'TOPAZ'
    if instrument == 'BL3':
        instrument = 'SNAP'

    if instrument == 'DEMAND':
        instrument = 'HB3A'
    if instrument == 'WAND2':
        instrument = 'HB2C'

    facility = 'SNS' if instrument in tof_instruments else 'HFIR'

    return facility, instrument

def projection_axes(n):

    n_ind = np.argmin(np.abs(n))

    u = np.zeros(3)
    u[n_ind] = 1

    u = np.cross(n, u)
    u /= np.linalg.norm(u)

    v = np.cross(n, u)
    v *= np.sign(np.dot(np.cross(u, n), v))

    return u, v

def integration_loop(keys, outname, ref_dict, peak_tree, int_list, filename,
                     spectrum_file, counts_file, tube_calibration, detector_calibration,
                     directory, facility, instrument, ipts, runs,
                     split_angle, a, b, c, alpha, beta, gamma, reflection_condition,
                     mod_vector_1, mod_vector_2, mod_vector_3, max_order, cross_terms,
                     chemical_formula, z_parameter, sample_mass, experiment, tmp):

    scale_constant = 1e+4

    load_normalization_calibration(facility, instrument, spectrum_file, counts_file,
                                   tube_calibration, detector_calibration)

    if facility == 'HFIR':
        ows = '{}_{}'.format(instrument,experiment)+'_{}'
    else:
        ows = '{}'.format(instrument)+'_{}'

    opk = ows+'_pk'
    omd = ows+'_md'

    LoadNexus(Filename=filename+'_pk.nxs', OutputWorkspace='tmp')
    LoadIsawUB(InputWorkspace='tmp', Filename=filename+'.mat')

    for r in runs:
        FilterPeaks(InputWorkspace='tmp',
                    FilterVariable='RunNumber',
                    FilterValue=r,
                    Operator='=',
                    OutputWorkspace=opk.format(r))

    peak_dictionary = PeakDictionary(a, b, c, alpha, beta, gamma)
    peak_dictionary.set_satellite_info(mod_vector_1, mod_vector_2, mod_vector_3, max_order)
    peak_dictionary.set_material_info(chemical_formula, z_parameter, sample_mass)
    peak_dictionary.set_scale_constant(scale_constant)

    LoadIsawUB(InputWorkspace='cws', Filename=filename+'.mat')

    for r in runs:

        if max_order > 0:

            ol = mtd[opk.format(r)].sample().getOrientedLattice()
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

            for pn in range(mtd[opk.format(r)].getNumberPeaks()):
                pk = mtd[opk.format(r)].getPeak(pn)
                hkl = pk.getHKL()
                for m, n, p in iter_mnp:
                    d_hkl = m*np.array(mod_vector_1)\
                          + n*np.array(mod_vector_2)\
                          + p*np.array(mod_vector_3)
                    HKL = np.round(hkl-d_hkl,4)
                    mnp = [m,n,p]
                    H, K, L = HKL
                    h, k, l = int(H), int(K), int(L)
                    if reflection_condition == 'Primitive':
                        allowed = True
                    elif reflection_condition == 'C-face centred':
                        allowed = (h + k) % 2 == 0
                    elif reflection_condition == 'A-face centred':
                        allowed = (k + l) % 2 == 0
                    elif reflection_condition == 'B-face centred':
                        allowed = (h + l) % 2 == 0
                    elif reflection_condition == 'Body centred':
                        allowed = (h + k + l) % 2 == 0
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

        peak_dictionary.add_peaks(opk.format(r))
        DeleteWorkspace(opk.format(r))

    peak_dictionary.split_peaks(split_angle)
    peak_dict = peak_dictionary.to_be_integrated()

    peak_envelope = PeakEnvelope(os.path.join(directory, '{}.pdf'.format(outname)))
    peak_envelope.show_plots(False)

    excl_envelope = PeakEnvelope(os.path.join(directory, 'rej_{}.pdf'.format(outname)))
    excl_envelope.show_plots(False)

    DeleteWorkspace('tmp')

    norm_scale = {}

    if facility == 'SNS':
        LoadEmptyInstrument(InstrumentName=instrument, OutputWorkspace=instrument+'_empty')
        for r in runs:
            logfile = '/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(instrument,ipts,instrument,r)
            LoadNexusLogs(Workspace=instrument+'_empty', Filename=logfile, OverwriteLogs=True, AllowList='gd_prtn_chrg')
            norm_scale[r] = mtd[instrument+'_empty'].getRun().getPropertyAsSingleValueWithTimeAveragedMean('gd_prtn_chrg')
        DeleteWorkspace(instrument+'_empty')

    bank_set = set()

    runs_banks = {}
    bank_keys = {}

    for key in keys:

        key = tuple(key)

        peaks = peak_dict[key]

        for peak in peaks:

            runs = peak.get_run_numbers()
            banks = peak.get_bank_numbers()

            for r, b in zip(runs, banks):

                bank_set.add(b)

                if runs_banks.get((r,b)) is None:
                    runs_banks[(r,b)] = [key]
                else:
                    peak_keys = runs_banks[(r,b)]
                    peak_keys.append(key)
                    runs_banks[(r,b)] = peak_keys

                if bank_keys.get(b) is None:
                    bank_keys[b] = [key]
                else:
                    key_list = bank_keys[b]
                    key_list.append(key)
                    bank_keys[b] = key_list

    banks = list(bank_set)

    if mtd.doesExist('van'):
        for b in banks:
            start, stop = 512*(b-1), 512*b
            SliceMDHisto(InputWorkspace='van',
                         Start='{},0,0'.format(start),
                         End='{},512,1'.format(stop),
                         OutputWorkspace='van_{}'.format(b))
        van_ws = ['van_{}'.format(b) for b in banks]
        if len(van_ws) > 0:
            GroupWorkspaces(InputWorkspaces=','.join(van_ws), OutputWorkspace='van')

    bank_group = {}

    if mtd.doesExist('flux'):

        if instrument == 'SNAP':
            logfile = '/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(instrument,ipts,instrument,r)
            LoadNexusLogs(Workspace='flux', Filename=logfile, OverwriteLogs=False)
            LoadNexusLogs(Workspace='sa', Filename=logfile, OverwriteLogs=False)
            LoadInstrument(Workspace='flux', InstrumentName='SNAP', RewriteSpectraMap=False)
            LoadInstrument(Workspace='sa', InstrumentName='SNAP', RewriteSpectraMap=False)

            load_normalization_calibration(facility, instrument, spectrum_file, counts_file,
                                           tube_calibration, detector_calibration)

        for hn, sn in zip(range(mtd['flux'].getNumberHistograms()), mtd['flux'].getSpectrumNumbers()):

            idx = mtd['flux'].getSpectrum(hn).getDetectorIDs()[0]
            comp_path = mtd['flux'].getInstrument().getDetector(idx).getFullName()
            b = int(re.search('bank[0-9]+', comp_path).group().lstrip('bank'))

            bank_group[b] = sn

        # all_banks = list(bank_group.keys())

        # for b in all_banks:
        #     if b not in banks:
        #         MaskBTP(Workspace='sa', Bank=b)
        #         MaskSpectra(InputWorkspace='flux', 
        #                     InputWorkspaceIndexType='SpectrumNumber',
        #                     InputWorkspaceIndexSet=bank_group[b],
        #                     OutputWorkspace='flux')

    peak_summary = open(os.path.join(directory, '{}_summary.txt'.format(outname)), 'w')
    excl_summary = open(os.path.join(directory, 'rej_{}_summary.txt'.format(outname)), 'w')

    peak_stats = open(os.path.join(directory, '{}_stats.txt'.format(outname)), 'w')
    excl_stats = open(os.path.join(directory, 'rej_{}_stats.txt'.format(outname)), 'w')

    fmt_summary = 3*'{:8.2f}'+'{:8.4f}'+6*'{:8.2f}'+'{:4.0f}'+6*'{:8.2f}'+'\n'
    fmt_stats = 3*'{:8.2f}'+'{:8.4f}'+4*'{:8.2f}'+7*'{:8.2f}'+'\n'

    for i, key in enumerate(keys[:]):

        key = tuple(key)

        print('\tIntegrating peak : {}'.format(key))

        peaks = peak_dict[key]

        peaks_list = peak_dictionary.peak_dict.get(key)

        fixed = False
        if ref_dict is not None:
            ref_peaks = ref_dict.get(key)
            if ref_peaks is not None:
                if len(ref_peaks) == len(peaks):
                    fixed = True

        h, k, l, m, n, p = key

        H, K, L = peak_dictionary.get_hkl(h, k, l, m, n, p)
        d = peak_dictionary.get_d(h, k, l, m, n, p)

        summary_list = [H, K, L, d]
        stats_list = [H, K, L, d]

        min_sig_noise_ratio = 3 if facility == 'SNS' else 1

        for j, peak in enumerate(peaks):

            runs = peak.get_run_numbers().tolist()
            banks = peak.get_bank_numbers().tolist()
            indices = peak.get_peak_indices().tolist()

            Q0 = peak.get_Q()

            phi = peak.get_phi_angles()
            chi = peak.get_chi_angles()
            omega = peak.get_omega_angles()

            R = peak.get_goniometers()[0]

            wls = peak.get_wavelengths()
            tts = peak.get_scattering_angles()
            azs = peak.get_azimuthal_angles()

            lamda = [np.min(wls), np.max(wls)]
            two_theta = np.rad2deg([np.min(tts), np.max(tts)]).tolist()
            az_phi = np.rad2deg([np.min(azs), np.max(azs)]).tolist()

            n_runs = len(wls)

            peak_envelope.clear_plots(key, d, lamda, two_theta, n_runs)
            excl_envelope.clear_plots(key, d, lamda, two_theta, n_runs)

            gon_omega = [np.min(omega), np.max(omega)]
            gon_chi = [np.min(chi), np.max(chi)]
            gon_phi = [np.min(phi), np.max(phi)]

            summary_list.extend([*lamda, *two_theta, *az_phi, n_runs, *gon_omega, *gon_chi, *gon_phi])

            peak_fit, peak_bkg_ratio, sig_noise_ratio, peak_total_data_ratio = 0, 0, 0, 0
            peak_fit2d, peak_score2d, sig_noise_ratio2d = 0, 0, 0

            max_width = 0.3 if facility == 'SNS' else 0.3
            max_offset = 0.3 if facility == 'SNS' else 0.3
            max_ratio = 5 if facility == 'SNS' else 7.5
                
            fit = True
            remove = False

            if fixed:

                ref_peak = ref_peaks[j]

                if not ref_peak.is_peak_integrated():

                    Q0 = ref_peak.get_Q()

                    if peak_tree is None:

                        W = ref_peak.get_W()
                        D = ref_peak.get_D()

                        fit = False

                    else:

                        dist, index = peak_tree.query(Q0, k=1)

                        int_key = int_list[index]
                        int_peaks = ref_dict[int_key]

                        rp, r2 = [], []
                        for int_peak in int_peaks:

                            D = int_peak.get_D()
                            radii = 1/np.sqrt(np.diag(D)) 

                            rp.append(radii[2])
                            r2.append(np.prod(radii[:2]))

                        radius = 1.5*np.mean(rp)
                        r = 2*np.sqrt(np.mean(r2))

                        n = Q0/np.linalg.norm(Q0)

#                         if facility == 'SNS':
#                             n = Q0/np.linalg.norm(Q0)
#                         else:
#                             Ql = np.dot(R,Q0)
#                             t, p = np.arccos(Ql[2]/np.linalg.norm(Ql)), np.arctan2(Ql[1],Ql[0])
# 
#                             w = np.arccos(R[0,0])
#                             n = np.array([-np.cos(p)*np.sin(t)*np.sin(w)+(1-np.cos(t))*np.cos(w),0,np.cos(p)*np.sin(t)*np.cos(w)+(1-np.cos(t))*np.sin(w)])
#                             n /= np.linalg.norm(n)

                        u, v = projection_axes(n)

                        W = np.zeros((3,3))
                        W[:,0] = u
                        W[:,1] = v
                        W[:,2] = n

                        D = np.diag([1/r**2,1/r**2,1/radius**2])
                        
                        radius = 2*np.cbrt(r**2*radius)

                    ol = mtd['cws'].sample().getOrientedLattice()

                    Qave = 2*np.pi*np.mean([ol.astar(), ol.bstar(), ol.cstar()])

                    binsize = 0.0025*Qave

                else:

                    remove = True

            else:

                ol = mtd['cws'].sample().getOrientedLattice()

                Qave = 2*np.pi*np.mean([ol.astar(), ol.bstar(), ol.cstar()])

                radius = 0.25*Qave
                binsize = 0.0025*Qave

                W = np.eye(3)
                D = np.diag([1/radius**2,1/radius**2,1/radius**2])

#             if fixed and not remove:
# 
#                 partial_load(facility, instrument, runs, banks, indices, 
#                              phi, chi, omega, norm_scale,
#                              directory, ipts, outname, experiment, tmp)
# 
#                 signal, Q_bin, Q_rot, ellipsoid_radii, scales, bin_norm = norm_integrator(facility, instrument, runs, banks, indices,
#                                                                                           Q0, D, W, exp=experiment)
# 
#                 u_extents, v_extents, Q_extents = Q_bin
# 
#                 peak_dictionary.integrated_result(key, Q0, D, W, peak_fit, peak_bkg_ratio, peak_score2d, bin_norm, j)
# 
#                 peak_envelope.plot_integration(signal, u_extents, v_extents, Q_extents, Q_rot, ellipsoid_radii, scales)
#                 peak_envelope.write_figure()

            if not remove:

                partial_load(facility, instrument, runs, banks, indices,
                             phi, chi, omega, norm_scale,
                             directory, ipts, outname, experiment, tmp)

                Q, Qx, Qy, Qz, data, norm = box_integrator(facility, instrument, runs, banks, indices, Q0, key,
                                                           binsize=binsize, D=D, W=W, exp=experiment)

                weights = data/norm

                rot = True if facility == 'HFIR' else False

                ellip = Ellipsoid(Qx, Qy, Qz, data, norm, Q0, size=0.8*radius, rotation=rot)

                prof = Profile()
                stats, params = prof.fit(ellip, 0.99)

                peak_fit_1, peak_bkg_ratio_1, sig_noise_ratio_1 = stats
                a, mu, sigma = params

                peak_total_data_ratio = prof.y.max()/prof.y_sub.max()

                if np.any(prof.y_sub > 0) and not np.isnan([a,mu,sigma]).any():

                    ellip.mu = mu
                    ellip.sigma = sigma

                    peak_envelope.plot_Q(prof.x, prof.y_sub, prof.y, prof.e, prof.y_fit, prof.y_bkg)
                    excl_envelope.plot_Q(prof.x, prof.y_sub, prof.y, prof.e, prof.y_fit, prof.y_bkg)
                    
                    print('\tPeak-fit Q: {}'.format(peak_fit_1))
                    print('\tPeak background ratio Q: {}'.format(peak_bkg_ratio_1))
                    print('\tSignal-noise ratio Q: {}'.format(sig_noise_ratio_1))
                    print('\tPeak-total to subtracted-data ratio Q: {}'.format(peak_total_data_ratio))

                    print('\tPeak-sigma Q: {}'.format(sigma))

                    stats_list.extend([peak_fit_1, peak_bkg_ratio_1, sig_noise_ratio_1, peak_total_data_ratio])

                else:

                    stats_list.extend([np.nan, np.nan, np.nan, np.nan])

                proj = Projection()
                stats, params = proj.fit(ellip, 0.99)

                peak_fit2d, peak_bkg_ratio2d, sig_noise_ratio2d = stats
                a, mu_x, mu_y, sigma_x, sigma_y, rho = params

                peak_total_data_ratio2d = proj.z.max()/proj.z_sub.max()

                if np.any(proj.z_sub > 0) and not np.isnan([a,mu_x,mu_y,sigma_x,sigma_y,rho]).any():

                    ellip.mu_x, ellip.mu_y = mu_x, mu_y
                    ellip.sigma_x, ellip.sigma_y, ellip.rho = sigma_x, sigma_y, rho

                    x_extents = [proj.x.min(), proj.x.max()]
                    y_extents = [proj.y.min(), proj.y.max()]

                    mu = [mu_x, mu_y]
                    sigma = [sigma_x, sigma_y]

                    peak_envelope.plot_projection(proj.z_sub, proj.z, x_extents, y_extents, mu, sigma, rho, peak_fit2d)
                    excl_envelope.plot_projection(proj.z_sub, proj.z, x_extents, y_extents, mu, sigma, rho, peak_fit2d)

                    print('\tPeak-score 2d: {}'.format(peak_score2d))
                    print('\tSignal-noise ratio 2d: {}'.format(sig_noise_ratio2d))

                    if (np.isinf(peak_score2d) or np.isnan(peak_score2d)):

                        remove = True

                    # summary_list.extend([*center2d, *np.diag(covariance2d), covariance2d[0,1], *amplitude2d, *background2d])
                    stats_list.extend([peak_fit2d, peak_score2d, sig_noise_ratio2d])

                else:

                    remove = True

                    stats_list.extend([np.nan, np.nan, np.nan])

                prof = Profile()
                stats, params = prof.fit(ellip, 0.99)

                peak_fit_2, peak_bkg_ratio_2, sig_noise_ratio_2 = stats
                a, mu, sigma = params

                peak_total_data_ratio = prof.y.max()/prof.y_sub.max()

                if np.any(prof.y_sub > 0) and not np.isnan([a,mu,sigma]).any():

                    ellip.mu = mu
                    ellip.sigma = sigma

                    if peak_fit_1 < 1 and peak_fit_2 < 1:
                        peak_fit = np.max([peak_fit_1,peak_fit_2])
                    elif peak_fit_1 > 1 and peak_fit_2 > 1:
                        peak_fit = np.min([peak_fit_1,peak_fit_2])
                    else:
                        peak_fit = peak_fit_2
                        
                    peak_bkg_ratio = np.max([peak_bkg_ratio_1, peak_bkg_ratio_2])
                    sig_noise_ratio = np.max([sig_noise_ratio_1, sig_noise_ratio_2])

                    peak_envelope.plot_extracted_Q(prof.x, prof.y_sub, prof.y, prof.e, prof.y_fit, prof.y_bkg, peak_fit)
                    excl_envelope.plot_extracted_Q(prof.x, prof.y_sub, prof.y, prof.e, prof.y_fit, prof.y_bkg, peak_fit)

                    print('\tPeak-fit Q second pass: {}'.format(peak_fit_2))
                    print('\tPeak background ratio Q second pass: {}'.format(peak_bkg_ratio_2))
                    print('\tSignal-noise ratio Q second pass: {}'.format(sig_noise_ratio_2))
                    print('\tPeak-total to subtracted-data ratio Q: {}'.format(peak_total_data_ratio))

                    # summary_list.extend([center, variance, amplitude, background, peak_fit])
                    stats_list.extend([peak_fit_2, peak_bkg_ratio_2, sig_noise_ratio_2, peak_total_data_ratio])

                else:

                    remove = True

                    stats_list.extend([np.nan, np.nan, np.nan, np.nan])

                Q0, W, D = ellip.ellipsoid()

                A = ellip.A(W, D)

                radii = 1/np.sqrt(np.diagonal(D)) 

                print('\tPeak-radii: {}'.format(radii))

                if peak_fit < 0.01 or peak_fit2d < 0.01 or peak_fit > 1000 or peak_fit2d > 1000 or sig_noise_ratio < 3 or sig_noise_ratio2d < 3:

                    remove = True

                if np.isclose(np.abs(np.linalg.det(W)),1) and not (radii > 1.5*radius).any() and not np.isclose(radii, 0).any() and (radii > 0).all() and not remove:

                    signal, Q_bin, Q_rot, ellipsoid_radii, scales, bin_norm = norm_integrator(facility, instrument, runs, banks, indices,
                                                                                              Q0, D, W, exp=experiment)

                    u_extents, v_extents, Q_extents = Q_bin

                    peak_dictionary.integrated_result(key, Q0, D, W, peak_fit, peak_bkg_ratio, peak_score2d, bin_norm, j)

                    peak_envelope.plot_integration(signal, u_extents, v_extents, Q_extents, Q_rot, ellipsoid_radii, scales)
                    peak_envelope.write_figure()

                    peak_summary.write(fmt_summary.format(*summary_list))
                    peak_stats.write(fmt_stats.format(*stats_list))

                else:

                    remove = True

                if remove:

                    excl_envelope.write_figure()

                    excl_summary.write(fmt_summary.format(*summary_list))
                    excl_stats.write(fmt_stats.format(*stats_list))

            runs_banks, bank_keys = partial_cleanup(runs, banks, indices, facility, instrument, runs_banks, bank_keys, bank_group, key, exp=experiment)

        if i % 15 == 0:

            peak_dictionary.save_hkl(os.path.join(directory, '{}.hkl'.format(outname)), min_signal_noise_ratio=min_sig_noise_ratio, cross_terms=cross_terms)
            peak_dictionary.save(os.path.join(directory, '{}.pkl'.format(outname)))

    peak_dictionary.save_hkl(os.path.join(directory, '{}.hkl'.format(outname)))
    peak_dictionary.save(os.path.join(directory, '{}.pkl'.format(outname)))

    peak_envelope.create_pdf()
    excl_envelope.create_pdf()

    peak_summary.close()
    excl_summary.close()

    peak_stats.close()
    excl_stats.close()

    if mtd.doesExist('sa'):
        DeleteWorkspace('sa')
    if mtd.doesExist('flux'):
        DeleteWorkspace('flux')
    if mtd.doesExist('van'):
        DeleteWorkspace('van')