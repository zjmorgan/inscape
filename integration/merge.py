from mantid.simpleapi import *

from mantid.kernel import V3D, FloatTimeSeriesProperty
from mantid.geometry import Goniometer

import os
import re
import psutil
import itertools

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstwobign

import peak
from peak import PeakEnvelope, PeakDictionary

from PIL import Image

#from PyPDF2 import PdfFileMerger
import img2pdf

import fitting
from fitting import Ellipsoid, Profile, Projection, LineCut, GaussianFit3D, SatelliteGaussianFit3D

def box_integrator(facility, instrument, runs, banks, indices, split_angle, Q0, delta_Q0, key, binsize=0.01, radius=0.15, exp=None, close=False):

    for j, (r, b, i) in enumerate(zip(runs, banks, indices)):

        if facility == 'SNS':
            if np.isclose(split_angle, 0):
                ows = '{}_{}'.format(instrument,r)
            else:
                ows = '{}_{}_{}'.format(instrument,r,b)
        elif instrument == 'HB2C':
            ows = '{}_{}_{}'.format(instrument,r,i)
        else:
            ows = '{}_{}_{}_{}'.format(instrument,exp,r,i)

        omd = ows+'_md'

        if j == 0:

            if close:
                n = delta_Q0/np.linalg.norm(delta_Q0)
            else:
                n = Q0/np.linalg.norm(Q0)

            n_ind = np.argmin(np.abs(n))

            u = np.zeros(3)
            u[n_ind] = 1

            u = np.cross(n, u)
            u /= np.linalg.norm(u)

            v = np.cross(n, u)
            v *= np.sign(np.dot(np.cross(u, n), v))

            if np.abs(u[1]) > np.abs(v[1]):
                u, v = v, -u

            W = np.column_stack((u,v,n))

            Q_rot = np.dot(W.T, Q0)

            if close:
                dQ = np.array([radius,radius,radius])
            else:
                dQ = np.array([radius,radius,radius])

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
                   OutputWorkspace='normDataMD',
                   OutputDataWorkspace='tmpDataMD_{}'.format(j),
                   OutputNormalizationWorkspace='tmpNormMD_{}'.format(j))

            DeleteWorkspace('normDataMD')

        else:

            lamda = 1.486 if instrument == 'HB2C' else float(mtd[ows].getExperimentInfo(0).run().getProperty('wavelength').value)

            ReplicateMD(ShapeWorkspace=ows, DataWorkspace='van_'+ows, OutputWorkspace=ows+'_norm')

            ConvertWANDSCDtoQ(InputWorkspace=ows,
                              NormalisationWorkspace=None,
                              UBWorkspace=ows,
                              OutputWorkspace='tmp',
                              Wavelength=lamda,
                              NormaliseBy='Time',
                              Frame='HKL', # not actually HKL,
                              KeepTemporaryWorkspaces=True,
                              Uproj='{},{},{}'.format(*W[:,0]),
                              Vproj='{},{},{}'.format(*W[:,1]),
                              Wproj='{},{},{}'.format(*W[:,2]),
                              BinningDim0='{},{},{}'.format(Q0_bin[0],Q0_bin[2],bins[0]),
                              BinningDim1='{},{},{}'.format(Q1_bin[0],Q1_bin[2],bins[1]),
                              BinningDim2='{},{},{}'.format(Q2_bin[0],Q2_bin[2],bins[2]))

            DeleteWorkspace('tmp')
            DeleteWorkspace('tmp_normalization')

            RenameWorkspace(InputWorkspace='tmp_data', OutputWorkspace='tmpDataMD_{}'.format(j))

            ConvertWANDSCDtoQ(InputWorkspace=ows+'_norm',
                              NormalisationWorkspace=None,
                              UBWorkspace=ows,
                              OutputWorkspace='tmp',
                              Wavelength=lamda,
                              NormaliseBy='Time',
                              Frame='HKL', # not actually HKL,
                              KeepTemporaryWorkspaces=True,
                              Uproj='{},{},{}'.format(*W[:,0]),
                              Vproj='{},{},{}'.format(*W[:,1]),
                              Wproj='{},{},{}'.format(*W[:,2]),
                              BinningDim0='{},{},{}'.format(Q0_bin[0],Q0_bin[2],bins[0]),
                              BinningDim1='{},{},{}'.format(Q1_bin[0],Q1_bin[2],bins[1]),
                              BinningDim2='{},{},{}'.format(Q2_bin[0],Q2_bin[2],bins[2]))

            DeleteWorkspace('tmp')
            DeleteWorkspace('tmp_normalization')

            RenameWorkspace(InputWorkspace='tmp_data', OutputWorkspace='tmpNormMD_{}'.format(j))

            scale = float(mtd['van_'+ows].getExperimentInfo(0).run().getProperty('Sum of Counts').value)
            mtd['tmpNormMD_{}'.format(j)] /= scale

        if j == 0:
            CloneMDWorkspace(InputWorkspace='tmpDataMD_{}'.format(j), OutputWorkspace='dataMD')
            CloneMDWorkspace(InputWorkspace='tmpNormMD_{}'.format(j), OutputWorkspace='normMD')
        else:
            PlusMD(LHSWorkspace='dataMD', RHSWorkspace='tmpDataMD_{}'.format(j), OutputWorkspace='dataMD')
            PlusMD(LHSWorkspace='normMD', RHSWorkspace='tmpNormMD_{}'.format(j), OutputWorkspace='normMD')

            # DeleteWorkspace('tmpDataMD_{}'.format(j))
            # DeleteWorkspace('tmpNormMD_{}'.format(j))

    DivideMD(LHSWorkspace='dataMD', RHSWorkspace='normMD', OutputWorkspace='normDataMD')

    SetMDFrame(InputWorkspace='dataMD', MDFrame='QSample', Axes=[0,1,2])
    SetMDFrame(InputWorkspace='normMD', MDFrame='QSample', Axes=[0,1,2])
    SetMDFrame(InputWorkspace='normDataMD', MDFrame='QSample', Axes=[0,1,2])

    mtd['dataMD'].clearOriginalWorkspaces()
    mtd['normMD'].clearOriginalWorkspaces()
    mtd['normDataMD'].clearOriginalWorkspaces()

    #SaveMD(InputWorkspace='dataMD', Filename='/tmp/dataMD_'+str(key)+'.nxs')
    #SaveMD(InputWorkspace='normMD', Filename='/tmp/normMD_'+str(key)+'.nxs')
    #SaveMD(InputWorkspace='normDataMD', Filename='/tmp/normDataMD_'+str(key)+'.nxs')

    QXaxis = mtd['normDataMD'].getXDimension()
    QYaxis = mtd['normDataMD'].getYDimension()
    QZaxis = mtd['normDataMD'].getZDimension()

    Qx = np.linspace(QXaxis.getMinimum(), QXaxis.getMaximum(), QXaxis.getNBoundaries())
    Qy = np.linspace(QYaxis.getMinimum(), QYaxis.getMaximum(), QYaxis.getNBoundaries())
    Qz = np.linspace(QZaxis.getMinimum(), QZaxis.getMaximum(), QZaxis.getNBoundaries())

    Qx = 0.5*(Qx[1:]+Qx[:-1])
    Qy = 0.5*(Qy[1:]+Qy[:-1])
    Qz = 0.5*(Qz[1:]+Qz[:-1])

    Qx, Qy, Qz = np.meshgrid(Qx, Qy, Qz, indexing='ij')

    Q0 = W[0,0]*Qx+W[0,1]*Qy+W[0,2]*Qz
    Q1 = W[1,0]*Qx+W[1,1]*Qy+W[1,2]*Qz
    Q2 = W[2,0]*Qx+W[2,1]*Qy+W[2,2]*Qz

    mask = mtd['normMD'].getSignalArray() > 0

    Q = np.sqrt(Q0**2+Q1**2+Q2**2)

    Q, Q0, Q1, Q2 = Q[mask], Q0[mask], Q1[mask], Q2[mask]

    data = mtd['dataMD'].getSignalArray().copy()[mask]
    norm = mtd['normMD'].getSignalArray().copy()[mask]

    return Q, Q0, Q1, Q2, data, norm

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

    return pk, bkg, pk_Q0, pk_Q1, pk_Q2, bkg_Q0, bkg_Q1, bkg_Q2 

def norm_integrator_fast(runs, Q0, delta_Q0, Q1, D, W, bin_size=0.025, box_size=2.0, peak_ellipsoid=1.26,
                         inner_bkg_ellipsoid=1.59, outer_bkg_ellipsoid=2.0, bins=[11,11,11], exp=None, close=False):

    if close:
        n = delta_Q0/np.linalg.norm(delta_Q0)
    else:
        n = Q0/np.linalg.norm(Q0)

    n_ind = np.argmin(np.abs(n))

    u = np.zeros(3)
    u[n_ind] = 1

    u = np.cross(n, u)
    u /= np.linalg.norm(u)

    v = np.cross(n, u)
    v *= np.sign(np.dot(np.cross(u, n), v))

    if np.abs(u[1]) > np.abs(v[1]):
        u, v = v, -u

    V = np.column_stack((u,v,n))

    QXaxis = mtd['normDataMD'].getXDimension()
    QYaxis = mtd['normDataMD'].getYDimension()
    QZaxis = mtd['normDataMD'].getZDimension()

    Q_radii = 1/np.sqrt(D.diagonal())

    dQ = box_size*Q_radii

    dQp = np.array([bin_size,bin_size,bin_size])

    D_pk = D/peak_ellipsoid**2
    D_bkg_in = D/inner_bkg_ellipsoid**2
    D_bkg_out = D/outer_bkg_ellipsoid**2

    Q_rot = np.dot(W.T,Q1)

    Q_min, Q_max = Q_rot-dQ, Q_rot+dQ

    _, Q0_bin_size = np.linspace(Q_min[0], Q_max[0], bins[0], retstep=True)
    _, Q1_bin_size = np.linspace(Q_min[1], Q_max[1], bins[1], retstep=True)
    _, Q2_bin_size = np.linspace(Q_min[2], Q_max[2], bins[2], retstep=True)

    if not np.isclose(Q0_bin_size, 0):
        dQp[0] = np.min([Q0_bin_size,bin_size])
    if not np.isclose(Q1_bin_size, 0):
        dQp[1] = np.min([Q1_bin_size,bin_size])
    if not np.isclose(Q2_bin_size, 0):
        dQp[2] = np.min([Q2_bin_size,bin_size])

    QXbin = (QXaxis.getMaximum()-QXaxis.getMinimum())/(QXaxis.getNBoundaries()-1)
    QYbin = (QYaxis.getMaximum()-QYaxis.getMinimum())/(QYaxis.getNBoundaries()-1)
    QZbin = (QZaxis.getMaximum()-QZaxis.getMinimum())/(QZaxis.getNBoundaries()-1)

    dQp[0] = np.max([dQp[0],QXbin])
    dQp[1] = np.max([dQp[1],QYbin])
    dQp[2] = np.max([dQp[2],QZbin])

    Qbins = np.round(2*dQ/dQp).astype(int)+1

    Qx = np.linspace(QXaxis.getMinimum(), QXaxis.getMaximum(), QXaxis.getNBoundaries())
    Qy = np.linspace(QYaxis.getMinimum(), QYaxis.getMaximum(), QYaxis.getNBoundaries())
    Qz = np.linspace(QZaxis.getMinimum(), QZaxis.getMaximum(), QZaxis.getNBoundaries())

    Qx = 0.5*(Qx[1:]+Qx[:-1])
    Qy = 0.5*(Qy[1:]+Qy[:-1])
    Qz = 0.5*(Qz[1:]+Qz[:-1])

    Qx, Qy, Qz = np.meshgrid(Qx, Qy, Qz, indexing='ij', copy=False)

    Qx, Qy, Qz = Qx.flatten(), Qy.flatten(), Qz.flatten()

    Q0 = V[0,0]*Qx+V[0,1]*Qy+V[0,2]*Qz
    Q1 = V[1,0]*Qx+V[1,1]*Qy+V[1,2]*Qz
    Q2 = V[2,0]*Qx+V[2,1]*Qy+V[2,2]*Qz

    Qx = W[0,0]*Q0+W[1,0]*Q1+W[2,0]*Q2
    Qy = W[0,1]*Q0+W[1,1]*Q1+W[2,1]*Q2
    Qz = W[0,2]*Q0+W[1,2]*Q1+W[2,2]*Q2

    Q0_bin_edges = np.histogram_bin_edges(Qx, bins=Qbins[0], range=(Q_min[0],Q_max[0]))
    Q1_bin_edges = np.histogram_bin_edges(Qy, bins=Qbins[1], range=(Q_min[1],Q_max[1]))
    Q2_bin_edges = np.histogram_bin_edges(Qz, bins=Qbins[2], range=(Q_min[2],Q_max[2]))

    Q0_bin_centers = 0.5*(Q0_bin_edges[1:]+Q0_bin_edges[:-1])
    Q1_bin_centers = 0.5*(Q1_bin_edges[1:]+Q1_bin_edges[:-1])
    Q2_bin_centers = 0.5*(Q2_bin_edges[1:]+Q2_bin_edges[:-1])

    Q0_bin_grid, Q1_bin_grid, Q2_bin_grid = np.meshgrid(Q0_bin_centers, Q1_bin_centers, Q2_bin_centers, indexing='ij', copy=False)

    sample = np.array([Qx,Qy,Qz]).T

    Q0_bin = [Q_min[0],dQp[0],Q_max[0]]
    Q1_bin = [Q_min[1],dQp[1],Q_max[1]]
    Q2_bin = [Q_min[2],dQp[2],Q_max[2]]

    box_data, box_norm = [], []

    pk_data, pk_norm = [], []
    bkg_data, bkg_norm = [], []

    for j, r in enumerate(runs):

        data = mtd['tmpDataMD_{}'.format(j)].getSignalArray().copy().flatten()
        norm = mtd['tmpNormMD_{}'.format(j)].getSignalArray().copy().flatten()

        bin_data, _ = np.histogramdd(sample, bins=[Q0_bin_edges,Q1_bin_edges,Q2_bin_edges], weights=data)
        bin_norm, _ = np.histogramdd(sample, bins=[Q0_bin_edges,Q1_bin_edges,Q2_bin_edges], weights=norm)

        box_data.append(bin_data)
        box_norm.append(bin_norm)

        pk, bkg, pk_Q0, pk_Q1, pk_Q2, bkg_Q0, bkg_Q1, bkg_Q2 = partial_integration(bin_data, Q0_bin_grid, Q1_bin_grid, Q2_bin_grid, Q_rot, D_pk, D_bkg_in, D_bkg_out)

        pk_data.append(pk)
        bkg_data.append(bkg)

        pk, bkg, pk_Q0, pk_Q1, pk_Q2, bkg_Q0, bkg_Q1, bkg_Q2 = partial_integration(bin_norm, Q0_bin_grid, Q1_bin_grid, Q2_bin_grid, Q_rot, D_pk, D_bkg_in, D_bkg_out)

        pk_norm.append(pk)
        bkg_norm.append(bkg)

        DeleteWorkspace('tmpDataMD_{}'.format(j))
        DeleteWorkspace('tmpNormMD_{}'.format(j))

    data = mtd['dataMD'].getSignalArray().copy().flatten()
    norm = mtd['normMD'].getSignalArray().copy().flatten()

    bin_data, _ = np.histogramdd(sample, bins=[Q0_bin_edges,Q1_bin_edges,Q2_bin_edges], weights=data)
    bin_norm, _ = np.histogramdd(sample, bins=[Q0_bin_edges,Q1_bin_edges,Q2_bin_edges], weights=norm)

    signal = bin_data/bin_norm
    error = np.sqrt(bin_data)/bin_norm

    pk_bkg_data_norm = (pk_data, pk_norm, bkg_data, bkg_norm, dQp)
    data_norm = (Q0_bin_grid, Q1_bin_grid, Q2_bin_grid, box_data, box_norm)

    pk_bkg_cntrs = (pk_Q0, pk_Q1, pk_Q2, bkg_Q0, bkg_Q1, bkg_Q2)

    Q_scales = np.array([peak_ellipsoid, inner_bkg_ellipsoid, outer_bkg_ellipsoid])

    Q_bin = (Q0_bin, Q1_bin, Q2_bin)

    return Q_bin, Q_rot, Q_radii, Q_scales, signal, error, data_norm, pk_bkg_data_norm, pk_bkg_cntrs

def norm_integrator(facility, instrument, runs, banks, indices, split_angle, Q0, delta_Q0, D, W, bin_size=0.025,
                    box_size=2.0, peak_ellipsoid=1.26, inner_bkg_ellipsoid=1.59, outer_bkg_ellipsoid=2.0, bins=[11,11,11], exp=None, close=False):

    Q_radii = 1/np.sqrt(D.diagonal())

    dQ = box_size*Q_radii

    dQp = np.array([bin_size,bin_size,bin_size])

    D_pk = D/peak_ellipsoid**2
    D_bkg_in = D/inner_bkg_ellipsoid**2
    D_bkg_out = D/outer_bkg_ellipsoid**2

    Q_rot = np.dot(W.T,Q0)

    Q_min, Q_max = Q_rot-dQ, Q_rot+dQ

    ext = np.isclose(Q_min, Q_max)

    if ext.any():
        Q_min[ext] -= 0.05
        Q_max[ext] += 0.05

    _, Q0_bin_size = np.linspace(Q_min[0], Q_max[0], bins[0], retstep=True)
    _, Q1_bin_size = np.linspace(Q_min[1], Q_max[1], bins[1], retstep=True)
    _, Q2_bin_size = np.linspace(Q_min[2], Q_max[2], bins[2], retstep=True)

    if not np.isclose(Q0_bin_size, 0):
        dQp[0] = np.min([Q0_bin_size,bin_size])
    if not np.isclose(Q1_bin_size, 0):
        dQp[1] = np.min([Q1_bin_size,bin_size])
    if not np.isclose(Q2_bin_size, 0):
        dQp[2] = np.min([Q2_bin_size,bin_size])

    Qbins = np.round(2*dQ/dQp).astype(int)+1

    Q0_bin = [Q_min[0],dQp[0],Q_max[0]]
    Q1_bin = [Q_min[1],dQp[1],Q_max[1]]
    Q2_bin = [Q_min[2],dQp[2],Q_max[2]]

    box_data, box_norm = [], []

    pk_data, pk_norm = [], []
    bkg_data, bkg_norm = [], []

    for j, (r, b, i) in enumerate(zip(runs, banks, indices)):

        if facility == 'SNS':
            if np.isclose(split_angle, 0):
                ows = '{}_{}'.format(instrument,r)
            else:
                ows = '{}_{}_{}'.format(instrument,r,b)
        elif instrument == 'HB2C':
            ows = '{}_{}_{}'.format(instrument,r,i)
        else:
            ows = '{}_{}_{}_{}'.format(instrument,exp,r,i)

        omd = ows+'_md'

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
                   OutputWorkspace='normDataMD',
                   OutputDataWorkspace='tmpDataMD_{}'.format(j),
                   OutputNormalizationWorkspace='tmpNormMD_{}'.format(j))

            DeleteWorkspace('normDataMD')

        else:

            lamda = 1.486 if instrument == 'HB2C' else float(mtd[ows].getExperimentInfo(0).run().getProperty('wavelength').value)

            ReplicateMD(ShapeWorkspace=ows, DataWorkspace='van_'+ows, OutputWorkspace=ows+'_norm')

            ConvertWANDSCDtoQ(InputWorkspace=ows,
                              NormalisationWorkspace=None,
                              UBWorkspace=ows,
                              OutputWorkspace='tmp',
                              Wavelength=lamda,
                              NormaliseBy='Time',
                              Frame='HKL', # not actually HKL,
                              KeepTemporaryWorkspaces=True,
                              Uproj='{},{},{}'.format(*W[:,0]),
                              Vproj='{},{},{}'.format(*W[:,1]),
                              Wproj='{},{},{}'.format(*W[:,2]),
                              BinningDim0='{},{},{}'.format(Q0_bin[0],Q0_bin[2],bins[0]),
                              BinningDim1='{},{},{}'.format(Q1_bin[0],Q1_bin[2],bins[1]),
                              BinningDim2='{},{},{}'.format(Q2_bin[0],Q2_bin[2],bins[2]))

            DeleteWorkspace('tmp')
            DeleteWorkspace('tmp_normalization')

            RenameWorkspace(InputWorkspace='tmp_data', OutputWorkspace='tmpDataMD_{}'.format(j))

            ConvertWANDSCDtoQ(InputWorkspace=ows+'_norm',
                              NormalisationWorkspace=None,
                              UBWorkspace=ows,
                              OutputWorkspace='tmp',
                              Wavelength=lamda,
                              NormaliseBy='Time',
                              Frame='HKL', # not actually HKL,
                              KeepTemporaryWorkspaces=True,
                              Uproj='{},{},{}'.format(*W[:,0]),
                              Vproj='{},{},{}'.format(*W[:,1]),
                              Wproj='{},{},{}'.format(*W[:,2]),
                              BinningDim0='{},{},{}'.format(Q0_bin[0],Q0_bin[2],bins[0]),
                              BinningDim1='{},{},{}'.format(Q1_bin[0],Q1_bin[2],bins[1]),
                              BinningDim2='{},{},{}'.format(Q2_bin[0],Q2_bin[2],bins[2]))

            DeleteWorkspace('tmp')
            DeleteWorkspace('tmp_normalization')

            RenameWorkspace(InputWorkspace='tmp_data', OutputWorkspace='tmpNormMD_{}'.format(j))

            scale = float(mtd['van_'+ows].getExperimentInfo(0).run().getProperty('Sum of Counts').value)
            mtd['tmpNormMD_{}'.format(j)] /= scale

        if j == 0:
            CloneMDWorkspace(InputWorkspace='tmpDataMD_{}'.format(j), OutputWorkspace='dataMD')
            CloneMDWorkspace(InputWorkspace='tmpNormMD_{}'.format(j), OutputWorkspace='normMD')
        else:
            PlusMD(LHSWorkspace='dataMD', RHSWorkspace='tmpDataMD_{}'.format(j), OutputWorkspace='dataMD')
            PlusMD(LHSWorkspace='normMD', RHSWorkspace='tmpNormMD_{}'.format(j), OutputWorkspace='normMD')

        bin_data = mtd['tmpDataMD_{}'.format(j)].getSignalArray().copy()
        bin_norm = mtd['tmpNormMD_{}'.format(j)].getSignalArray().copy()

        if j == 0:

            QXaxis = mtd['tmpDataMD_{}'.format(j)].getXDimension()
            QYaxis = mtd['tmpDataMD_{}'.format(j)].getYDimension()
            QZaxis = mtd['tmpDataMD_{}'.format(j)].getZDimension()

            Qx = np.linspace(QXaxis.getMinimum(), QXaxis.getMaximum(), QXaxis.getNBoundaries())
            Qy = np.linspace(QYaxis.getMinimum(), QYaxis.getMaximum(), QYaxis.getNBoundaries())
            Qz = np.linspace(QZaxis.getMinimum(), QZaxis.getMaximum(), QZaxis.getNBoundaries())

            Qx = 0.5*(Qx[1:]+Qx[:-1])
            Qy = 0.5*(Qy[1:]+Qy[:-1])
            Qz = 0.5*(Qz[1:]+Qz[:-1])

            Q0_bin_grid, Q1_bin_grid, Q2_bin_grid = np.meshgrid(Qx, Qy, Qz, indexing='ij')

        signal = bin_data/bin_norm
        error = np.sqrt(bin_data)/bin_norm

        box_data.append(bin_data)
        box_norm.append(bin_norm)

        pk, bkg, pk_Q0, pk_Q1, pk_Q2, bkg_Q0, bkg_Q1, bkg_Q2 = partial_integration(bin_data, Q0_bin_grid, Q1_bin_grid, Q2_bin_grid, Q_rot, D_pk, D_bkg_in, D_bkg_out)

        pk_data.append(pk)
        bkg_data.append(bkg)

        pk, bkg, pk_Q0, pk_Q1, pk_Q2, bkg_Q0, bkg_Q1, bkg_Q2 = partial_integration(bin_norm, Q0_bin_grid, Q1_bin_grid, Q2_bin_grid, Q_rot, D_pk, D_bkg_in, D_bkg_out)

        pk_norm.append(pk)
        bkg_norm.append(bkg)

        DeleteWorkspace('tmpDataMD_{}'.format(j))
        DeleteWorkspace('tmpNormMD_{}'.format(j))

    bin_data = np.sum(box_data, axis=0)
    bin_norm = np.sum(box_norm, axis=0)

    signal = bin_data/bin_norm
    error = np.sqrt(bin_data)/bin_norm

    pk_bkg_data_norm = (pk_data, pk_norm, bkg_data, bkg_norm, dQp)
    data_norm = (Q0_bin_grid, Q1_bin_grid, Q2_bin_grid, box_data, box_norm)

    pk_bkg_cntrs = (pk_Q0, pk_Q1, pk_Q2, bkg_Q0, bkg_Q1, bkg_Q2)

    Q_scales = np.array([peak_ellipsoid, inner_bkg_ellipsoid, outer_bkg_ellipsoid])

    Q_bin = (Q0_bin, Q1_bin, Q2_bin)

    return Q_bin, Q_rot, Q_radii, Q_scales, signal, error, data_norm, pk_bkg_data_norm, pk_bkg_cntrs

def load_normalization_calibration(facility, instrument, spectrum_file, counts_file,
                                   tube_calibration, detector_calibration, mask_file):

    print(spectrum_file, counts_file)

    if instrument == 'CORELLI':
        k_min, k_max = 2.5, 10
    elif instrument == 'TOPAZ':
        k_min, k_max = 1.8, 18
    elif instrument == 'MANDI':
        k_min, k_max = 1.5, 3.0
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
            if instrument == 'HB3A':
                LoadMD(Filename=counts_file, OutputWorkspace='van')
            else:
                LoadWANDSCD(Filename=counts_file, 
                            NormalizedBy='None',
                            Grouping='None',
                            OutputWorkspace='van')

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

    if mtd.doesExist('sa') and mask_file is not None:
        LoadMask(Instrument=instrument, InputFile=mask_file, OutputWorkspace='mask')
        MaskDetectors(Workspace='sa', MaskedWorkspace='mask')
        DeleteWorkspace('mask')

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

def pre_integration(runs, proc, outname, outdir, dbgdir, directory, facility, instrument, ipts, all_runs, ub_file, reflection_condition, min_d,
                    spectrum_file, counts_file, tube_calibration, detector_calibration, mask_file,
                    mod_vector_1=[0,0,0], mod_vector_2=[0,0,0], mod_vector_3=[0,0,0],
                    max_order=0, cross_terms=False, exp=None, tmp=None):

    # peak centroid radius ---------------------------------------------------------
    centroid_radius = 0.125

    load_normalization_calibration(facility, instrument, spectrum_file, counts_file,
                                   tube_calibration, detector_calibration, mask_file)

    CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace='tmp_lean')

    CreateEmptyTableWorkspace(OutputWorkspace='run_info')

    mtd['run_info'].addColumn('Int', 'RunNumber')
    mtd['run_info'].addColumn('Double', 'Scale')

    if instrument == 'HB3A':
        LoadEmptyInstrument(InstrumentName='HB3A', OutputWorkspace='rws')
    elif instrument == 'HB2C':
        LoadEmptyInstrument(InstrumentName='WAND', OutputWorkspace='rws')

    if mtd.doesExist('sa'):
        CreatePeaksWorkspace(InstrumentWorkspace='sa', NumberOfPeaks=0, OutputType='Peak', OutputWorkspace='tmp')
        CreatePeaksWorkspace(InstrumentWorkspace='sa', NumberOfPeaks=0, OutputType='Peak', OutputWorkspace='tmp_ellip')
    else:
        CreatePeaksWorkspace(InstrumentWorkspace='van', NumberOfPeaks=0, OutputType='Peak', OutputWorkspace='tmp')
       
    for i, r in enumerate(runs):

        if facility == 'SNS' or instrument == 'HB2C':
            ows = '{}_{}'.format(instrument,r)
        else:
            ows = '{}_{}_{}'.format(instrument,exp,r)
       
        print('Process {} integrating {}'.format(proc,ows))

        omd = ows+'_md'
        opk = ows+'_pk'

        if mtd.doesExist('md'):
            RenameWorkspace(InputWorkspace='md', OutputWorkspace=omd)

        if not mtd.doesExist(opk) and not mtd.doesExist('pks'):

            if facility == 'SNS':
                filename = '/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(instrument,ipts,instrument,r)
                LoadEventNexus(Filename=filename, OutputWorkspace=ows)

                if mtd.doesExist('sa'):
                    #MaskDetectors(Workspace=ows, MaskedWorkspace='mask')

                    if instrument == 'SNAP':
                        if not mtd['sa'].run().hasProperty('det_arc1') or \
                           not mtd['sa'].run().hasProperty('det_arc2') or \
                           not mtd['sa'].run().hasProperty('det_lin1') or \
                           not mtd['sa'].run().hasProperty('det_lin2'):
                            LoadNexusLogs(Workspace='sa', Filename=filename, OverwriteLogs=False)
                            LoadInstrument(Workspace='sa', InstrumentName='SNAP', RewriteSpectraMap=False)

                            load_normalization_calibration(facility, instrument, spectrum_file, counts_file,
                                                           tube_calibration, detector_calibration, mask_file)

                    # CopyInstrumentParameters(InputWorkspace='sa', OutputWorkspace=ows)
                    if mtd.doesExist('tube_table'):
                        ApplyCalibration(Workspace=ows, CalibrationTable='tube_table')

                    if detector_calibration is not None:
                        ext = os.path.splitext(detector_calibration)[1]
                        if ext == '.xml':
                            LoadParameterFile(Workspace=ows, Filename=detector_calibration)
                        else:
                            LoadIsawDetCal(InputWorkspace=ows, Filename=detector_calibration)
  
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
                    k_max, two_theta_max = 3.0, 160
                elif instrument == 'SNAP':
                    k_max, two_theta_max = 12.5, 138

                lamda_min = 2*np.pi/k_max  

                Q_max = 4*np.pi/lamda_min*np.sin(np.deg2rad(two_theta_max)/2)

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
                            MinValues='{},{},{}'.format(-Q_max,-Q_max,-Q_max),
                            MaxValues='{},{},{}'.format(Q_max,Q_max,Q_max),
                            Uproj='1,0,0',
                            Vproj='0,1,0',
                            Wproj='0,0,1')

            else:

                if instrument == 'HB3A':
                    filename = '/HFIR/{}/IPTS-{}/shared/autoreduce/{}_exp{:04}_scan{:04}.nxs'.format(instrument,ipts,instrument,exp,r)
                    LoadMD(Filename=filename, OutputWorkspace=ows)

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

                    SetGoniometer(Workspace=ows,
                                  Axis0='omega,0,1,0,-1',
                                  Axis1='chi,0,0,1,-1',
                                  Axis2='phi,0,1,0,-1',
                                  Average=False)

                elif instrument == 'HB2C':

                    LoadWANDSCD(IPTS=ipts,
                                RunNumbers=','.join([str(val) for val in all_runs]),
                                NormalizedBy='None',
                                Grouping='4x4', 
                                OutputWorkspace=ows)

                    SetGoniometer(Workspace=ows,
                                  Axis0='s1,0,1,0,1',
                                  Average=False)

                lamda = 1.486 if instrument == 'HB2C' else float(mtd[ows].getExperimentInfo(0).run().getProperty('wavelength').value)

                if instrument == 'HB3A':
                    two_theta_max = 155
                elif instrument == 'HB2C':
                    two_theta_max = 156

                Q_max = 4*np.pi/lamda*np.sin(np.deg2rad(two_theta_max)/2)

                ConvertHFIRSCDtoMDE(InputWorkspace=ows,
                                    Wavelength=lamda,
                                    MinValues='{},{},{}'.format(-Q_max,-Q_max,-Q_max),
                                    MaxValues='{},{},{}'.format(Q_max,Q_max,Q_max),
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

        if not mtd.doesExist(opk) and not mtd.doesExist('pks'):

            if facility == 'SNS':

                if instrument == 'CORELLI':
                    k_min, k_max = 2.5, 10
                elif instrument == 'TOPAZ':
                    k_min, k_max = 1.8, 12.5
                elif instrument == 'MANDI':
                    k_min, k_max = 1.5, 3.0
                elif instrument == 'SNAP':
                    k_min, k_max = 1.8, 12.5

                lamda_min, lamda_max = 2*np.pi/k_max, 2*np.pi/k_min

                Q_max = 4*np.pi/lamda_min*np.sin(np.deg2rad(two_theta_max)/2)

                min_d_spacing = 2*np.pi/Q_max
                max_d_spacing = np.max([mtd[omd].getExperimentInfo(0).sample().getOrientedLattice().d(*hkl) for hkl in [(1,0,0),(0,1,0),(0,0,1)]])

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

                lamda = 1.486 if instrument == 'HB2C' else float(run.getProperty('wavelength').value)

                lamda_min, lamda_max = 0.95*lamda, 1.05*lamda

                gon = run.getGoniometer().getEulerAngles('YZY')

                if instrument == 'HB3A':

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

                else:

                    use_inner = False

                    min_angle = run.getLogData('s1').value.min()
                    max_angle = run.getLogData('s1').value.max()

                min_d_spacing = 2*np.pi/Q_max
                max_d_spacing = np.max([mtd[omd].getExperimentInfo(0).sample().getOrientedLattice().d(*hkl) for hkl in [(1,0,0),(0,1,0),(0,0,1)]])

                PredictPeaks(InputWorkspace=omd,
                             WavelengthMin=lamda_min,
                             WavelengthMax=lamda_max,
                             MinDSpacing=min_d_spacing,
                             MaxDSpacing=max_d_spacing,
                             ReflectionCondition=reflection_condition,
                             CalculateGoniometerForCW=False,
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
                    pk.setWavelength(lamda)

                for col in ['h', 'k', 'l']:
                    SortPeaksWorkspace(InputWorkspace=opk, 
                                       ColumnNameToSortBy=col,
                                       SortAscending=False,
                                       OutputWorkspace=opk)

                for pn in range(mtd[opk].getNumberPeaks()-1,0,-1):
                    pk_1, pk_2 = mtd[opk].getPeak(pn), mtd[opk].getPeak(pn-1)
                    if np.allclose([list(pk_1.getHKL())],
                                   [list(pk_2.getHKL())]):
                        mtd[opk].removePeak(pn)

                for pn in range(mtd[opk].getNumberPeaks()):
                    pk = mtd[opk].getPeak(pn)
                    pk.setRunNumber(r)
                    pk.setPeakNumber(pn+1)
                    
            if max_order > 0:

                if facility == 'HFIR':

                    PredictPeaks(InputWorkspace=omd,
                                 WavelengthMin=lamda_min,
                                 WavelengthMax=lamda_max,
                                 MinDSpacing=min_d_spacing,
                                 MaxDSpacing=max_d_spacing,
                                 ReflectionCondition=reflection_condition,
                                 CalculateGoniometerForCW=False,
                                 CalculateWavelength=False,
                                 Wavelength=lamda,
                                 InnerGoniometer=use_inner,
                                 MinAngle=min_angle,
                                 MaxAngle=max_angle,
                                 FlipX=True if instrument == 'HB3A' else False,
                                 OutputType='LeanElasticPeak',
                                 OutputWorkspace='main')

                    HFIRCalculateGoniometer(Workspace='main',
                                            Wavelength=lamda,
                                            OverrideProperty=True,
                                            InnerGoniometer=use_inner,
                                            FlipX=True if instrument == 'HB3A' else False)

                    for pn in range(mtd['main'].getNumberPeaks()):
                        pk = mtd['main'].getPeak(pn)
                        pk.setWavelength(lamda)

                    for col in ['h', 'k', 'l']:
                        SortPeaksWorkspace(InputWorkspace='main', 
                                           ColumnNameToSortBy=col,
                                           SortAscending=False,
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

                    # PredictSatellitePeaks(Peaks=opk,
                    #                       SatellitePeaks='sat',
                    #                       WavelengthMin=lamda_min,
                    #                       WavelengthMax=lamda_max,
                    #                       MinDSpacing=min_d_spacing,
                    #                       MaxDSpacing=max_d_spacing,
                    #                       ModVector1=mod_vector_1,
                    #                       ModVector2=mod_vector_2,
                    #                       ModVector3=mod_vector_3,
                    #                       MaxOrder=max_order,
                    #                       CrossTerms=cross_terms,
                    #                       IncludeIntegerHKL=False,
                    #                       IncludeAllPeaksInRange=True)

                    # for pn in range(mtd['sat'].getNumberPeaks()-1,0,-1):
                    #     print(mtd['sat'].getPeak(pn).getHKL())
                    #     print(mtd['sat'].getPeak(pn).getQSampleFrame())
                    #     print(mtd['sat'].getPeak(pn).getGoniometerMatrix())

                    HFIRCalculateGoniometer(Workspace='sat',
                                            Wavelength=lamda,
                                            OverrideProperty=True,
                                            InnerGoniometer=use_inner,
                                            FlipX=True if instrument == 'HB3A' else False)

                    for col in ['h', 'k', 'l']:
                        SortPeaksWorkspace(InputWorkspace='sat', 
                                           ColumnNameToSortBy=col,
                                           SortAscending=False,
                                           OutputWorkspace='sat')

                    for pn in range(mtd['sat'].getNumberPeaks()):
                        pk = mtd['sat'].getPeak(pn)
                        pk.setWavelength(lamda)

                    for pn in range(mtd['sat'].getNumberPeaks()-1,0,-1):
                        pk_1, pk_2 = mtd['sat'].getPeak(pn), mtd['sat'].getPeak(pn-1)
                        print(pk_1.getHKL(),pk_2.getHKL())
                        if np.allclose([list(pk_1.getHKL())],
                                       [list(pk_2.getHKL())]):
                            mtd['sat'].removePeak(pn)

                    ns = mtd[opk].getNumberPeaks()
                    for pn in range(mtd['sat'].getNumberPeaks()):
                        pk = mtd['sat'].getPeak(pn)
                        pk.setRunNumber(r)
                        pk.setPeakNumber(pk.getPeakNumber()+ns+1)

                    ConvertPeaksWorkspace(PeakWorkspace='sat',
                                          InstrumentWorkspace=opk,
                                          OutputWorkspace='sat')

                    DeleteWorkspace('main')

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
                    elif reflection_condition == 'All-face centred':
                        allowed = (h + l) % 2 == 0 and (k + l) % 2 == 0 and (h + k) % 2 == 0
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

            FilterPeaks(InputWorkspace=opk,
                        FilterVariable='QMod',
                        FilterValue=0,
                        Operator='>',
                        OutputWorkspace=opk)

            FilterPeaks(InputWorkspace=opk,
                        FilterVariable='h^2+k^2+l^2',
                        FilterValue=0,
                        Operator='>',
                        OutputWorkspace=opk)

            IntegratePeaksMD(InputWorkspace=omd,
                             PeakRadius=centroid_radius,
                             BackgroundInnerRadius=centroid_radius+0.01,
                             BackgroundOuterRadius=centroid_radius+0.02,
                             PeaksWorkspace=opk,
                             OutputWorkspace=opk)

            FilterPeaks(InputWorkspace=opk,
                        FilterVariable='Intensity',
                        FilterValue=0,
                        Operator='>',
                        OutputWorkspace=opk)

            if facility == 'SNS':

                IntegrateEllipsoids(InputWorkspace=ows, 
                                    PeaksWorkspace=opk,
                                    OutputWorkspace=opk+'_ellip',
                                    RegionRadius=2*centroid_radius,
                                    CutoffIsigI=3,
                                    NumSigmas=3,
                                    IntegrateIfOnEdge=True)

                FilterPeaks(InputWorkspace=opk+'_ellip',
                            FilterVariable='Intensity',
                            FilterValue=0,
                            Operator='>',
                            OutputWorkspace=opk+'_ellip')   

                FilterPeaks(InputWorkspace=opk+'_ellip',
                            FilterVariable='Signal/Noise',
                            FilterValue=30,
                            Operator='>',
                            OutputWorkspace=opk+'_ellip')   

            det_IDs = mtd[opk].column(1)

            for pn in range(mtd[opk].getNumberPeaks()-1,-1,-1):
                if det_IDs[pn] == -1:
                    mtd[opk].removePeak(pn)

            if instrument == 'HB2C':
                g = Goniometer()
                run_numbers = np.array(mtd[ows].getExperimentInfo(0).run().getProperty('run_number').value).astype(int).tolist()
                s1 = np.array(mtd[ows].getExperimentInfo(0).run().getProperty('s1').value)
                for pn in range(mtd[opk].getNumberPeaks()):
                    pk = mtd[opk].getPeak(pn)
                    R = pk.getGoniometerMatrix()
                    g.setR(R)
                    rot, _, _ = g.getEulerAngles('YZY')
                    run = run_numbers[np.argmin(np.abs(s1-rot))]
                    pk.setRunNumber(run)

            SortPeaksWorkspace(InputWorkspace=opk, 
                               ColumnNameToSortBy='DSpacing',
                               SortAscending=False,
                               OutputWorkspace=opk)

            for pn in range(mtd[opk].getNumberPeaks()-1,0,-1):
                pk_1, pk_2 = mtd[opk].getPeak(pn), mtd[opk].getPeak(pn-1)
                if np.allclose([pk_1.getRunNumber(), *list(pk_1.getHKL())],
                               [pk_2.getRunNumber(), *list(pk_2.getHKL())]): 
                    mtd[opk].removePeak(pn)

            if instrument == 'HB2C':
                CloneWorkspace(InputWorkspace=opk, OutputWorkspace='pks')
                DeleteWorkspace(opk)
                for run_no in list(set(mtd['pks'].column(0))):
                    if run_no in runs:
                        FilterPeaks(InputWorkspace='pks',
                                    FilterVariable='RunNumber',
                                    FilterValue=run_no,
                                    Operator='=',
                                    OutputWorkspace='{}_{}_pk'.format(instrument,run_no))

        if facility == 'HFIR' and mtd.doesExist(opk):

            det_IDs = mtd[opk].column(1)

            if instrument == 'HB3A':
                columns = 512*3
                rows = 512
            else:
                columns = 512
                rows = 8*480

            run = mtd[omd].getExperimentInfo(0).run()
            scans = run.getNumGoniometers()

            signs = np.array([1 if run.getGoniometer(i).getR()[0,2] > run.getGoniometer(i).getR()[2,0] else -1 for i in range(scans)])

            rotation = signs*np.rad2deg(np.array([np.arccos((np.trace(run.getGoniometer(i).getR())-1)/2) for i in range(scans)]))

            rotation_start, rotation_stop = rotation[0], rotation[-1]

            rotation_diff = np.diff(rotation).mean()

            r_diff = 128
            c_diff = 128

            i_diff = abs(int(round(10/rotation_diff)))

            if instrument == 'HB3A':

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

            else:

                detz = np.mean(run.getProperty('HB2C:Mot:detz.RBV').value)
                s2 = np.mean(run.getProperty('HB2C:Mot:s2.RBV').value)

                AddSampleLog(Workspace='rws',
                             LogName='HB2C:Mot:detz.RBV', 
                             LogText=str(detz),
                             LogType='Number Series',
                             LogUnit='degree',
                             NumberType='Double')

                AddSampleLog(Workspace='rws',
                             LogName='HB2C:Mot:s2.RBV', 
                             LogText=str(s2),
                             LogType='Number Series',
                             LogUnit='millimeter',
                             NumberType='Double')

                LoadInstrument(Workspace='rws', InstrumentName='WAND', RewriteSpectraMap=False)

            if instrument == 'HB3A':

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

            twotheta = np.array(mtd['preproc'].column(2))
            azimuthal = np.array(mtd['preproc'].column(3))

            if instrument == 'HB3A':
                twotheta = twotheta.reshape(columns,rows)
                azimuthal = azimuthal.reshape(columns,rows)
            else:
                twotheta = twotheta.reshape(rows,columns)
                azimuthal = azimuthal.reshape(rows,columns)

            if instrument == 'HB3A':
                logs = ['omega', 'chi', 'phi', 'time', 'monitor', 'monitor2']
            else:
                logs = ['s1', 'duration', 'monitor_count']

            log_dict = {}

            for log in logs:
                if run.hasProperty(log):
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

                #start, stop = 512*(bank-1), 512*bank

                pk.setBinCount(bank)

                R = pk.getGoniometerMatrix()

                sign = 1 if R[0,2] > R[2,0] else -1

                rot = sign*np.rad2deg(np.arccos((np.trace(R)-1)/2))

                i_ind = int(round((rot-rotation_start)/(rotation_stop-rotation_start)*(scans-1))) 

                i_start = i_ind-i_diff if i_ind-i_diff > 0 else 0
                i_stop = i_ind+i_diff if i_ind+i_diff < scans else scans

                if instrument == 'HB2C':
                    run_nos = run_numbers[i_start:i_stop]
                    ows = '{}_{}'.format(instrument,r)
                    opk = ows+'_pk'
                    LoadWANDSCD(IPTS=ipts,
                                RunNumbers=','.join([str(val) for val in run_nos]),
                                NormalizedBy='None',
                                Grouping='None', 
                                OutputWorkspace=ows)
                    i_start, i_stop = 0, len(run_nos)

                if instrument == 'HB3A':
                    c_ind, r_ind = np.unravel_index(det_ID, (columns, rows))
                else:
                    r_ind, c_ind = np.unravel_index(det_ID, (rows, columns))

                r_start = r_ind-r_diff if r_ind-r_diff > 0 else 0
                r_stop = r_ind+r_diff if r_ind+r_diff < rows else rows

                c_start = c_ind-c_diff if c_ind-c_diff > 0 else 0
                c_stop = c_ind+c_diff if c_ind+c_diff < columns else columns

                SliceMDHisto(InputWorkspace=ows,
                             Start='{},{},{}'.format(c_start,r_start,i_start),
                             End='{},{},{}'.format(c_stop,r_stop,i_stop),
                             OutputWorkspace='crop_data')

                SliceMDHisto(InputWorkspace='van',
                             Start='{},{},{}'.format(c_start,r_start,0),
                             End='{},{},{}'.format(c_stop,r_stop,1),
                             OutputWorkspace='crop_norm')

                crop_run = mtd['crop_data'].getExperimentInfo(0).run()

                for log in logs:
                    if log_dict.get(log) is not None:
                        prop_times, prop_values = log_dict[log]
                        times, values = prop_times[i_start:i_stop], prop_values[i_start:i_stop]
                        new_log = FloatTimeSeriesProperty(log)
                        for t, v in zip(times, values):
                            new_log.addValue(t, v)
                        crop_run[log] = new_log

                if instrument == 'HB3A':
                    crop_run.addProperty('twotheta', twotheta[c_start:c_stop,r_start:r_stop].T.flatten().tolist(), True)
                    crop_run.addProperty('azimuthal', azimuthal[c_start:c_stop,r_start:r_stop].T.flatten().tolist(), True)
                else:
                    crop_run.addProperty('twotheta', twotheta[r_start:r_stop,c_start:c_stop].flatten().tolist(), True)
                    crop_run.addProperty('azimuthal', azimuthal[r_start:r_stop,c_start:c_stop].flatten().tolist(), True)

                tmp_directory = '{}/{}/'.format(dbgdir, tmp)

                if instrument == 'HB3A':
                    tmp_outname = '{}_{}_{}_{}.nxs'.format(instrument,exp,r,no)
                else:
                    tmp_outname = '{}_{}_{}.nxs'.format(instrument,r,no)

                SaveMD(InputWorkspace='crop_data', Filename=os.path.join(tmp_directory, tmp_outname))
                SaveMD(InputWorkspace='crop_norm', Filename=os.path.join(tmp_directory, 'van_'+tmp_outname))

                DeleteWorkspace('crop_data')
                DeleteWorkspace('crop_norm')

        if mtd.doesExist(opk):

            ConvertPeaksWorkspace(PeakWorkspace=opk, OutputWorkspace=opk+'_lean')

            CombinePeaksWorkspaces(LHSWorkspace='tmp_lean', RHSWorkspace=opk+'_lean', OutputWorkspace='tmp_lean')
            SetUB(Workspace='tmp_lean', UB=mtd[opk+'_lean'].sample().getOrientedLattice().getUB())

            CombinePeaksWorkspaces(LHSWorkspace='tmp', RHSWorkspace=opk, OutputWorkspace='tmp')
            SetUB(Workspace='tmp', UB=mtd[opk].sample().getOrientedLattice().getUB())
            
            if mtd.doesExist('tmp_ellip'):
                CombinePeaksWorkspaces(LHSWorkspace='tmp_ellip', RHSWorkspace=opk+'_ellip', OutputWorkspace='tmp_ellip')
                SetUB(Workspace='tmp_ellip', UB=mtd[opk].sample().getOrientedLattice().getUB())

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

        if instrument == 'HB2C':
            RenameWorkspace(InputWorkspace=omd, OutputWorkspace='md')

        if facility == 'SNS':
            norm_scale = mtd[ows].getRun().getPropertyAsSingleValueWithTimeAveragedMean('gd_prtn_chrg')
        else:
            norm_scale = 1
        
        mtd['run_info'].addRow([r,norm_scale])

        if mtd.doesExist(ows):
            DeleteWorkspace(ows)
        if mtd.doesExist(opk):
            DeleteWorkspace(opk)
        if mtd.doesExist(omd):
            DeleteWorkspace(omd)
        if mtd.doesExist(opk+'_lean'):
            DeleteWorkspace(opk+'_lean')
        if mtd.doesExist(opk+'_ellip'):
            DeleteWorkspace(opk+'_ellip')

    if mtd.doesExist('md'):
        DeleteWorkspace('md')

    SaveNexus(InputWorkspace='run_info', Filename=os.path.join(dbgdir, outname+'_log.nxs'))
    SaveNexus(InputWorkspace='tmp_lean', Filename=os.path.join(dbgdir, outname+'_pk_lean.nxs'))
    SaveNexus(InputWorkspace='tmp', Filename=os.path.join(dbgdir, outname+'_pk.nxs'))
    SaveIsawUB(InputWorkspace='tmp', Filename=os.path.join(dbgdir, outname+'.mat'))
    SaveIsawUB(InputWorkspace='tmp', Filename=os.path.join(dbgdir, outname+'.mat'))

    if mtd.doesExist('tmp_ellip'):
        SaveNexus(InputWorkspace='tmp_ellip', Filename=os.path.join(dbgdir, outname+'_pk_ellip.nxs'))

    if mtd.doesExist('sa'):
        DeleteWorkspace('sa')    
    if mtd.doesExist('flux'):
        DeleteWorkspace('flux')
    if mtd.doesExist('van'):
        DeleteWorkspace('van')
 
def partial_load(facility, instrument, runs, banks, indices, phi, chi, omega, norm_scale, split_angle,
                 dbgdir, ipts, outname, detector_calibration, elastic, timing_offset, exp=None, tmp=None):

    for r, b, i, p, c, o in zip(runs, banks, indices, phi, chi, omega):

        if facility == 'SNS':
            if np.isclose(split_angle, 0):
                ows = '{}_{}'.format(instrument,r)
            else:
                ows = '{}_{}_{}'.format(instrument,r,b)
        elif instrument == 'HB2C':
            ows = '{}_{}_{}'.format(instrument,r,i)
        else:
            ows = '{}_{}_{}_{}'.format(instrument,exp,r,i)

        omd = ows+'_md'

        if facility == 'SNS':

            if not mtd.doesExist(omd):

                filename = '/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(instrument,ipts,instrument,r)

                if split_angle > 0:

                    if instrument == 'CORELLI':
                        if b == 1 or b == 91:
                            bank = 'bank{}'.format(b)
                        else:
                            bank = ','.join(['bank{}'.format(bank) for bank in [b-1,b,b+1]])
                    else:
                        bank = 'bank{}'.format(b)

                    LoadEventNexus(Filename=filename, 
                                   BankName=bank, 
                                   SingleBankPixelsOnly=True,
                                   Precount=True,
                                   LoadLogs=False,
                                   LoadNexusInstrumentXML=False,
                                   OutputWorkspace=ows)

                    if elastic:
                        LoadNexusLogs(Workspace=ows, Filename=filename, AllowList='chopper4_TDC,BL9:Chop:Skf4:MotorSpeed')
                        CopyInstrumentParameters(InputWorkspace=instrument, OutputWorkspace=ows)
                        CorelliCrossCorrelate(InputWorkspace=ows, OutputWorkspace=ows, TimingOffset=timing_offset)

                else:

                    LoadEventNexus(Filename=filename, 
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

                    if split_angle > 0:
                        CopyInstrumentParameters(InputWorkspace='sa', OutputWorkspace=ows)
                        if mtd.doesExist('tube_table'):
                            ApplyCalibration(Workspace=ows, CalibrationTable='tube_table')

                    else:
                        if detector_calibration is not None:
                            ext = os.path.splitext(detector_calibration)[1]
                            if ext == '.xml':
                                LoadParameterFile(Workspace=ows, Filename=detector_calibration)
                            else:
                                LoadIsawDetCal(InputWorkspace=ows, Filename=detector_calibration)

                    MaskDetectors(Workspace=ows, MaskedWorkspace='mask')

                SetGoniometer(Workspace=ows, Goniometers='Universal')

                if mtd.doesExist('flux'):

                    ConvertUnits(InputWorkspace=ows, OutputWorkspace=ows, EMode='Elastic', Target='Momentum')

                    CropWorkspaceForMDNorm(InputWorkspace=ows,
                                           XMin=mtd['flux'].dataX(0).min(),
                                           XMax=mtd['flux'].dataX(0).max(),
                                           OutputWorkspace=ows)

                min_vals, max_vals = ConvertToMDMinMaxGlobal(InputWorkspace=ows,
                                                             QDimensions='Q3D',
                                                             dEAnalysisMode='Elastic',
                                                             Q3DFrames='Q')

                if np.isinf(min_vals).any() or np.isinf(max_vals).any():
                    min_vals, max_vals = [-20,-20,-20], [20,20,20]

                ConvertToMD(InputWorkspace=ows,
                            OutputWorkspace=omd,
                            QDimensions='Q3D',
                            dEAnalysisMode='Elastic',
                            Q3DFrames='Q_sample',
                            LorentzCorrection=False,
                            PreprocDetectorsWS='-',
                            MinValues=min_vals,
                            MaxValues=max_vals,
                            Uproj='1,0,0',
                            Vproj='0,1,0',
                            Wproj='0,0,1')

                RecalculateTrajectoriesExtents(InputWorkspace=omd,
                                               OutputWorkspace=omd)

                DeleteWorkspace(ows)

        else:

            if not mtd.doesExist(ows):

                filename = '{}/{}/{}.nxs'.format(dbgdir, tmp, ows)
                LoadMD(Filename=filename, OutputWorkspace=ows)
                filename = '{}/{}/{}.nxs'.format(dbgdir, tmp, 'van_'+ows)
                LoadMD(Filename=filename, OutputWorkspace='van_'+ows)

                if instrument == 'HB3A':
                    SetGoniometer(Workspace=ows,
                                  Axis0='omega,0,1,0,-1',
                                  Axis1='chi,0,0,1,-1',
                                  Axis2='phi,0,1,0,-1',
                                  Average=False)
                else:
                    SetGoniometer(Workspace=ows,
                                  Axis0='s1,0,1,0,1',
                                  Average=False)

def partial_cleanup(runs, banks, indices, facility, instrument, split_angle, runs_banks, run_keys, bank_keys, bank_group, key, exp=None):

    for r, b, i in zip(runs, banks, indices):

        if facility == 'SNS':
            if np.isclose(split_angle, 0):
                ows = '{}_{}'.format(instrument,r)
            else:
                ows = '{}_{}_{}'.format(instrument,r,b)            
        elif instrument == 'HB2C':
            ows = '{}_{}_{}'.format(instrument,r,i)
        else:
            ows = '{}_{}_{}_{}'.format(instrument,exp,r,i)

        omd = ows+'_md'

        peak_keys = runs_banks[(r,b)]
        peak_keys.remove(key)
        runs_banks[(r,b)] = peak_keys

        run_key_list = run_keys[r]
        run_key_list.remove(key)
        run_keys[r] = run_key_list

        bank_key_list = bank_keys[b]
        bank_key_list.remove(key)
        bank_keys[b] = bank_key_list

        if facility == 'SNS':

            #print(peak_keys)

            if split_angle > 0:
                if len(peak_keys) == 0 or psutil.virtual_memory().percent > 85:
                    if mtd.doesExist(omd):
                        DeleteWorkspace(omd)
            else:
                if len(run_key_list) == 0 or psutil.virtual_memory().percent > 85:
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
            if mtd.doesExist('van_'+ows):
                DeleteWorkspace('van_'+ows)

    return runs_banks, run_keys, bank_keys

def integration_loop(keys, inds, proc, outname, ref_dict, int_list, filename, box_fit_size,
                     spectrum_file, counts_file, tube_calibration, detector_calibration, mask_file,
                     outdir, dbgdir, directory, facility, instrument, ipts, runs,
                     split_angle, min_d, min_d_sat, a, b, c, alpha, beta, gamma, reflection_condition,
                     mod_vector_1, mod_vector_2, mod_vector_3, max_order, cross_terms,
                     chemical_formula, z_parameter, sample_mass, elastic, timing_offset, experiment, tmp, cluster):

    if elastic:
        LoadEmptyInstrument(InstrumentName='CORELLI', OutputWorkspace='CORELLI')

    scale_constant = 1e+4

    load_normalization_calibration(facility, instrument, spectrum_file, counts_file,
                                   tube_calibration, detector_calibration, mask_file)

    if instrument == 'HB3A':
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

        if min_d is not None:
            FilterPeaks(InputWorkspace=opk.format(r),
                        OutputWorkspace=opk.format(r),
                        FilterVariable='DSpacing',
                        FilterValue=min_d, 
                        Operator='>')

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

        if mtd.doesExist('flux'):
            lamda_min = 2*np.pi/mtd['flux'].dataX(0).max()
            lamda_max = 2*np.pi/mtd['flux'].dataX(0).min()
        else:
            lamda_min = None
            lamda_max = None

        print('Adding run {}'.format(opk.format(r)), cluster, lamda_min, lamda_max)

        peak_dictionary.add_peaks(opk.format(r), cluster, lamda_min, lamda_max)
        
        DeleteWorkspace(opk.format(r))

    peak_dictionary.split_peaks(split_angle)
    peak_dict = peak_dictionary.to_be_integrated()

    peak_envelope = PeakEnvelope()
    peak_envelope.show_plots(False)

    DeleteWorkspace('tmp')

    norm_scale = {}

    LoadNexus(Filename=os.path.join(dbgdir, filename+'_log.nxs'), OutputWorkspace='log')
    for j in range(mtd['log'].rowCount()):
        items = mtd['log'].row(j)
        r, scale = items.values()
        norm_scale[r] = scale

    bank_set = set()

    runs_banks = {}
    run_keys = {}
    bank_keys = {}

    peak_dictionary.construct_tree()

    for key, ind in zip(keys,inds):

        key = tuple(key)

        peak = peak_dict[key][ind]

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

            if run_keys.get(r) is None:
                run_keys[r] = [key]
            else:
                key_list = run_keys[r]
                key_list.append(key)
                run_keys[r] = key_list

            if bank_keys.get(b) is None:
                bank_keys[b] = [key]
            else:
                key_list = bank_keys[b]
                key_list.append(key)
                bank_keys[b] = key_list

    banks = list(bank_set)

    # if mtd.doesExist('van'):
    #     for b in banks:
    #         start, stop = 512*(b-1), 512*b
    #         SliceMDHisto(InputWorkspace='van',
    #                      Start='{},0,0'.format(start),
    #                      End='{},512,1'.format(stop),
    #                      OutputWorkspace='van_{}'.format(b))
    #     van_ws = ['van_{}'.format(b) for b in banks]
    #     if len(van_ws) > 0:
    #         GroupWorkspaces(InputWorkspaces=','.join(van_ws), OutputWorkspace='van')

    bank_group = {}

    if mtd.doesExist('flux'):

        ExtractMask(InputWorkspace='sa', OutputWorkspace='mask')

        if instrument == 'SNAP':
            logfile = '/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(instrument,ipts,instrument,r)
            LoadNexusLogs(Workspace='flux', Filename=logfile, OverwriteLogs=False)
            LoadNexusLogs(Workspace='sa', Filename=logfile, OverwriteLogs=False)
            LoadInstrument(Workspace='flux', InstrumentName='SNAP', RewriteSpectraMap=False)
            LoadInstrument(Workspace='sa', InstrumentName='SNAP', RewriteSpectraMap=False)

            load_normalization_calibration(facility, instrument, spectrum_file, counts_file,
                                           tube_calibration, detector_calibration, mask_file)

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

    peak_summary = open(os.path.join(dbgdir, '{}_summary.txt'.format(outname)), 'w')
    excl_summary = open(os.path.join(dbgdir, 'rej_{}_summary.txt'.format(outname)), 'w')

    peak_stats = open(os.path.join(dbgdir, '{}_stats.txt'.format(outname)), 'w')
    excl_stats = open(os.path.join(dbgdir, 'rej_{}_stats.txt'.format(outname)), 'w')

    peak_params = open(os.path.join(dbgdir, '{}_params.txt'.format(outname)), 'w')
    excl_params = open(os.path.join(dbgdir, 'rej_{}_params.txt'.format(outname)), 'w')

    fmt_summary = 3*'{:8.2f}'+'{:8.4f}'+6*'{:8.2f}'+'{:4.0f}'+6*'{:8.2f}'+'\n'
    fmt_stats = 3*'{:8.2f}'+'{:8.4f}'+12*'{:10.2f}'+'{:10}\n'
    fmt_params = 3*'{:8.2f}'+'{:8.4f}'+2*'{:10.2e}'+6*'{:8.3f}'+3*'{:8.2f}'+'{:6.0f}'+2*'{:6}'+6*'{:8.3f}'+'\n'

    min_sig_noise_ratio = 3

    reason = '   no/ok  '

    for i, (key, j) in enumerate(zip(keys,inds)):

        key = tuple(key)

        print('Process {} integrating peak ({} {} {} {} {} {})'.format(proc,*key))

        peak = peak_dict[key][j]

        h, k, l, m, n, p = key

        H, K, L = peak_dictionary.get_hkl(h, k, l, m, n, p)
        d = peak_dictionary.get_d(h, k, l, m, n, p)

        pk_env = os.path.join(dbgdir, '{}_{}_{}.png'.format(outname,i,j))
        ex_env = os.path.join(dbgdir, 'rej_{}_{}_{}.png'.format(outname,i,j))

        summary_list = [H, K, L, d]
        stats_list = [H, K, L, d]
        params_list = [H, K, L, d]

        runs = peak.get_run_numbers().tolist()
        banks = peak.get_bank_numbers().tolist()
        indices = peak.get_peak_indices().tolist()

        fixed = False
        if ref_dict is not None:
            ref_peaks = ref_dict.get(key)
            if ref_peaks is not None:
                if len(ref_peaks) == len(peak_dict[key]):
                    ref_peak = ref_dict[key][j]
                    fixed = True

        sat_keys, sat_Qs = peak.get_close_satellites()

        Q0 = peak.get_Q()

        delta_Q0 = np.zeros(3)
        delta = 0
        close = False

        if cluster and d > min_d_sat:
            if len(sat_Qs) > 0:
                delta_Q0 = Q0-sat_Qs[0]
                delta = np.linalg.norm(delta_Q0)
                close = True

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

        peak_envelope.clear_plots(key, d, lamda, two_theta, az_phi, n_runs)

        gon_omega = [np.min(omega), np.max(omega)]
        gon_chi = [np.min(chi), np.max(chi)]
        gon_phi = [np.min(phi), np.max(phi)]

        summary_list.extend([*lamda, *two_theta, *az_phi, n_runs, *gon_omega, *gon_chi, *gon_phi])

        peak_fit, peak_bkg_ratio, sig_noise_ratio = 0, 0, 0
        peak_fit2d, peak_bkg_ratio2d, sig_noise_ratio2d = 0, 0, 0

        fit = True
        remove = False
        reason = '   none   '

        ol = mtd['cws'].sample().getOrientedLattice()

        radius = box_fit_size[0]+box_fit_size[1]*2*np.pi/d
        binsize = radius/20

        if close:
            radius *= 2

        if not remove:

            partial_load(facility, instrument, runs, banks, indices,
                         phi, chi, omega, norm_scale, split_angle,
                         dbgdir, ipts, outname, detector_calibration, elastic, timing_offset, experiment, tmp)

            Q, Qx, Qy, Qz, data, norm = box_integrator(facility, instrument, runs, banks, indices, split_angle, Q0, delta_Q0, key,
                                                       binsize=binsize, radius=radius, exp=experiment, close=close)

            if not close:

                midpoints, normals = peak_dictionary.query_planes(Q0, 0.75*radius)

                for midpoint, normal in zip(midpoints, normals):
                    mask = normal[0]*(Qx-midpoint[0])\
                         + normal[1]*(Qy-midpoint[1])\
                         + normal[2]*(Qz-midpoint[2]) > 0
                    norm[mask] = 0.0

                mask = norm > 0

                Q = np.sqrt(Qx**2+Qy**2+Qz**2)

                Q, Qx, Qy, Qz, data, norm = Q[mask], Qx[mask], Qy[mask], Qz[mask], data[mask], norm[mask]

            rot = True if facility == 'HFIR' else False

            if close:
                bins = [11,11,31] 
            elif rot:
                bins = [11,11,11] 
            else:
                bins = [11,11,11] 

            ellip = Ellipsoid(Qx, Qy, Qz, data, norm, Q0, size=radius, rotation=rot)

            if close:
                ellip.reset_axes(delta_Q0)

            int_mask, bkg_mask = ellip.profile_mask(extend=close)
            Qp, data, norm = ellip.Qp, ellip.data, ellip.norm

            prof = Profile() if not close else LineCut(delta=delta)
            stats, params = prof.fit(Qp, data, norm, int_mask, bkg_mask, 0.99)

            peak_fit_1, peak_bkg_ratio_1, sig_noise_ratio_1 = stats

            if not close:
                a, mu, sigma = params
            else:
                a, mu, sigma = np.mean(params[0:3]), np.mean(params[3:6]), params[-1]+(np.max(params[3:6])-np.min(params[3:6]))/6
                delta = (np.max(params[3:6])-np.min(params[3:6]))/2
                if (not np.isfinite(delta)) or (not delta > 0):
                    delta = np.linalg.norm(delta_Q0)

            if np.any(prof.y_sub > 0) and np.isfinite([a,mu,sigma]).all():

                ellip.mu = mu
                ellip.sigma = sigma

                peak_envelope.plot_Q(prof.x, prof.y_sub, prof.y, prof.e, prof.y_fit, prof.y_bkg)

                stats_list.extend([peak_fit_1, peak_bkg_ratio_1, sig_noise_ratio_1])

            else:

                stats_list.extend([np.nan, np.nan, np.nan, np.nan])

            sig_noise_ratio = sig_noise_ratio_1

            int_mask, bkg_mask = ellip.projection_mask()
            dQ1, dQ2, data, norm = ellip.dQ1, ellip.dQ2, ellip.data, ellip.norm

            max_size = 0.75*ellip.size

            proj = Projection()
            stats, params = proj.fit(dQ1, dQ2, data, norm, int_mask, bkg_mask, 0.99, max_size)

            peak_fit2d_1, peak_bkg_ratio2d_1, sig_noise_ratio2d_1 = stats
            a, mu_x, mu_y, sigma_x, sigma_y, rho = params

            # ---

            if np.any(proj.z_sub > 0) and np.isfinite([a,mu_x,mu_y,sigma_x,sigma_y,rho]).all():

                ellip.mu_x, ellip.mu_y = mu_x, mu_y
                ellip.sigma_x, ellip.sigma_y, ellip.rho = sigma_x, sigma_y, rho

                x_extents = [proj.x.min(), proj.x.max()]
                y_extents = [proj.y.min(), proj.y.max()]

                mu = [mu_x, mu_y]
                sigma = [sigma_x, sigma_y]

                peak_envelope.plot_projection(proj.z_sub, proj.z, x_extents, y_extents, mu, sigma, rho, peak_fit2d)

                if np.isinf(peak_bkg_ratio2d_1) or np.isnan(peak_bkg_ratio2d_1):

                    remove = True
                    reason = '   2d-proj'

                stats_list.extend([peak_fit2d_1, peak_bkg_ratio2d_1, sig_noise_ratio2d_1])

            else:

                remove = True
                reason = '   2d-norm'

                stats_list.extend([np.nan, np.nan, np.nan])

            # ---

            peak_bkg_ratio2d = peak_bkg_ratio2d_1
            sig_noise_ratio2d = sig_noise_ratio2d_1

            ellip.size = 6*np.max([ellip.sigma_x*2,ellip.sigma_y*2,ellip.sigma])

            if peak_bkg_ratio2d > 2 and sig_noise_ratio2d > 20 and sig_noise_ratio > 20:

                b, cx, cy, cxy = proj.b, proj.cx, proj.cy, proj.cxy

                int_mask, bkg_mask = ellip.projection_mask()
                dQ1, dQ2, data, norm = ellip.dQ1, ellip.dQ2, ellip.data, ellip.norm

                proj = Projection()

                x = dQ1.copy()
                y = dQ2.copy()

                xh, yh, _, _, z_sub, e_sub, _ = proj.histogram(x, y, data, norm, int_mask, bkg_mask, 0.99)

                bkg = proj.nonlinear(xh, yh, b, cx, cy, cxy)

                z_sub -= bkg

                args, params, bounds = proj.estimate(xh, yh, z_sub, e_sub)

                a, mu_x, mu_y, sigma_1, sigma_2, theta, b, cx, cy, cxy = params

                if np.isfinite([mu_x,mu_y,sigma_1,sigma_2,theta]).all() and 3*sigma_1 > 0.02 and 3*sigma_2 > 0.02:

                    R = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta),  np.cos(theta)]])

                    cov = np.dot(R, np.dot(np.diag([sigma_1**2, sigma_2**2]), R.T))

                    sigma_x, sigma_y = np.sqrt(np.diag(cov))
                    rho = cov[0,1]/(sigma_x*sigma_y)

                    ellip.mu_x, ellip.mu_y = mu_x, mu_y
                    ellip.sigma_x, ellip.sigma_y, ellip.rho = sigma_x, sigma_y, rho

                    mu = [mu_x, mu_y]
                    sigma = [sigma_x, sigma_y]

                    peak_envelope.update_ellipse(mu, sigma, rho)

            int_mask, bkg_mask = ellip.profile_mask()
            Qp, data, norm = ellip.Qp, ellip.data, ellip.norm

            prof = Profile() if not close else LineCut(delta=delta)
            stats, params = prof.fit(Qp, data, norm, int_mask, bkg_mask, 0.99)

            peak_fit_2, peak_bkg_ratio_2, sig_noise_ratio_2 = stats

            if not close:
                a, mu, sigma = params
            else:
                a, mu, sigma = np.mean(params[0:3]), np.mean(params[3:6]), params[-1]+(np.max(params[3:6])-np.min(params[3:6]))/6
                delta = (np.max(params[3:6])-np.min(params[3:6]))/2
                if (not np.isfinite(delta)) or (not delta > 0):
                    delta = np.linalg.norm(delta_Q0)

            if np.any(prof.y_sub > 0) and np.isfinite([a,mu,sigma]).all():

                ellip.mu = mu
                ellip.sigma = sigma

                if peak_fit_1 < 1 and peak_fit_2 < 1:
                    peak_fit = np.max([peak_fit_1,peak_fit_2])
                elif peak_fit_1 > 1 and peak_fit_2 > 1:
                    peak_fit = np.min([peak_fit_1,peak_fit_2])
                else:
                    peak_fit = peak_fit_1 if np.abs(peak_fit_1-1) < np.abs(peak_fit_2-1) else peak_fit_2

                peak_bkg_ratio = np.nanmax([peak_bkg_ratio_1, peak_bkg_ratio_2])
                sig_noise_ratio = np.nanmax([sig_noise_ratio_1, sig_noise_ratio_2])

                peak_envelope.plot_extracted_Q(prof.x, prof.y_sub, prof.y, prof.e, prof.y_fit, prof.y_bkg, peak_fit)

                stats_list.extend([peak_fit_2, peak_bkg_ratio_2, sig_noise_ratio_2])

            else:

                remove = True
                reason = '   1d-prof'

                stats_list.extend([np.nan, np.nan, np.nan])

            int_mask, bkg_mask = ellip.projection_mask()
            dQ1, dQ2, data, norm = ellip.dQ1, ellip.dQ2, ellip.data, ellip.norm

            max_size = 0.75*ellip.size

            proj = Projection()
            stats, params = proj.fit(dQ1, dQ2, data, norm, int_mask, bkg_mask, 0.99, max_size)

            peak_fit2d_2, peak_bkg_ratio2d_2, sig_noise_ratio2d_2 = stats
            a, mu_x, mu_y, sigma_x, sigma_y, rho = params

            # ---

            if np.any(proj.z_sub > 0) and np.isfinite([a,mu_x,mu_y,sigma_x,sigma_y,rho]).all():

                ellip.mu_x, ellip.mu_y = mu_x, mu_y
                ellip.sigma_x, ellip.sigma_y, ellip.rho = sigma_x, sigma_y, rho

                if peak_fit2d_1 < 1 and peak_fit2d_2 < 1:
                    peak_fit2d = np.max([peak_fit2d_1,peak_fit2d_2])
                elif peak_fit2d_1 > 1 and peak_fit2d_2 > 1:
                    peak_fit2d = np.min([peak_fit2d_1,peak_fit2d_2])
                else:
                    peak_fit2d = peak_fit2d_1 if np.abs(peak_fit2d_1-1) < np.abs(peak_fit2d_2-1) else peak_fit2d_2

                x_extents = [proj.x.min(), proj.x.max()]
                y_extents = [proj.y.min(), proj.y.max()]

                mu = [mu_x, mu_y]
                sigma = [sigma_x, sigma_y]

                peak_envelope.plot_extracted_projection(proj.z_sub, proj.z, x_extents, y_extents, mu, sigma, rho, peak_fit2d)

                if np.isinf(peak_bkg_ratio2d_2) or np.isnan(peak_bkg_ratio2d_2):

                    remove = True
                    reason = '   2d-proj'

                stats_list.extend([peak_fit2d_2, peak_bkg_ratio2d_2, sig_noise_ratio2d_2])

            else:

                remove = True
                reason = '   2d-norm'

                stats_list.extend([np.nan, np.nan, np.nan])

            # ---

            peak_bkg_ratio2d = peak_bkg_ratio2d_2
            sig_noise_ratio2d = sig_noise_ratio2d_2

            ellip.size = 6*np.max([ellip.sigma_x,ellip.sigma_y,ellip.sigma])

            if peak_bkg_ratio2d > 2 and sig_noise_ratio2d > 20 and sig_noise_ratio > 20:

                b, cx, cy, cxy = proj.b, proj.cx, proj.cy, proj.cxy

                int_mask, bkg_mask = ellip.projection_mask()
                dQ1, dQ2, data, norm = ellip.dQ1, ellip.dQ2, ellip.data, ellip.norm

                proj = Projection()

                x = dQ1.copy()
                y = dQ2.copy()

                xh, yh, _, _, z_sub, e_sub, _ = proj.histogram(x, y, data, norm, int_mask, bkg_mask, 0.99)

                bkg = proj.nonlinear(xh, yh, b, cx, cy, cxy)

                z_sub -= bkg

                args, params, bounds = proj.estimate(xh, yh, z_sub, e_sub)

                a, mu_x, mu_y, sigma_1, sigma_2, theta, b, cx, cy, cxy = params

                if np.isfinite([mu_x,mu_y,sigma_1,sigma_2,theta]).all() and 3*sigma_1 > 0.02 and 3*sigma_2 > 0.02:

                    R = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta),  np.cos(theta)]])

                    cov = np.dot(R, np.dot(np.diag([sigma_1**2, sigma_2**2]), R.T))

                    sigma_x, sigma_y = np.sqrt(np.diag(cov))
                    rho = cov[0,1]/(sigma_x*sigma_y)

                    ellip.mu_x, ellip.mu_y = mu_x, mu_y
                    ellip.sigma_x, ellip.sigma_y, ellip.rho = sigma_x, sigma_y, rho

                    mu = [mu_x, mu_y]
                    sigma = [sigma_x, sigma_y]

                    peak_envelope.update_ellipse2(mu, sigma, rho)

            # ---

            Q1, W, D = ellip.ellipsoid()

            if fixed:

                if ref_peak.is_peak_integrated():

                    print('\nUsing fixed envelope\n')

                    # Q1 = ref_peak.get_Q()

                    W = ref_peak.get_W()
                    D = ref_peak.get_D()

            A = ellip.A(W, D)

            radii = 1/np.sqrt(np.diagonal(D)) 

            fit_1d = [ellip.mu, ellip.sigma]
            fit_2d = [ellip.mu_x, ellip.mu_y, ellip.sigma_x, ellip.sigma_y, ellip.rho]

            bound_str, fit_est_str = '  none', '  none'
            A, B, C0, C1, C2, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01, N = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

            Q_rot = [np.nan, np.nan, np.nan]
            Q_sigs = [np.nan, np.nan, np.nan]

            if (not np.isclose(np.abs(np.linalg.det(W)),1)) or np.isclose(radii, 0).any() or (not np.isfinite(radii).all()) or (not (radii > 0).all()) or np.isclose(1/radii**2, 0).any() or (not np.isfinite(1/radii**2).all()) or (not (1/radii**2 > 0).all()) and not remove:

                remove = True
                reason = '   3d-env '

            if close:
                if np.isclose(delta, 0) or delta <= 0 or not np.isfinite(delta):
                    remove = True
                    reason = '   3d-sat '

            if not remove:

                #Q_bin, Q_rot, Q_radii, Q_scales, signal, error, data_norm, pk_bkg, cntrs = norm_integrator_fast(runs, Q0, delta_Q0, Q1, D, W,
                #                                                                                                bins=bins, exp=experiment, close=close)

                Q_bin, Q_rot, Q_radii, Q_scales, signal, error, data_norm, pk_bkg, cntrs = norm_integrator(facility, instrument, runs, banks, indices, split_angle,
                                                                                                           Q1, delta_Q0, D, W, bins=bins, exp=experiment, close=close)

                dQ1_extents, dQ2_extents, Qp_extents = Q_bin

                fit_stats = [peak_fit, peak_bkg_ratio, sig_noise_ratio, peak_fit2d, peak_bkg_ratio2d, sig_noise_ratio2d]

                peak_envelope.plot_integration(signal, dQ1_extents, dQ2_extents, Qp_extents, Q_rot, Q_radii, Q_scales)

                Q_sigs = ellip.sig()
                dQ1, dQ2, Qp, _, _ = data_norm

                mask = (signal > 0) & (error > 0) & np.isfinite(signal/error)
                N = signal[mask].size

                if N > 100:

                    pk = peak_dictionary.peak_dict.get(key)[j]

                    peak_dictionary.integrated_result(key, Q1, D, W, fit_stats, data_norm, pk_bkg, cntrs, j)

                    I_est = pk.get_merged_intensity()
                    sig_est = pk.get_merged_intensity_error()

                    if not close:
                        peak_fit_3d = GaussianFit3D((dQ1[mask], dQ2[mask], Qp[mask]), signal[mask], error[mask], Q_rot, Q_sigs)
                    else:
                        peak_fit_3d = SatelliteGaussianFit3D((dQ1[mask], dQ2[mask], Qp[mask]), signal[mask], error[mask], Q_rot, Q_sigs, delta)

                    if not close:
                        A, B, C0, C1, C2, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01, boundary = peak_fit_3d.fit()
                    else:
                        A0, A1, A2, B, C0, C1, C2, mu0, mu1, mu2, delta, sig0, sig1, sig2, rho12, rho02, rho01, boundary = peak_fit_3d.fit()

                    fit_est_str = '   fit'

                    bound_str = ' false' if boundary else '  true'

                    fit_3d = [mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01]

                    if close:
                        fit_3d += [delta]

                    pk.add_fit(fit_1d, fit_2d, fit_3d, 0)

                    if not close:
                        A, B, C0, C1, C2 = pk.integrate()
                        fit = peak_fit_3d.model((dQ1, dQ2, Qp), A, B, C0, C1, C2, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01)
                    else:
                        A0, A1, A2, B, C0, C1, C2 = pk.integrate()
                        fit = peak_fit_3d.model((dQ1, dQ2, Qp), A0, A1, A2, B, C0, C1, C2, mu0, mu1, mu2, delta, sig0, sig1, sig2, rho12, rho02, rho01)

                    I_fit = pk.get_fitted_intensity()
                    sig_fit = pk.get_fitted_intensity_error()

                    chi_sq = np.sum((signal[mask]-fit[mask])**2/error[mask]**2)/(N-11)
                    
                    if not np.isfinite(chi_sq):
                        remove = True
                        reason = '   3d-fit '

                    I = [I_est, I_fit]
                    err = [sig_est, sig_fit]

                    peak_envelope.plot_fitting(fit, I, err, chi_sq)

                    # ---

                    vals, vecs = peak_fit_3d.eigendecomposition()

                    radii = 3*np.sqrt(vals)

                    D = np.diag(1/radii**2)

                    if close:
                        delta_Q0 = np.dot(W, [0,0,delta])

                    Q1 = np.dot(W, [mu0, mu1, mu2])

                    W = np.dot(W, vecs)

                    Q_sigs = np.sqrt(vals)

                    if (not np.isclose(np.abs(np.linalg.det(W)),1)) or np.isclose(radii, 0).any() or (not np.isfinite(radii).all()) or (not (radii > 0).all()) or np.isclose(1/radii**2, 0).any() or (not np.isfinite(1/radii**2).all()) or (not (1/radii**2 > 0).all()):

                        remove = True
                        reason = '   3d-env '

                    if close:
                        if np.isclose(delta, 0) or delta <= 0 or not np.isfinite(delta):
                            remove = True
                            reason = '   3d-sat '

                    if not remove:

                        Q_bin, Q_rot, Q_radii, Q_scales, signal, error, data_norm, pk_bkg, cntrs = norm_integrator(facility, instrument, runs, banks, indices, split_angle,
                                                                                                                   Q1, delta_Q0, D, W, bins=[11,11,11], exp=experiment, close=False)

                        dQ1_extents, dQ2_extents, Qp_extents = Q_bin

                        fit_stats = [peak_fit, peak_bkg_ratio, sig_noise_ratio, peak_fit2d, peak_bkg_ratio2d, sig_noise_ratio2d]

                        peak_envelope.plot_extracted_integration(signal, dQ1_extents, dQ2_extents, Qp_extents, Q_rot, Q_radii, Q_scales)

                        dQ1, dQ2, Qp, _, _ = data_norm

                        mask = (signal > 0) & (error > 0) & np.isfinite(signal/error)
                        N = signal[mask].size

                        if N > 100:

                            pk = peak_dictionary.peak_dict.get(key)[j]

                            peak_dictionary.integrated_result(key, Q1, D, W, fit_stats, data_norm, pk_bkg, cntrs, j)

                            I_est = pk.get_merged_intensity()
                            sig_est = pk.get_merged_intensity_error()

                            peak_fit_3d = GaussianFit3D((dQ1[mask], dQ2[mask], Qp[mask]), signal[mask], error[mask], Q_rot, Q_sigs)

                            A, B, C0, C1, C2, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01, boundary = peak_fit_3d.fit()

                            fit_est_str = '   fit'

                            bound_str = ' false' if boundary else '  true'

                            fit_3d = [mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01]

                            pk.add_fit(fit_1d, fit_2d, fit_3d, 0)

                            A, B, C0, C1, C2 = pk.integrate()
                            fit = peak_fit_3d.model((dQ1, dQ2, Qp), A, B, C0, C1, C2, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01)

                            I_fit = pk.get_fitted_intensity()
                            sig_fit = pk.get_fitted_intensity_error()

                            chi_sq = np.sum((signal[mask]-fit[mask])**2/error[mask]**2)/(N-11)

                            if not np.isfinite(chi_sq):
                                remove = True
                                reason = '   3d-fit '

                            I = [I_est, I_fit]
                            err = [sig_est, sig_fit]

                            peak_envelope.plot_extracted_fitting(fit, I, err, chi_sq)

                            # ---

                            if not close:
                                if boundary:
                                    remove = True
                                    reason = '   3d-bndr'
                                elif I_est <= sig_est or np.isclose(I_est, 0):
                                    remove = True
                                    reason = '   3d-est '
                                elif np.isclose(I_fit, 0):
                                    remove = True
                                    reason = '   3d-fitn'
                                elif I_fit <= sig_fit:
                                    remove = True
                                    reason = '   3d-fits'

                        else:

                            remove = True
                            reason = '   3d-cnts'

                else:

                    remove = True
                    reason = '   3d-cnts'

                if remove:

                    chi_sq = np.inf
                    fit_3d = [0, 0, 0, 0, 0, 0, 0, 0, 0]

                peak_dictionary.fitted_result(key, fit_1d, fit_2d, fit_3d, chi_sq, j)

            else:

                remove = True
                reason = '   3d-env '

            params_list.extend([A, B, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01, N, bound_str, fit_est_str, *Q_rot, *Q_sigs])

            stats_list.append(reason)

            if remove:

                peak_envelope.write_figure(ex_env)

                excl_summary.write(fmt_summary.format(*summary_list))
                excl_stats.write(fmt_stats.format(*stats_list))
                excl_params.write(fmt_params.format(*params_list))

            else:

                peak_envelope.write_figure(pk_env)

                peak_summary.write(fmt_summary.format(*summary_list))
                peak_stats.write(fmt_stats.format(*stats_list))
                peak_params.write(fmt_params.format(*params_list))

                if close:

                    for s, (sat_key, sat_Q) in enumerate(zip(sat_keys, sat_Qs)):

                        remove = False
                        reason = '   none   '

                        h, k, l, m, n, p = sat_key

                        h, k, l, m, n, p = int(h), int(k), int(l), int(m), int(n), int(p)

                        sat_key = (h,k,l,m,n,p)

                        peak_dictionary.clone_peak(pk, sat_key, sat_Q)

                        n_sat = len(peak_dictionary.peak_dict.get(sat_key))

                        sat_pk = peak_dictionary.peak_dict.get(sat_key)[n_sat-1]

                        H, K, L = peak_dictionary.get_hkl(h, k, l, m, n, p)
                        d = peak_dictionary.get_d(h, k, l, m, n, p)

                        pk_env = os.path.join(dbgdir, '{}_{}_{}_{}.png'.format(outname,i,j,s))
                        ex_env = os.path.join(dbgdir, 'rej_{}_{}_{}_{}.png'.format(outname,i,j,s))

                        peak_envelope.update_plots(sat_key, d)

                        summary_list[0:4] = [H, K, L, d]
                        stats_list[0:4] = [H, K, L, d]
                        params_list[0:4] = [H, K, L, d]

                        if s == 0:
                            Q2 = Q1-delta_Q0
                        else:
                            Q2 = Q1+delta_Q0

                        Q_bin, Q_rot, Q_radii, Q_scales, signal, error, data_norm, pk_bkg, cntrs = norm_integrator(facility, instrument, runs, banks, indices, split_angle,
                                                                                                                   Q2, np.zeros(3), D, W, bins=[11,11,11], exp=experiment, close=False)

                        dQ1_extents, dQ2_extents, Qp_extents = Q_bin

                        fit_stats = [peak_fit, peak_bkg_ratio, sig_noise_ratio, peak_fit2d, peak_bkg_ratio2d, sig_noise_ratio2d]

                        peak_envelope.plot_extracted_integration(signal, dQ1_extents, dQ2_extents, Qp_extents, Q_rot, Q_radii, Q_scales)

                        dQ1, dQ2, Qp, _, _ = data_norm

                        mask = (signal > 0) & (error > 0) & np.isfinite(signal/error)
                        N = signal[mask].size

                        if N > 100:

                            peak_dictionary.integrated_result(sat_key, Q2, D, W, fit_stats, data_norm, pk_bkg, cntrs, n_sat-1)

                            I_est = sat_pk.get_merged_intensity()
                            sig_est = sat_pk.get_merged_intensity_error()

                            peak_fit_3d = GaussianFit3D((dQ1[mask], dQ2[mask], Qp[mask]), signal[mask], error[mask], Q_rot, Q_sigs)

                            A, B, C0, C1, C2, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01, boundary = peak_fit_3d.fit()

                            fit_est_str = '   fit'

                            bound_str = ' false' if boundary else '  true'

                            fit_3d = [mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01]

                            sat_pk.add_fit(fit_1d, fit_2d, fit_3d, 0)

                            A, B, C0, C1, C2 = sat_pk.integrate()
                            fit = peak_fit_3d.model((dQ1, dQ2, Qp), A, B, C0, C1, C2, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01)

                            I_fit = sat_pk.get_fitted_intensity()
                            sig_fit = sat_pk.get_fitted_intensity_error()

                            chi_sq = np.sum((signal[mask]-fit[mask])**2/error[mask]**2)/(N-11)

                            I = [I_est, I_fit]
                            err = [sig_est, sig_fit]

                            peak_envelope.plot_extracted_fitting(fit, I, err, chi_sq)

                            if boundary:
                                remove = True
                                reason = '   3d-bndr'
                            elif I_est <= sig_est or np.isclose(I_est, 0):
                                remove = True
                                reason = '   3d-est '
                            elif np.isclose(I_fit, 0):
                                remove = True
                                reason = '   3d-fitn'
                            elif I_fit <= sig_fit:
                                remove = True
                                reason = '   3d-fits'

                        else:

                            remove = True
                            reason = '   3d-cnts'

                        if remove:

                            peak_envelope.write_figure(ex_env)

                            excl_summary.write(fmt_summary.format(*summary_list))
                            excl_stats.write(fmt_stats.format(*stats_list))
                            excl_params.write(fmt_params.format(*params_list))

                        else:

                            peak_envelope.write_figure(pk_env)

                            peak_summary.write(fmt_summary.format(*summary_list))
                            peak_stats.write(fmt_stats.format(*stats_list))
                            peak_params.write(fmt_params.format(*params_list))

                        if remove:

                            chi_sq = np.inf
                            fit_3d = [0, 0, 0, 0, 0, 0, 0, 0, 0]

                        peak_dictionary.fitted_result(sat_key, fit_1d, fit_2d, fit_3d, chi_sq, n_sat-1)

            runs_banks, run_keys, bank_keys = partial_cleanup(runs, banks, indices, facility, instrument, split_angle,
                                                              runs_banks, run_keys, bank_keys, bank_group, key, exp=experiment)

        if i % 15 == 0:

            peak_summary.flush()
            excl_summary.flush()

            peak_stats.flush()
            excl_stats.flush()

            peak_params.flush()
            excl_params.flush()

            peak_dictionary.save_hkl(os.path.join(dbgdir, '{}.hkl'.format(outname)), min_signal_noise_ratio=min_sig_noise_ratio, cross_terms=cross_terms)
            peak_dictionary.save(os.path.join(dbgdir, '{}.pkl'.format(outname)))

    peak_dictionary.save_hkl(os.path.join(dbgdir, '{}.hkl'.format(outname)))
    peak_dictionary.save(os.path.join(dbgdir, '{}.pkl'.format(outname)))

    peak_summary.close()
    excl_summary.close()

    peak_stats.close()
    excl_stats.close()

    peak_params.close()
    excl_params.close()

    if mtd.doesExist('sa'):
        DeleteWorkspace('sa')
    if mtd.doesExist('flux'):
        DeleteWorkspace('flux')
    if mtd.doesExist('van'):
        DeleteWorkspace('van')

    with open(os.path.join(dbgdir, '{}.pdf'.format(outname)), 'wb') as f:
        merger = []
        for i, (key, j) in enumerate(zip(keys,inds)):
            env = os.path.join(dbgdir, '{}_{}_{}.png'.format(outname,i,j))
            if os.path.exists(env):
                with Image.open(env) as im:
                    fig = env.replace('png','jpg')
                    im = im.convert('RGB')
                    im.save(fig, 'JPEG')
                merger.append(fig)
            for s in range(2):
                env = os.path.join(dbgdir, '{}_{}_{}_{}.png'.format(outname,i,j,s))
                if os.path.exists(env):
                    with Image.open(env) as im:
                        fig = env.replace('png','jpg')
                        im = im.convert('RGB')
                        im.save(fig, 'JPEG')
                    merger.append(fig)
        if len(merger) > 0:
            f.write(img2pdf.convert(merger))
        else:
            os.remove(os.path.join(dbgdir, '{}.pdf'.format(outname)))

    with open(os.path.join(dbgdir, 'rej_{}.pdf'.format(outname)), 'wb') as f:
        merger = []
        for i, (key, j) in enumerate(zip(keys,inds)):
            env = os.path.join(dbgdir, 'rej_{}_{}_{}.png'.format(outname,i,j))
            if os.path.exists(env):
                with Image.open(env) as im:
                    fig = env.replace('png','jpg')
                    im = im.convert('RGB')
                    im.save(fig, 'JPEG')
                merger.append(fig)
            for s in range(2):
                env = os.path.join(dbgdir, 'rej_{}_{}_{}_{}.png'.format(outname,i,j,s))
                if os.path.exists(env):
                    with Image.open(env) as im:
                        fig = env.replace('png','jpg')
                        im = im.convert('RGB')
                        im.save(fig, 'JPEG')
                    merger.append(fig)
        if len(merger) > 0:
            f.write(img2pdf.convert(merger))
        else:
            os.remove(os.path.join(dbgdir, 'rej_{}.pdf'.format(outname)))

    for i, (key, j) in enumerate(zip(keys,inds)):
        env = os.path.join(dbgdir, '{}_{}_{}.png'.format(outname,i,j))
        if os.path.exists(env):
            os.remove(env)
        env = os.path.join(dbgdir, 'rej_{}_{}_{}.png'.format(outname,i,j))
        if os.path.exists(env):
            os.remove(env)
        env = os.path.join(dbgdir, '{}_{}_{}.jpg'.format(outname,i,j))
        if os.path.exists(env):
            os.remove(env)
        env = os.path.join(dbgdir, 'rej_{}_{}_{}.jpg'.format(outname,i,j))
        if os.path.exists(env):
            os.remove(env)
        for s in range(2):
            env = os.path.join(dbgdir, '{}_{}_{}_{}.png'.format(outname,i,j,s))
            if os.path.exists(env):
                os.remove(env)
            env = os.path.join(dbgdir, 'rej_{}_{}_{}_{}.png'.format(outname,i,j,s))
            if os.path.exists(env):
                os.remove(env)            
            env = os.path.join(dbgdir, '{}_{}_{}_{}.jpg'.format(outname,i,j,s))
            if os.path.exists(env):
                os.remove(env)
            env = os.path.join(dbgdir, 'rej_{}_{}_{}_{}.jpg'.format(outname,i,j,s))
            if os.path.exists(env):
                os.remove(env)