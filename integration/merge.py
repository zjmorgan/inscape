from mantid.simpleapi import *

from mantid.kernel import V3D, FloatTimeSeriesProperty
from mantid.geometry import Goniometer

import os
import re
import psutil
import itertools

import numpy as np

import peak
from peak import PeakEnvelope, PeakDictionary

def box_integrator(facility, instrument, runs, banks, indices, norm_scale, Q0, binsize=0.001, radius=0.15, exp=None):

    for j, (r, b, i) in enumerate(zip(runs, banks, indices)):

        if exp is None:
            ows = '{}_{}_{}'.format(instrument,r,b)
        else:
            ows = '{}_{}_{}_{}'.format(instrument,exp,r,i)

        omd = ows+'_md'

        if j == 0:

            Qr = np.array([radius,radius,radius])

            nQ = np.round(2*Qr/binsize).astype(int)+1

            Qmin, Qmax = Q0-Qr, Q0+Qr

        if facility == 'SNS':

            BinMD(InputWorkspace=omd,
                  AlignedDim0='Q_sample_x,{},{},{}'.format(Qmin[0],Qmax[0],nQ[0]),
                  AlignedDim1='Q_sample_y,{},{},{}'.format(Qmin[1],Qmax[1],nQ[1]),
                  AlignedDim2='Q_sample_z,{},{},{}'.format(Qmin[2],Qmax[2],nQ[2]),
                  OutputWorkspace='__tmp')

            __tmp = mtd['__tmp']#/norm_scale[r]

        else:

            lamda = float(mtd[ows].getExperimentInfo(0).run().getProperty('wavelength').value)

            ConvertWANDSCDtoQ(InputWorkspace=ows,
                              OutputWorkspace='__tmp',
                              Wavelength=lamda,
                              NormaliseBy='None',
                              Frame='Q_sample',
                              BinningDim0='{},{},{}'.format(Qmin[0],Qmax[0],nQ[0]),
                              BinningDim1='{},{},{}'.format(Qmin[1],Qmax[1],nQ[1]),
                              BinningDim2='{},{},{}'.format(Qmin[2],Qmax[2],nQ[2]))

            __tmp = mtd['__tmp']#/norm_scale[(r,i)]

        if j == 0:
            CloneMDWorkspace(InputWorkspace=__tmp, OutputWorkspace='box')
        else:
            PlusMD(LHSWorkspace='box', RHSWorkspace=__tmp, OutputWorkspace='box')

    SetMDFrame(InputWorkspace='box', MDFrame='QSample', Axes=[0,1,2])
    mtd['box'].clearOriginalWorkspaces()

    Qxaxis = mtd['box'].getXDimension()
    Qyaxis = mtd['box'].getYDimension()
    Qzaxis = mtd['box'].getZDimension()

    Qx, Qy, Qz = np.meshgrid(np.linspace(Qxaxis.getMinimum(), Qxaxis.getMaximum(), Qxaxis.getNBins()),
                             np.linspace(Qyaxis.getMinimum(), Qyaxis.getMaximum(), Qyaxis.getNBins()),
                             np.linspace(Qzaxis.getMinimum(), Qzaxis.getMaximum(), Qzaxis.getNBins()), indexing='ij', copy=False)

    mask = mtd['box'].getSignalArray() > 0

    Q = np.sqrt(Qx[mask]**2+Qy[mask]**2+Qz[mask]**2)

    weights = mtd['box'].getSignalArray()[mask]

    return Q, Qx[mask], Qy[mask], Qz[mask], weights

def Q_profile(peak_envelope, key, Q, Qx, Qy, Qz, weights, Q0, n, radius=0.15, bins=30):

    Qmod = np.linalg.norm(Q0)

    mask = (np.abs(Q-Qmod) < radius)

    Qp = Qx*n[0]+Qy*n[1]+Qz*n[2]

    x = Qp[~mask].flatten()
    y = weights[~mask].flatten()

    #A = (np.array([x*0+1, x])*weights[~mask]**2).T
    #B = y.flatten()*weights[~mask]**2

    A = (np.array([x*0+1, x])).T
    B = y.flatten()

    if np.size(y) <= 3 or np.any(np.isnan(x)):
        return Qmod, np.nan, np.nan, np.nan, np.nan, np.nan

    coeff, r, rank, s = np.linalg.lstsq(A, B)

    bkg_weights = (coeff[0]+coeff[1]*Qp)

    data_weights = weights-bkg_weights

    data_weights[data_weights < 0] = 0

    if np.sum(data_weights > 0) > bins:

        bin_counts, bin_edges = np.histogram(Qp[mask], bins=bins, weights=data_weights[mask])

        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])

        data = bin_counts.copy()

        pos_data = data > 0

        if (pos_data.sum() > 0):
            data -= data[pos_data].min()
        else:
            return Qmod, np.nan, np.nan, np.nan, np.nan, np.nan

        data[data < 0] = 0

        total, _ = np.histogram(Qp[mask], bins=bin_edges, weights=weights[mask])

        #arg_center = np.argmax(bin_counts) # this works for no nearby contamination
        #center = bin_centers[arg_center]

        if ((data > 0).sum() > 2):
            center = np.average(bin_centers, weights=data)
        else:
            return Qmod, np.nan, np.nan, np.nan, np.nan, np.nan
            
        #if (data > 0).any() and data.sum() > 0:
        #    center = np.average(bin_centers, weights=data)
        #else:
        #    print('Q-profile failed: mask all zeros')
        #    return Qmod, np.nan, np.nan, np.nan, min_bkg_count, np.nan

        min_data, max_data = np.min(bin_counts), np.max(bin_counts)

        factor = np.exp(-4*(2*(bin_centers-center))**2/(bin_centers.max()-bin_centers.min())**2)
        decay_weights = data**2*factor

        if (decay_weights > 0).any() and decay_weights.sum() > 0:
            variance = np.average((bin_centers-center)**2, weights=decay_weights)
            mask = (np.abs(bin_centers-center) < 4*np.sqrt(variance))

            if (data[mask] > 0).any() and data[mask].sum() > 0:
                center = np.average(bin_centers[mask], weights=data[mask])
                variance = np.average((bin_centers[mask]-center)**2, weights=data[mask])
            else:
                print('Q-profile failed: mask all zeros')
                return Qmod, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            print('Q-profile failed: decay_weights failed')
            return Qmod, np.nan, np.nan, np.nan, np.nan, np.nan

        sigma = np.sqrt(total)

        nonzero = sigma > 0
        if (sigma[nonzero].size//3 > 0):
            indices = np.argsort(sigma[nonzero])[0:sigma[nonzero].size//3]
            med = np.median(sigma[nonzero][indices])
            sigma[~nonzero] = med
        else:
            print('Q-profile failed: cannot find medium sigma')
            return Qmod, np.nan, np.nan, np.nan, np.nan, np.nan

        expected_data = norm(bin_centers, variance, center)*data.max()

        chi_sq = np.sum((data-expected_data)**2/sigma**2)/(data.size-1)

        bkg_ratio = np.std(bin_counts)/np.median(sigma)

        interp_bin_centers = np.linspace(bin_centers.min(),bin_centers.max(),200)

        peak_envelope.plot_Q(key, bin_centers, data, total,
                             1.96*sigma, interp_bin_centers,
                             norm(interp_bin_centers, variance, center)*data.max())

        sig_noise_ratio = np.sum(data)/np.sqrt(np.sum(sigma**2))

        peak_total_data_ratio = total.max()/data.max()

        return center, variance, chi_sq, bkg_ratio, sig_noise_ratio, peak_total_data_ratio

    else:

        return Qmod, np.nan, np.nan, np.nan, np.nan, np.nan

def extracted_Q_profile(peak_envelope, key, Q, Qx, Qy, Qz, weights,
                        Q0, n, u, v, center, variance, center2d, covariance2d, bins=30):

    Qmod = np.linalg.norm(Q0)

    Qu = u[0]*(Qx-Q0[0])+u[1]*(Qy-Q0[1])+u[2]*(Qz-Q0[2])
    Qv = v[0]*(Qx-Q0[0])+v[1]*(Qy-Q0[1])+v[2]*(Qz-Q0[2])

    Qp = Qx*n[0]+Qy*n[1]+Qz*n[2]

    u_center, v_center = center2d

    eigenvalues, eigenvectors = np.linalg.eig(covariance2d)

    radii = 4*np.sqrt(eigenvalues)

    D = np.diag(1/radii**2)
    W = eigenvectors.copy()

    A = np.dot(np.dot(W,D),W.T)

    mask = (A[0,0]*(Qu-u_center)+A[0,1]*(Qv-v_center))*(Qu-u_center)+\
         + (A[1,0]*(Qu-u_center)+A[1,1]*(Qv-v_center))*(Qv-v_center) <= 1 & (np.abs(Qp-center) < 4*np.sqrt(variance))

    Qp = Qx*n[0]+Qy*n[1]+Qz*n[2]

    x = Qp[~mask].flatten()
    y = weights[~mask].flatten()

    A = (np.array([x*0+1, x])).T
    B = y.flatten()

    if np.size(y) <= 3 or np.any(np.isnan(x)):
        return Qmod, np.nan, np.nan, np.nan, np.nan, np.nan

    coeff, r, rank, s = np.linalg.lstsq(A, B)

    bkg_weights = (coeff[0]+coeff[1]*Qp)

    data_weights = weights-bkg_weights

    data_weights[data_weights < 0] = 0

    if np.sum(data_weights > 0) > bins:

        bin_counts, bin_edges = np.histogram(Qp[mask], bins=bins, weights=data_weights[mask])

        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])

        data = bin_counts.copy()

        pos_data = data > 0

        if (pos_data.sum() > 0):
            data -= data[pos_data].min()
        else:
            return Qmod, np.nan, np.nan, np.nan, np.nan, np.nan

        data[data < 0] = 0

        total, _ = np.histogram(Qp[mask], bins=bin_edges, weights=weights[mask])

        if (data > 0).any():
            center = np.average(bin_centers, weights=data)
            variance = np.average((bin_centers-center)**2, weights=data)
        else:
            return Qmod, np.nan, np.nan, np.nan, np.nan, np.nan

        sigma = np.sqrt(total)

        #sigma /= data.max()
        #data /= data.max()

        expected_data = norm(bin_centers, variance, center)

        interp_bin_centers = np.linspace(bin_centers.min(),bin_centers.max(),200)

        peak_envelope.plot_extracted_Q(key, bin_centers, data, total,
                                       1.96*sigma, interp_bin_centers,
                                       norm(interp_bin_centers, variance, center)*data.max())

        nonzero = sigma > 0
        if (sigma[nonzero].size//3 > 0):
            indices = np.argsort(sigma[nonzero])[0:sigma[nonzero].size//3]
            med = np.median(sigma[nonzero][indices])
            sigma[~nonzero] = med
        else:
            print('Q-profile failed: cannot find medium sigma')
            return Qmod, np.nan, np.nan, np.nan, np.nan, np.nan

        expected_data = norm(bin_centers, variance, center)*data.max()

        chi_sq = np.sum((data-expected_data)**2/sigma**2)/(data.size-1)

        bkg_ratio = np.std(bin_counts)/np.median(sigma)

        sig_noise_ratio = np.sum(data)/np.sqrt(np.sum(sigma**2))

        peak_total_data_ratio = total.max()/data.max()

        return center, variance, chi_sq, bkg_ratio, sig_noise_ratio, peak_total_data_ratio

    else:

        return Qmod, np.nan, np.nan, np.nan, np.nan, np.nan

def norm(x, var, mu):

    return np.exp(-0.5*(x-mu)**2/var) #1/np.sqrt(2*np.pi*var)

def projection_axes(n):

    n_ind = np.argmin(np.abs(n))

    u = np.zeros(3)
    u[n_ind] = 1

    u = np.cross(n, u)
    u /= np.linalg.norm(u)

    v = np.cross(n, u)
    v *= np.sign(np.dot(np.cross(u,n),v))

    return u, v

def projected_profile(peak_envelope, key, Q, Qx, Qy, Qz, weights,
                      Q0, n, u, v, center, variance, radius=0.15, bins=16, bins2d=50):

    # Estimate 1d variance
    
    Qmod = np.linalg.norm(Q0)

    Qu = u[0]*(Qx-Q0[0])+u[1]*(Qy-Q0[1])+u[2]*(Qz-Q0[2])
    Qv = v[0]*(Qx-Q0[0])+v[1]*(Qy-Q0[1])+v[2]*(Qz-Q0[2])

    width = 4*np.sqrt(variance)
    
    Qp = Qx*n[0]+Qy*n[1]+Qz*n[2]

    if np.sum(weights > 0) > 0:

        mask = (np.abs(Qp-center) < width)\
             & ((Qx-Q0[0])**2+(Qy-Q0[1])**2+(Qz-Q0[2])**2 < np.max([width,radius])**2)

        x = Qu[~mask].flatten()
        y = Qv[~mask].flatten()
        z = weights[~mask].flatten()

        A = (np.array([x*0+1, x, y])*weights[~mask]**2).T
        B = z.flatten()*weights[~mask]**2

        if np.size(B) < 4 or np.any(np.isnan(A)):
            return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan
            
        coeff, r, rank, s = np.linalg.lstsq(A, B)

        bkg_weights = (coeff[0]+coeff[1]*Qu+coeff[2]*Qv)

        data_weights = weights-bkg_weights

        data_weights[data_weights < 0] = 0

        u_bin_counts, u_bin_edges = np.histogram(Qu[mask], bins, weights=data_weights[mask])
        v_bin_counts, v_bin_edges = np.histogram(Qv[mask], bins, weights=data_weights[mask])

        u_bin_centers = 0.5*(u_bin_edges[1:]+u_bin_edges[:-1])
        v_bin_centers = 0.5*(v_bin_edges[1:]+v_bin_edges[:-1])

        u_data = u_bin_counts.copy()
        v_data = v_bin_counts.copy()

        u_data[:-1] += u_bin_counts[1:]
        v_data[:-1] += v_bin_counts[1:]

        u_data[1:] += u_bin_counts[:-1]
        v_data[1:] += v_bin_counts[:-1]

        u_data /= 3
        v_data /= 3

        pos_u_data = u_data > 0
        pos_v_data = v_data > 0

        if (pos_u_data.sum() > 0):
            u_data -= u_data[pos_u_data].min()
        else:
            return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan

        if (pos_v_data.sum() > 0):
            v_data -= v_data[pos_v_data].min()
        else:
            return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan

        u_data[u_data < 0] = 0
        v_data[v_data < 0] = 0

        if (u_data > 0).any() and (v_data > 0).any():
            u_center = np.average(u_bin_centers, weights=u_data**2)
            v_center = np.average(v_bin_centers, weights=v_data**2)

            u_variance = np.average((u_bin_centers-u_center)**2, weights=u_data**2)
            v_variance = np.average((v_bin_centers-v_center)**2, weights=v_data**2)
        else:
            print('First pass failure 2d covariance')
            return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan

        # Correct 1d variance

        mask = (np.abs(Qu-u_center) < 6*np.sqrt(u_variance))\
             & (np.abs(Qv-v_center) < 6*np.sqrt(v_variance))

        x = Qu[~mask].flatten()
        y = Qv[~mask].flatten()
        z = weights[~mask].flatten()

        A = (np.array([x*0+1, x, y])*weights[~mask]**2).T
        B = z.flatten()*weights[~mask]**2

        if np.size(B) < 4 or np.any(np.isnan(A)):
            return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan

        coeff, r, rank, s = np.linalg.lstsq(A, B)

        bkg_weights = (coeff[0]+coeff[1]*Qu+coeff[2]*Qv)

        data_weights = weights-bkg_weights

        data_weights[data_weights < 0] = 0

        u_bin_counts, u_bin_edges = np.histogram(Qu[mask], bins, weights=data_weights[mask])
        v_bin_counts, v_bin_edges = np.histogram(Qv[mask], bins, weights=data_weights[mask])

        u_bin_centers = 0.5*(u_bin_edges[1:]+u_bin_edges[:-1])
        v_bin_centers = 0.5*(v_bin_edges[1:]+v_bin_edges[:-1])

        u_data = u_bin_counts.copy()
        v_data = v_bin_counts.copy()

        u_data[:-1] += u_bin_counts[1:]
        v_data[:-1] += v_bin_counts[1:]

        u_data[1:] += u_bin_counts[:-1]
        v_data[1:] += v_bin_counts[:-1]

        u_data /= 3
        v_data /= 3

        pos_u_data = u_data > 0
        pos_v_data = v_data > 0

        if (pos_u_data.sum() > 0):
            u_data -= u_data[pos_u_data].min()
        else:
            return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan

        if (pos_v_data.sum() > 0):
            v_data -= v_data[pos_v_data].min()
        else:
            return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan

        u_data[u_data < 0] = 0
        v_data[v_data < 0] = 0

        if (u_data > 0).any() and (v_data > 0).any():
            u_center = np.average(u_bin_centers, weights=u_data**2)
            v_center = np.average(v_bin_centers, weights=v_data**2)

            u_variance = np.average((u_bin_centers-u_center)**2, weights=u_data**2)
            v_variance = np.average((v_bin_centers-v_center)**2, weights=v_data**2)
        else:
            print('Second pass failure for 2d covariance')
            return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan

        # calculate 2d covariance

        u_width = 4*np.sqrt(u_variance)
        v_width = 4*np.sqrt(v_variance)

        mask = (np.abs(Qu-u_center) < u_width)\
             & (np.abs(Qv-v_center) < v_width)

        x = Qu[~mask].flatten()
        y = Qv[~mask].flatten()
        z = weights[~mask].flatten()

        A = (np.array([x*0+1, x, y])*weights[~mask]**2).T
        B = z.flatten()*weights[~mask]**2

        if np.size(B) < 4 or np.any(np.isnan(A)):
            return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan
            
        coeff, r, rank, s = np.linalg.lstsq(A, B)

        bkg_weights = (coeff[0]+coeff[1]*Qu+coeff[2]*Qv)

        data_weights = weights-bkg_weights

        data_weights[data_weights < 0] = 0

        range2d = [[u_center-u_width,u_center+u_width],
                   [v_center-v_width,v_center+v_width]]
             
        if (np.diff(range2d, axis=1) < 0.0001).any() or data_weights[mask].size < 4:
            return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan

        uv_bin_counts, u_bin_edges, v_bin_edges = np.histogram2d(Qu[mask], Qv[mask], bins=[bins2d,bins2d+1],
                                                                 range=range2d, weights=data_weights[mask])

        uv_bin_counts = uv_bin_counts.T

        uv_data = uv_bin_counts.copy()

        uv_data += np.roll(uv_bin_counts,3,axis=0)
        uv_data += np.roll(uv_bin_counts,-3,axis=0)

        uv_data += np.roll(uv_bin_counts,3,axis=1)
        uv_data += np.roll(uv_bin_counts,-3,axis=1)

        uv_data /= 5

        u_bin_centers_grid, v_bin_centers_grid = np.meshgrid(0.5*(u_bin_edges[1:]+u_bin_edges[:-1]),
                                                             0.5*(v_bin_edges[1:]+v_bin_edges[:-1]))

        pos_uv_data = uv_data > 0

        if (pos_uv_data.sum() > 0):
            uv_data -= uv_data[pos_uv_data].min()
        else:
            return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan

        uv_data[uv_data < 0] = 0

        if (uv_data > 0).any():
            u_center = np.average(u_bin_centers_grid, weights=uv_data**2)
            v_center = np.average(v_bin_centers_grid, weights=uv_data**2)

            u_variance = np.average((u_bin_centers_grid-u_center)**2, weights=uv_data**2)
            v_variance = np.average((v_bin_centers_grid-v_center)**2, weights=uv_data**2)

            uv_covariance = np.average((u_bin_centers_grid-u_center)\
                                      *(v_bin_centers_grid-v_center), weights=uv_data**2)
        else:
            print('Not enough data for first pass 2d covariance calculation')
            return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan

        center2d = np.array([u_center,v_center])
        covariance2d = np.array([[u_variance,uv_covariance],
                                 [uv_covariance,v_variance]])

        # ---

        u_width = 4*np.sqrt(u_variance)
        v_width = 4*np.sqrt(v_variance)

        eigenvalues, eigenvectors = np.linalg.eig(covariance2d)

        radii = 4.5*np.sqrt(eigenvalues)

        D = np.diag(1/radii**2)
        W = eigenvectors.copy()

        A = np.dot(np.dot(W,D),W.T)

        mask = (A[0,0]*(Qu-u_center)+A[0,1]*(Qv-v_center))*(Qu-u_center)+\
             + (A[1,0]*(Qu-u_center)+A[1,1]*(Qv-v_center))*(Qv-v_center) <= 1 & (np.abs(Qp-center) < width)

        radii = 4*np.sqrt(eigenvalues)

        D = np.diag(1/radii**2)
        W = eigenvectors.copy()

        A = np.dot(np.dot(W,D),W.T)

        veil = (A[0,0]*(Qu[mask]-u_center)+A[0,1]*(Qv[mask]-v_center))*(Qu[mask]-u_center)+\
             + (A[1,0]*(Qu[mask]-u_center)+A[1,1]*(Qv[mask]-v_center))*(Qv[mask]-v_center) <= 1

        x = Qu[mask][~veil].flatten()
        y = Qv[mask][~veil].flatten()
        z = weights[mask][~veil].flatten()

        sort = np.argsort(z)

        x = x[sort][0:z.size*99//100]
        y = y[sort][0:z.size*99//100]
        z = z[sort][0:z.size*99//100]

        A = (np.array([x*0+1, x, y])*z.flatten()**2).T
        B = z.flatten()*z.flatten()**2

        if np.size(B) < 4 or np.any(np.isnan(A)):
            return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan
            
        coeff, r, rank, s = np.linalg.lstsq(A, B)

        bkg_weights = (coeff[0]+coeff[1]*Qu+coeff[2]*Qv)

        data_weights = weights-bkg_weights

        data_weights[data_weights < 0] = 0

        range2d = [[u_center-u_width,u_center+u_width],
                   [v_center-v_width,v_center+v_width]]

        uv_bin_counts, u_bin_edges, v_bin_edges = np.histogram2d(Qu[mask][veil], Qv[mask][veil], bins=[bins2d,bins2d+1],
                                                                 range=range2d, weights=data_weights[mask][veil])

        uv_bin_counts = uv_bin_counts.T

        uv_data = uv_bin_counts.copy()

        uv_data += np.roll(uv_bin_counts,1,axis=0)
        uv_data += np.roll(uv_bin_counts,-1,axis=0)

        uv_data += np.roll(uv_bin_counts,1,axis=1)
        uv_data += np.roll(uv_bin_counts,-1,axis=1)

        uv_data /= 5

        u_bin_centers_grid, v_bin_centers_grid = np.meshgrid(0.5*(u_bin_edges[1:]+u_bin_edges[:-1]),
                                                             0.5*(v_bin_edges[1:]+v_bin_edges[:-1]))

        uv_sigma = np.sqrt(uv_data)

        pos_uv_data = uv_data > 0

        if (pos_uv_data.sum() > 0):
            uv_data -= uv_data[pos_uv_data].min()
        else:
            return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan

        uv_data[uv_data < 0] = 0

        if (uv_data > 0).any():
            u_center = np.average(u_bin_centers_grid, weights=uv_data**2)
            v_center = np.average(v_bin_centers_grid, weights=uv_data**2)

            u_variance = np.average((u_bin_centers_grid-u_center)**2, weights=uv_data**2)
            v_variance = np.average((v_bin_centers_grid-v_center)**2, weights=uv_data**2)

            uv_covariance = np.average((u_bin_centers_grid-u_center)\
                                      *(v_bin_centers_grid-v_center), weights=uv_data**2)
        else:
            print('Not enough data for second pass 2d covariance calculation')
            return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan

        center2d = np.array([u_center,v_center])
        covariance2d = np.array([[u_variance,uv_covariance],
                                 [uv_covariance,v_variance]])

        # ---

        u_width = 6*np.sqrt(u_variance)
        v_width = 6*np.sqrt(v_variance)

        width = np.max([u_width,v_width])

        range2d = [[u_center-width,u_center+width],
                   [v_center-width,v_center+width]]

        u_interp_bin_centers = np.linspace(range2d[0][0],range2d[0][1],200)
        v_interp_bin_centers = np.linspace(range2d[1][0],range2d[1][1],200)

        u_data /= u_data.max()
        v_data /= v_data.max()

        mask = weights > 0

        peak_envelope.plot_projection(key, Qu[mask], Qv[mask], weights[mask], range2d,
                                      u_bin_centers, u_data, v_bin_centers, v_data,
                                      u_interp_bin_centers, norm(u_interp_bin_centers, u_variance, u_center),
                                      v_interp_bin_centers, norm(v_interp_bin_centers, v_variance, v_center))

        # Calculate peak score

        u_pk_width = 2.5*np.sqrt(u_variance)
        v_pk_width = 2.5*np.sqrt(v_variance)

        u_bkg_width = 6*np.sqrt(u_variance)
        v_bkg_width = 6*np.sqrt(v_variance)

        mask_veil = (np.abs(u_bin_centers_grid-u_center) < u_pk_width) \
                  & (np.abs(v_bin_centers_grid-v_center) < v_pk_width)

        sigstd = np.std(uv_bin_counts[mask_veil])

        mask_veil = (np.abs(u_bin_centers_grid-u_center) > u_pk_width)\
                  & (np.abs(v_bin_centers_grid-v_center) > v_pk_width)\
                  & (np.abs(u_bin_centers_grid-u_center) < u_bkg_width)\
                  & (np.abs(v_bin_centers_grid-v_center) < v_bkg_width)

        bgstd = np.std(uv_bin_counts[mask_veil])

        sig_noise_ratio = np.sum(uv_data)/np.sqrt(np.sum(uv_sigma**2))

        if bgstd > 0:
            peak_score = sigstd/bgstd
        else:
            peak_score = sigstd/0.01

        return center2d, covariance2d, peak_score, sig_noise_ratio

    else:

        print('Sum of projected weights must be greater than number of bins')
        return np.array([0,0]), np.array([[1,0],[0,1]]), np.nan, np.nan

def ellipsoid(Q0, center, variance, center2d, covariance2d, n, u, v, xsigma=3, lscale=5.99):

    Qp = np.dot(Q0,n)

    Q_offset_1d = (center-Qp)*n
    Q_offset_2d = u*center2d[0]+v*center2d[1]

    Q = Q0+Q_offset_1d+Q_offset_2d

    radius = xsigma*np.sqrt(variance)

    eigenvalues, eigenvectors = np.linalg.eig(covariance2d)

    radii = lscale*np.sqrt(eigenvalues)

    u_ = u*eigenvectors[0,0]+v*eigenvectors[1,0]
    v_ = u*eigenvectors[0,1]+v*eigenvectors[1,1]

    w0 = u_ if radii[0] < radii[1] else v_
    w1 = v_ if radii[0] < radii[1] else u_

    r0 = radii[0] if radii[0] < radii[1] else radii[1] 
    r1 = radii[1] if radii[0] < radii[1] else radii[0] 

    W = np.zeros((3,3))
    W[:,0] = w0
    W[:,1] = w1
    W[:,2] = n

    D = np.zeros((3,3))
    D[0,0] = 1/r0**2
    D[1,1] = 1/r1**2
    D[2,2] = 1/radius**2

    A = np.dot(np.dot(W, D), W.T)

    return Q, A, W, D

def decompose_ellipsoid(A, xsigma=3, lscale=5.99):

    center, center2d = 0, np.array([0.,0.])

    D, W = np.linalg.eig(A)
    #D = np.diag(D)

    n = W[:,2].copy()

    u, v = projection_axes(n)

    radius = 1/np.sqrt(D[2])
    radii = 1/np.sqrt([D[0],D[1]])

    variance = (radius/xsigma)**2
    eigenvalues = (radii/lscale)**2

    a = np.column_stack([u,v,np.ones(3)])
    b = np.column_stack([W[:,0],W[:,1],np.ones(3)])

    x = np.linalg.solve(a,b)

    eigenvectors = x[0:2,0:2].copy()
    eigenvalues = np.array([[eigenvalues[0],0],[0,eigenvalues[1]]])

    covariance2d = np.dot(np.dot(eigenvectors, eigenvalues), eigenvectors.T)

    return center, center2d, variance, covariance2d

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

def norm_integrator(peak_envelope, facility, instrument, runs, banks, indices, Q0, D, W, bin_size=0.013, box_size=1.65,
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

            print('dQp = ', dQp)
            print('Peak radius = ', 1/np.sqrt(D_pk.diagonal()))
            print('Inner radius = ', 1/np.sqrt(D_bkg_in.diagonal()))
            print('Outer radius = ', 1/np.sqrt(D_bkg_out.diagonal()))

            print('Q0_bin', Q0_bin)
            print('Q1_bin', Q1_bin)
            print('Q2_bin', Q2_bin)

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

            Q0 = np.linspace(Q0axis.getMinimum(), Q0axis.getMaximum(), Q0axis.getNBins()+1)
            Q1 = np.linspace(Q1axis.getMinimum(), Q1axis.getMaximum(), Q1axis.getNBins()+1)
            Q2 = np.linspace(Q2axis.getMinimum(), Q2axis.getMaximum(), Q2axis.getNBins()+1)

            Q0, Q1, Q2 = 0.5*(Q0[:-1]+Q0[1:]), 0.5*(Q1[:-1]+Q1[1:]), 0.5*(Q2[:-1]+Q2[1:])

            Q0, Q1, Q2 = np.meshgrid(Q0, Q1, Q2, indexing='ij', copy=False)

            # u_extents = [Q0axis.getMinimum(),Q0axis.getMaximum()]
            # v_extents = [Q1axis.getMinimum(),Q1axis.getMaximum()]
            # Q_extents = [Q2axis.getMinimum(),Q2axis.getMaximum()]

#         if facility == 'SNS':
# 
#             data = mtd['__tmpDataMD'].getSignalArray().copy()
#             norm = mtd['__tmpNormMD'].getSignalArray().copy()
# 
#             data, hQ0, hQ1, hQ2, _, _, _, _ = partial_integration(data, Q0, Q1, Q2, Q_rot, D_pk, D_bkg_in, D_bkg_out)
#             norm, _,   _,   _,   _, _, _, _ = partial_integration(norm, Q0, Q1, Q2, Q_rot, D_pk, D_bkg_in, D_bkg_out)
# 
#             mask = ~(np.isnan(data) | np.isinf(data) | (data <= 0) | np.isnan(norm) | np.isinf(norm) | (norm <= 0))
# 
#             if mask.sum() > 0.15*mask.size:
# 
#                 data_bin_counts_0, bin_edges_0 = np.histogram(hQ0[mask], bins=13, weights=data[mask])
#                 norm_bin_counts_0, _           = np.histogram(hQ0[mask], bins=13, weights=norm[mask])
# 
#                 data_bin_counts_1, bin_edges_1 = np.histogram(hQ1[mask], bins=13, weights=data[mask])
#                 norm_bin_counts_1, _           = np.histogram(hQ1[mask], bins=13, weights=norm[mask])
# 
#                 data_bin_counts_2, bin_edges_2 = np.histogram(hQ2[mask], bins=13, weights=data[mask])
#                 norm_bin_counts_2, _           = np.histogram(hQ2[mask], bins=13, weights=norm[mask])
# 
#                 bin_centers_0 = 0.5*(bin_edges_0[1:]+bin_edges_0[:-1])
#                 bin_centers_1 = 0.5*(bin_edges_1[1:]+bin_edges_1[:-1])
#                 bin_centers_2 = 0.5*(bin_edges_2[1:]+bin_edges_2[:-1])
# 
#                 tmp_0 = data_bin_counts_0/norm_bin_counts_0
#                 tmp_1 = data_bin_counts_1/norm_bin_counts_1
#                 tmp_2 = data_bin_counts_2/norm_bin_counts_2
#                 
#                 mask_0 = ~np.isnan(tmp_0) 
#                 mask_1 = ~np.isnan(tmp_1) 
#                 mask_2 = ~np.isnan(tmp_2) 
#                                 
#                 Q0_corr = np.average(bin_centers_0[mask_0], weights=tmp_0[mask_0]**2)
#                 Q1_corr = np.average(bin_centers_1[mask_1], weights=tmp_1[mask_1]**2)
#                 Q2_corr = np.average(bin_centers_2[mask_2], weights=tmp_2[mask_2]**2)
# 
#                 shifts = [Q0_corr-Q_rot[0],Q1_corr-Q_rot[1],Q2_corr-Q_rot[2]]
# 
#                 print('Peak shifts', shifts)
# 
#                 Q0_bin_corr = [extents[0][0]+shifts[0]+1e-06,steps[0],extents[0][1]+shifts[0]-1e-06]
#                 Q1_bin_corr = [extents[1][0]+shifts[1]+1e-06,steps[1],extents[1][1]+shifts[1]-1e-06]
#                 Q2_bin_corr = [extents[2][0]+shifts[2]+1e-06,steps[2],extents[2][1]+shifts[2]-1e-06]
# 
#                 print('Q0_bin_corr', Q0_bin_corr)
#                 print('Q1_bin_corr', Q1_bin_corr)
#                 print('Q2_bin_corr', Q2_bin_corr)
# 
#                 MDNorm(InputWorkspace=omd,
#                        SolidAngleWorkspace='sa',
#                        FluxWorkspace='flux',
#                        RLU=True, # not actually HKL
#                        QDimension0='{},{},{}'.format(*W[:,0]),
#                        QDimension1='{},{},{}'.format(*W[:,1]),
#                        QDimension2='{},{},{}'.format(*W[:,2]),
#                        Dimension0Name='QDimension0',
#                        Dimension1Name='QDimension1',
#                        Dimension2Name='QDimension2',
#                        Dimension0Binning='{},{},{}'.format(*Q0_bin_corr),
#                        Dimension1Binning='{},{},{}'.format(*Q1_bin_corr),
#                        Dimension2Binning='{},{},{}'.format(*Q2_bin_corr),
#                        OutputWorkspace='normDataMD',
#                        OutputDataWorkspace='tmpDataMD',
#                        OutputNormalizationWorkspace='tmpNormMD')
# 
#             else:
# 

        if facility == 'SNS':
            RenameWorkspace(InputWorkspace='__tmpDataMD', OutputWorkspace='tmpDataMD')
            RenameWorkspace(InputWorkspace='__tmpNormMD', OutputWorkspace='tmpNormMD')
            RenameWorkspace(InputWorkspace='__normDataMD', OutputWorkspace='normDataMD')

        signal = mtd['tmpDataMD'].getSignalArray().copy()

        pk, pk_Q0, pk_Q1, pk_Q2, bkg, bkg_Q0, bkg_Q1, bkg_Q2 = partial_integration(signal, Q0, Q1, Q2, Q_rot, D_pk, D_bkg_in, D_bkg_out)

        pk_data.append(pk)
        bkg_data.append(bkg)

        mask = (D_bkg_in[0,0]*(Q0-Q_rot[0])**2\
               +D_bkg_in[1,1]*(Q1-Q_rot[1])**2\
               +D_bkg_in[2,2]*(Q2-Q_rot[2])**2 > 1)\
             & (D_bkg_out[0,0]*(Q0-Q_rot[0])**2\
               +D_bkg_out[1,1]*(Q1-Q_rot[1])**2\
               +D_bkg_out[2,2]*(Q2-Q_rot[2])**2 <= 1)

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

    radii_pk_u = 1/np.sqrt(D_pk[0,0])
    radii_pk_v = 1/np.sqrt(D_pk[1,1])
    radii_pk_Q = 1/np.sqrt(D_pk[2,2])

    radii_in_u = 1/np.sqrt(D_bkg_in[0,0])
    radii_in_v = 1/np.sqrt(D_bkg_in[1,1])
    radii_in_Q = 1/np.sqrt(D_bkg_in[2,2])

    radii_out_u = 1/np.sqrt(D_bkg_out[0,0])
    radii_out_v = 1/np.sqrt(D_bkg_out[1,1])
    radii_out_Q = 1/np.sqrt(D_bkg_out[2,2])

    t = np.linspace(0,2*np.pi,100)

    x_pk_Qu, y_pk_Qu = radii_pk_Q*np.cos(t)+Q_rot[2], radii_pk_u*np.sin(t)+Q_rot[0]
    x_pk_Qv, y_pk_Qv = radii_pk_Q*np.cos(t)+Q_rot[2], radii_pk_v*np.sin(t)+Q_rot[1]
    x_pk_uv, y_pk_uv = radii_pk_u*np.cos(t)+Q_rot[0], radii_pk_v*np.sin(t)+Q_rot[1]

    x_in_Qu, y_in_Qu = radii_in_Q*np.cos(t)+Q_rot[2], radii_in_u*np.sin(t)+Q_rot[0]
    x_in_Qv, y_in_Qv = radii_in_Q*np.cos(t)+Q_rot[2], radii_in_v*np.sin(t)+Q_rot[1]
    x_in_uv, y_in_uv = radii_in_u*np.cos(t)+Q_rot[0], radii_in_v*np.sin(t)+Q_rot[1]

    x_out_Qu, y_out_Qu = radii_out_Q*np.cos(t)+Q_rot[2], radii_out_u*np.sin(t)+Q_rot[0]
    x_out_Qv, y_out_Qv = radii_out_Q*np.cos(t)+Q_rot[2], radii_out_v*np.sin(t)+Q_rot[1]
    x_out_uv, y_out_uv = radii_out_u*np.cos(t)+Q_rot[0], radii_out_v*np.sin(t)+Q_rot[1]

    peak_envelope.plot_integration(signal, Q0_bin, Q1_bin, Q2_bin,
                                   x_pk_Qu, y_pk_Qu, x_pk_Qv, y_pk_Qv, x_pk_uv, y_pk_uv,
                                   x_in_Qu, y_in_Qu, x_in_Qv, y_in_Qv, x_in_uv, y_in_uv,
                                   x_out_Qu, y_out_Qu, x_out_Qv, y_out_Qv, x_out_uv, y_out_uv)

    return pk_data, pk_norm, bkg_data, bkg_norm, dQp, pk_Q0, pk_Q1, pk_Q2, bkg_Q0, bkg_Q1, bkg_Q2

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

def pre_integration(runs, outname, directory, facility, instrument, ipts, ub_file, reflection_condition,
                    spectrum_file, counts_file, tube_calibration, detector_calibration,
                    mod_vector_1=[0,0,0], mod_vector_2=[0,0,0], mod_vector_3=[0,0,0],
                    max_order=0, cross_terms=False, exp=None):

    min_d_spacing = 0.7
    max_d_spacing= 20

    # peak centroid radius ---------------------------------------------------------
    centroid_radius = 0.125

    # goniometer axis --------------------------------------------------------------
    gon_axis = 'BL9:Mot:Sample:Axis3.RBV'

    load_normalization_calibration(facility, instrument, spectrum_file, counts_file,
                                   tube_calibration, detector_calibration)

    if mtd.doesExist('sa'):
        CreatePeaksWorkspace(InstrumentWorkspace='sa', NumberOfPeaks=0, OutputType='Peak', OutputWorkspace='tmp')
    else:
        CreatePeaksWorkspace(InstrumentWorkspace='van', NumberOfPeaks=0, OutputType='Peak', OutputWorkspace='tmp')

    if facility == 'HFIR':
        LoadEmptyInstrument(InstrumentName='HB3A', OutputWorkspace='rws')

    for i, r in enumerate(runs):
        print('Processing run : {}'.format(r))
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
                UB = mtd[ows].getExperimentInfo(0).run().getProperty('ubmatrix').value
                UB = np.array([float(ub) for ub in UB.split(' ')]).reshape(3,3)
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

                use_inner = True

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
                                 CalculateWavelength=True,
                                 Wavelength=lamda,
                                 InnerGoniometer=use_inner,
                                 MinAngle=min_angle,
                                 MaxAngle=max_angle,
                                 FlipX=True if instrument == 'HB3A' else False,
                                 OutputType='LeanElasticPeak',
                                 OutputWorkspace='main')

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

            i_diff = abs(int(round(5/rotation_diff)))

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
                         LogUnit='meter',
                         NumberType='Double')

            LoadInstrument(Workspace='rws', InstrumentName='HB3A', RewriteSpectraMap=False)
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

                tmp_directory = '{}/{}/'.format(directory,'_'.join(outname.split('_')[:-1]))
                tmp_outname = '{}_{}_{}_{}.nxs'.format(instrument,exp,r,no)

                SaveMD(InputWorkspace='crop_data', Filename=os.path.join(tmp_directory, tmp_outname))
                DeleteWorkspace('crop_data')

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

    SaveNexus(InputWorkspace='tmp', Filename=os.path.join(directory, outname+'_pk.nxs'))
    SaveIsawUB(InputWorkspace='tmp', Filename=os.path.join(directory, outname+'.mat'))

def partial_load(facility, instrument, runs, banks, indices, phi, chi, omega, norm_scale,
                 directory, ipts, outname, exp=None):

    for r, b, i, p, c, o in zip(runs, banks, indices, phi, chi, omega):

        if facility == 'SNS':
            ows = '{}_{}_{}'.format(instrument,r,b)
        else:
            ows = '{}_{}_{}_{}'.format(instrument,exp,r,i)

        omd = ows+'_md'

        if facility == 'SNS':

            if not mtd.doesExist(omd):

                filename = '/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(instrument,ipts,instrument,r)
                LoadEventNexus(Filename=filename, 
                               BankName='bank{}'.format(b), 
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

                filename = '{}/{}/{}.nxs'.format(directory,'_'.join(outname.split('_')[:-1]),ows)
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

            if len(key_list) == 0:
                MaskBTP(Workspace='sa', Bank=b)
                if bank_group.get(b) is not None:
                    MaskSpectra(InputWorkspace='flux', 
                                InputWorkspaceIndexType='SpectrumNumber',
                                InputWorkspaceIndexSet=bank_group[b],
                                OutputWorkspace='flux')

        else:

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

def integration_loop(keys, outname, ref_peak_dictionary, ref_dict, filename,
                     spectrum_file, counts_file, tube_calibration, detector_calibration,
                     directory, facility, instrument, ipts, runs,
                     split_angle, a, b, c, alpha, beta, gamma, reflection_condition,
                     mod_vector_1, mod_vector_2, mod_vector_3, max_order, cross_terms,
                     chemical_formula, z_parameter, sample_mass, experiment):

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
    peaks = peak_dictionary.to_be_integrated()

    peak_envelope = PeakEnvelope(directory+'/{}.pdf'.format(outname))
    peak_envelope.show_plots(False)

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

        peaks_list = peak_dictionary.peak_dict.get(key)

        redundancies = peaks[key]

        for j, redundancy in enumerate(redundancies):

            runs = peaks_list[j].get_run_numbers()
            banks = peaks_list[j].get_bank_numbers()

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

        all_banks = list(bank_group.keys())

        for b in all_banks:
            if b not in banks:
                MaskBTP(Workspace='sa', Bank=b)
                MaskSpectra(InputWorkspace='flux', 
                            InputWorkspaceIndexType='SpectrumNumber',
                            InputWorkspaceIndexSet=bank_group[b],
                            OutputWorkspace='flux')

    for i, key in enumerate(keys):

        key = tuple(key)

        print('Integrating peak : {}'.format(key))

        redundancies = peaks[key]

        peaks_list = peak_dictionary.peak_dict.get(key)

        fixed = False
        if ref_dict is not None:
            ref_peaks = ref_peak_dictionary.peak_dict.get(key)
            if ref_peaks is not None:
                if len(ref_peaks) == len(redundancies):
                    fixed = True

        h, k, l, m, n, p = key

        d = peak_dictionary.get_d(h, k, l, m, n, p)

        for j, redundancy in enumerate(redundancies):

            runs, numbers = redundancy

            peak_envelope.clear_plots()

            banks = peaks_list[j].get_bank_numbers().tolist()
            indices = peaks_list[j].get_peak_indices().tolist()

            Q0 = peaks_list[j].get_Q()

            phi = peaks_list[j].get_phi_angles()
            chi = peaks_list[j].get_chi_angles()
            omega = peaks_list[j].get_omega_angles()

            R = peaks_list[j].get_goniometers()[0]

            partial_load(facility, instrument, runs, banks, indices, 
                         phi, chi, omega, norm_scale,
                         directory, ipts, outname, experiment)

            if fixed:

                ref_peak = ref_peaks[j]
                Q0 = ref_peak.get_Q()
                A = ref_peak.get_A()
                D, W = np.linalg.eig(A)
                D = np.diag(D)

                radii = 1/np.sqrt(np.diagonal(D)) 

                peak_fit, peak_bkg_ratio, peak_score2d = 0, 0, 0

                if np.isclose(np.abs(np.linalg.det(W)),1) and (radii < 0.3).all() and (radii > 0).all():

                    data = norm_integrator(peak_envelope, facility, instrument, runs, banks, indices, Q0, D, W, exp=experiment)

                    peak_dictionary.integrated_result(key, Q0, A, peak_fit, peak_bkg_ratio, peak_score2d, data, j)

            else:

                remove = False

                Q, Qx, Qy, Qz, weights = box_integrator(facility, instrument, runs, banks, indices, norm_scale, Q0, binsize=0.005, radius=0.15, exp=experiment)
# 
#                 if facility == 'SNS':
#                     n = Q0/np.linalg.norm(Q0)
#                 else:
#                     Ql = np.dot(R,Q0)
#                     t, p = np.arccos(Ql[2]/np.linalg.norm(Ql)), np.arctan2(Ql[1],Ql[0])
#                     n = np.array([np.cos(t)*np.cos(p),np.cos(t)*np.sin(p),-np.sin(t)])
                n = Q0/np.linalg.norm(Q0)

                u, v = projection_axes(n)

                center, variance, peak_fit, peak_bkg_ratio, sig_noise_ratio, peak_total_data_ratio = Q_profile(peak_envelope, key, Q, Qx, Qy, Qz, weights, 
                                                                                                               Q0, n, radius=0.15, bins=31)

                # print('Peak-fit Q: {}'.format(peak_fit))
                # print('Peak background ratio Q: {}'.format(peak_bkg_ratio))
                # print('Signal-noise ratio Q: {}'.format(sig_noise_ratio))
                # print('Peak-total to subtrated-data ratio Q: {}'.format(peak_total_data_ratio))

                Qp = np.dot(Q0,n)

                max_width = 0.1 if facility == 'SNS' else 0.2

                if (sig_noise_ratio < 3 or 3*np.sqrt(variance) > max_width or np.abs(Qp-center) > 0.1):

                    remove = True

                center2d, covariance2d, peak_score2d, sig_noise_ratio2d = projected_profile(peak_envelope, d, Q, Qx, Qy, Qz, weights,
                                                                                            Q0, n, u, v, center, variance, radius=0.1,
                                                                                            bins=21, bins2d=21)

                # print('Peak-score 2d: {}'.format(peak_score2d))
                # print('Signal-noise ratio 2d: {}'.format(sig_noise_ratio2d))

                if (peak_score2d < 2 or np.isinf(peak_score2d) or np.isnan(peak_score2d) or np.linalg.norm(center2d) > 0.15 or sig_noise_ratio2d < 3):

                    remove = True

                Qc, A, W, D = ellipsoid(Q0, center, variance, center2d, covariance2d, 
                                        n, u, v, xsigma=4, lscale=5)

                peak_envelope.plot_projection_ellipse(*peak.draw_ellispoid(center2d, covariance2d, lscale=5))

                center, variance, peak_fit, peak_bkg_ratio, sig_noise_ratio, peak_total_data_ratio = extracted_Q_profile(peak_envelope, key, Q, Qx, Qy, Qz, weights, 
                                                                                                                         Q0, n, u, v, center, variance, center2d, covariance2d, bins=21)

                # print('Peak-fit Q second pass: {}'.format(peak_fit))
                # print('Peak background ratio Q second pass: {}'.format(peak_bkg_ratio))
                # print('Signal-noise ratio Q second pass: {}'.format(sig_noise_ratio))
                # print('Peak-total to subtrated-data ratio Q: {}'.format(peak_total_data_ratio))

                Qp = np.dot(Qc,n)

                if (sig_noise_ratio < 3 or 3*np.sqrt(variance) > max_width or np.abs(Qp-center) > 0.1 or peak_total_data_ratio > 2.5):

                    remove = True

                if not np.isnan(covariance2d).any():

                    Q0, A, W, D = ellipsoid(Q0, center, variance, center2d, covariance2d, 
                                            n, u, v, xsigma=4, lscale=5)

                    radii = 1/np.sqrt(np.diagonal(D)) 

                    print('Peak-radii: {}'.format(radii), remove)

                    if np.isclose(np.abs(np.linalg.det(W)),1) and (radii < 0.3).all() and (radii > 0).all() and not np.isclose(radii, 0).any() and not remove:

                        data = norm_integrator(peak_envelope, facility, instrument, runs, banks, indices, Q0, D, W, exp=experiment)

                        peak_dictionary.integrated_result(key, Q0, A, peak_fit, peak_bkg_ratio, peak_score2d, data, j)

                        peak_envelope.write_figure()

                    else:

                        remove = True

                else:

                    remove = True

                if remove:

                    peak_dictionary.partial_result(key, Q0, A, peak_fit, peak_bkg_ratio, peak_score2d, j)

            #peak_envelope.write_figure()

            runs_banks, bank_keys = partial_cleanup(runs, banks, indices, facility, instrument, runs_banks, bank_keys, bank_group, key, exp=experiment)

        if i % 15 == 0:
            peak_dictionary.save_hkl(directory+'/{}.hkl'.format(outname))       
            peak_dictionary.save(directory+'/{}.pkl'.format(outname))

    peak_dictionary.save_hkl(directory+'/{}.hkl'.format(outname))       
    peak_dictionary.save(directory+'/{}.pkl'.format(outname))
    peak_envelope.create_pdf()

    if mtd.doesExist('sa'):
        DeleteWorkspace('sa')
    if mtd.doesExist('flux'):
        DeleteWorkspace('flux')
    if mtd.doesExist('van'):
        DeleteWorkspace('van')