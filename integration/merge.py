from mantid.simpleapi import *

import os

import numpy as np

import peak
from peak import PeakEnvelope, PeakDictionary

def box_integrator(instrument, runs, banks, proton_charge, Q0, binsize=0.001, radius=0.15, exp=None):

    for i, (r, b) in enumerate(zip(runs, banks)):

        if exp is None:
            ows = '{}_{}_{}'.format(instrument,r,b)
        else:
            ows = '{}_{}_{}'.format(instrument,exp,r)

        omd = ows+'_md'

        if i == 0:

            Qr = np.array([radius,radius,radius])

            nQ = np.round(2*Qr/binsize).astype(int)+1

            Qmin, Qmax = Q0-Qr, Q0+Qr

        BinMD(InputWorkspace=omd,
              AlignedDim0='Q_sample_x,{},{},{}'.format(Qmin[0],Qmax[0],nQ[0]),
              AlignedDim1='Q_sample_y,{},{},{}'.format(Qmin[1],Qmax[1],nQ[1]),
              AlignedDim2='Q_sample_z,{},{},{}'.format(Qmin[2],Qmax[2],nQ[2]),
              OutputWorkspace='__tmp')

        __tmp = mtd['__tmp']*proton_charge[r]

        if i == 0:
            CloneMDWorkspace(InputWorkspace=__tmp, OutputWorkspace='box')
        else:
            PlusMD(LHSWorkspace='box', RHSWorkspace=__tmp, OutputWorkspace='box')

    SetMDFrame('box', MDFrame='QSample', Axes=[0,1,2])
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

def Q_profile(peak_envelope, key, Q, weights, Q0, radius=0.15, bins=30):

    Qmod = np.linalg.norm(Q0)

    mask = (np.abs(Q-Qmod) < radius)

    x = Q[~mask].flatten()
    y = weights[~mask].flatten()

    #A = (np.array([x*0+1, x])*weights[~mask]**2).T
    #B = y.flatten()*weights[~mask]**2

    A = (np.array([x*0+1, x])).T
    B = y.flatten()

    coeff, r, rank, s = np.linalg.lstsq(A, B)

    bkg_weights = (coeff[0]+coeff[1]*Q)

    data_weights = weights-bkg_weights

    data_weights[data_weights < 0] = 0

    if np.sum(data_weights > 0) > bins:

        bin_counts, bin_edges = np.histogram(Q[mask], bins=bins, weights=data_weights[mask])

        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])

        data = bin_counts.copy()

        pos_data = data > 0

        if (pos_data.sum() > 0):
            data -= data[pos_data].min()
        else:
            return Qmod, np.nan, np.nan, np.nan, np.nan, np.nan

        data[data < 0] = 0

        total, _ = np.histogram(Q[mask], bins=bin_edges, weights=weights[mask])

        arg_center = np.argmax(bin_counts) # this works for no nearby contamination
        center = bin_centers[arg_center]

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
            return Qmod, np.nan, np.nan, np.nan, np.nan

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
                        Q0, u, v, center, variance, center2d, covariance2d, bins=30):

    Qmod = np.linalg.norm(Q0)

    Qu = u[0]*(Qx-Q0[0])+u[1]*(Qy-Q0[1])+u[2]*(Qz-Q0[2])
    Qv = v[0]*(Qx-Q0[0])+v[1]*(Qy-Q0[1])+v[2]*(Qz-Q0[2])

    u_center, v_center = center2d

    eigenvalues, eigenvectors = np.linalg.eig(covariance2d)

    radii = 4*np.sqrt(eigenvalues)

    D = np.diag(1/radii**2)
    W = eigenvectors.copy()

    A = np.dot(np.dot(W,D),W.T)

    mask = (A[0,0]*(Qu-u_center)+A[0,1]*(Qv-v_center))*(Qu-u_center)+\
         + (A[1,0]*(Qu-u_center)+A[1,1]*(Qv-v_center))*(Qv-v_center) <= 1 & (np.abs(Q-center) < 4*np.sqrt(variance))

    x = Q[~mask].flatten()
    y = weights[~mask].flatten()

    A = (np.array([x*0+1, x])).T
    B = y.flatten()

    coeff, r, rank, s = np.linalg.lstsq(A, B)

    bkg_weights = (coeff[0]+coeff[1]*Q)

    data_weights = weights-bkg_weights

    data_weights[data_weights < 0] = 0

    if np.sum(data_weights > 0) > bins:

        bin_counts, bin_edges = np.histogram(Q[mask], bins=bins, weights=data_weights[mask])

        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])

        data = bin_counts.copy()

        pos_data = data > 0

        if (pos_data.sum() > 0):
            data -= data[pos_data].min()
        else:
            return Qmod, np.nan, np.nan, np.nan, np.nan, np.nan

        data[data < 0] = 0

        total, _ = np.histogram(Q[mask], bins=bin_edges, weights=weights[mask])

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

def norm(Q, var, mu):

    return np.exp(-0.5*(Q-mu)**2/var) #1/np.sqrt(2*np.pi*var)

def projection_axes(Q0):

    n = Q0/np.linalg.norm(Q0)
    n_ind = np.argmin(np.abs(n))

    u = np.zeros(3)
    u[n_ind] = 1

    u = np.cross(n, u)
    u /= np.linalg.norm(u)

    v = np.cross(n, u)
    v *= np.sign(np.dot(np.cross(u,n),v))

    return n, u, v

def projected_profile(peak_envelope, key, Q, Qx, Qy, Qz, weights,
                      Q0, u, v, center, variance, radius=0.15, bins=16, bins2d=50):

    # Estimate 1d variance

    Qu = u[0]*(Qx-Q0[0])+u[1]*(Qy-Q0[1])+u[2]*(Qz-Q0[2])
    Qv = v[0]*(Qx-Q0[0])+v[1]*(Qy-Q0[1])+v[2]*(Qz-Q0[2])

    width = 4*np.sqrt(variance)

    if np.sum(weights > 0) > 0:

        mask = (np.abs(Q-center) < width)\
             & ((Qx-Q0[0])**2+(Qy-Q0[1])**2+(Qz-Q0[2])**2 < np.max([width,radius])**2)

        x = Qu[~mask].flatten()
        y = Qv[~mask].flatten()
        z = weights[~mask].flatten()

        A = (np.array([x*0+1, x, y])*weights[~mask]**2).T
        B = z.flatten()*weights[~mask]**2

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

        coeff, r, rank, s = np.linalg.lstsq(A, B)

        bkg_weights = (coeff[0]+coeff[1]*Qu+coeff[2]*Qv)

        data_weights = weights-bkg_weights

        data_weights[data_weights < 0] = 0

        range2d = [[u_center-u_width,u_center+u_width],
                   [v_center-v_width,v_center+v_width]]

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
             + (A[1,0]*(Qu-u_center)+A[1,1]*(Qv-v_center))*(Qv-v_center) <= 1 & (np.abs(Q-center) < width)

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

    Q_offset_1d = (center-np.linalg.norm(Q0))*n
    Q_offset_2d = u*center2d[0]+v*center2d[1]

    Q = Q0+Q_offset_1d+Q_offset_2d

    radius = xsigma*np.sqrt(variance)

    eigenvalues, eigenvectors = np.linalg.eig(covariance2d)

    radii = lscale*np.sqrt(eigenvalues)

    u_ = u*eigenvectors[0,0]+v*eigenvectors[1,0]
    v_ = u*eigenvectors[0,1]+v*eigenvectors[1,1]

    W = np.zeros((3,3))
    W[:,0] = u_
    W[:,1] = v_
    W[:,2] = n

    D = np.zeros((3,3))
    D[0,0] = 1/radii[0]**2
    D[1,1] = 1/radii[1]**2
    D[2,2] = 1/radius**2

    A = np.dot(np.dot(W, D), W.T)

    return Q, A, W, D

def decompose_ellipsoid(A, xsigma=3, lscale=5.99):

    center, center2d = 0, np.array([0.,0.])

    D, W = np.linalg.eig(A)
    #D = np.diag(D)

    n = W[:,2].copy()

    _, u, v = projection_axes(n)

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

def partial_integration(signal, Qx, Qy, Qz, Q_rot, D_pk, D_bkg_in, D_bkg_out):

    mask = D_pk[0,0]*(Qx-Q_rot[0])**2\
         + D_pk[1,1]*(Qy-Q_rot[1])**2\
         + D_pk[2,2]*(Qz-Q_rot[2])**2 <= 1

    pk = signal[mask].astype('f')

    mask = (D_bkg_in[0,0]*(Qx-Q_rot[0])**2\
           +D_bkg_in[1,1]*(Qy-Q_rot[1])**2\
           +D_bkg_in[2,2]*(Qz-Q_rot[2])**2 > 1)\
         & (D_bkg_out[0,0]*(Qx-Q_rot[0])**2\
           +D_bkg_out[1,1]*(Qy-Q_rot[1])**2\
           +D_bkg_out[2,2]*(Qz-Q_rot[2])**2 <= 1)

    bkg = signal[mask].astype('f')

    return pk, bkg

def norm_integrator(peak_envelope, facility, instrument, runs, banks, Q0, D, W, bin_size=0.013, box_size=1.65,
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
    _, Q_bin_size = np.linspace(Q_rot[2]-dQ[2],Q_rot[2]+dQ[2], 27, retstep=True)

    if not np.isclose(Q0_bin_size, 0):
        dQp[0] = np.min([Q0_bin_size,bin_size])
    if not np.isclose(Q1_bin_size, 0):
        dQp[1] = np.min([Q1_bin_size,bin_size])
    dQp[2] = Q_bin_size

    pk_data, pk_norm = [], []
    bkg_data, bkg_norm = [], []

    if mtd.doesExist('dataMD'): DeleteWorkspace('dataMD')
    if mtd.doesExist('normMD'): DeleteWorkspace('normMD')

    for i, (r, b) in enumerate(zip(runs, banks)):

        if facility == 'SNS':
            ows = '{}_{}_{}'.format(instrument,r,b)
        else:
            ows = '{}_{}_{}'.format(instrument,exp,r)

        omd = ows+'_md'

        if mtd.doesExist('tmpDataMD'): DeleteWorkspace('tmpDataMD')
        if mtd.doesExist('tmpNormMD'): DeleteWorkspace('tmpNormMD')

        SetUB(omd, UB=np.eye(3)/(2*np.pi)) # hack to transform axes

        if i == 0:
            Q0_bin = [Q_rot[0]-dQ[0],dQp[0],Q_rot[0]+dQ[0]]
            Q1_bin = [Q_rot[1]-dQ[1],dQp[1],Q_rot[1]+dQ[1]]
            Q2_bin = [Q_rot[2]-dQ[2],dQp[2],Q_rot[2]+dQ[2]]

            print('dQp = ', dQp)
            print('Peak radius = ', 1/np.sqrt(D_pk.diagonal()))
            print('Inner radius = ', 1/np.sqrt(D_bkg_in.diagonal()))
            print('Outer radius = ', 1/np.sqrt(D_bkg_out.diagonal()))

            print('Q0_bin', Q0_bin)
            print('Q1_bin', Q1_bin)
            print('Q2_bin', Q2_bin)

            extents = [Q_rot[0]-dQ[0],Q_rot[0]+dQ[0],
                       Q_rot[1]-dQ[1],Q_rot[1]+dQ[1],
                       Q_rot[2]-dQ[2],Q_rot[2]+dQ[2]]

            bins = [int(round(2*dQ[0]/dQp[0]))+1,
                    int(round(2*dQ[1]/dQp[1]))+1,
                    int(round(2*dQ[2]/dQp[2]))+1]

        if facility == 'SNS':
            MDNorm(InputWorkspace=omd,
                   SolidAngleWorkspace='sa_{}'.format(b),
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
                   OutputDataWorkspace='tmpDataMD',
                   OutputNormalizationWorkspace='tmpNormMD')
                   
        else:
            BinMD(InputWorkspace=omd, AxisAligned=False, NormalizeBasisVectors=False,
                  BasisVector0='Q0,A^-1,{},{},{}'.format(*W[:,0]),
                  BasisVector1='Q1,A^-1,{},{},{}'.format(*W[:,1]),
                  BasisVector2='Q2,A^-1,{},{},{}'.format(*W[:,2]),
                  OutputExtents='{},{},{},{},{},{}'.format(*extents),
                  OutputBins='{},{},{}'.format(*bins),
                  OutputWorkspace='tmpDataMD')

            BinMD(InputWorkspace=ows+'_van', AxisAligned=False, NormalizeBasisVectors=False,
                  BasisVector0='Q0,A^-1,{},{},{}'.format(*W[:,0]),
                  BasisVector1='Q1,A^-1,{},{},{}'.format(*W[:,1]),
                  BasisVector2='Q2,A^-1,{},{},{}'.format(*W[:,2]),
                  OutputExtents='{},{},{},{},{},{}'.format(*extents),
                  OutputBins='{},{},{}'.format(*bins),
                  OutputWorkspace='tmpNormMD')

            DivideMD(LHSWorkspace='tmpDataMD',
                     RHSWorkspace='tmpNormMD',
                     OutputWorkspace='__normDataMD')

        if i == 0:
            Qxaxis = mtd['__normDataMD'].getXDimension()
            Qyaxis = mtd['__normDataMD'].getYDimension()
            Qzaxis = mtd['__normDataMD'].getZDimension()

            Qx = np.linspace(Qxaxis.getMinimum(), Qxaxis.getMaximum(), Qxaxis.getNBins()+1)
            Qy = np.linspace(Qyaxis.getMinimum(), Qyaxis.getMaximum(), Qyaxis.getNBins()+1)
            Qz = np.linspace(Qzaxis.getMinimum(), Qzaxis.getMaximum(), Qzaxis.getNBins()+1)

            Qx, Qy, Qz = 0.5*(Qx[:-1]+Qx[1:]), 0.5*(Qy[:-1]+Qy[1:]), 0.5*(Qz[:-1]+Qz[1:])

            Qx, Qy, Qz = np.meshgrid(Qx, Qy, Qz, indexing='ij', copy=False)

            # u_extents = [Qxaxis.getMinimum(),Qxaxis.getMaximum()]
            # v_extents = [Qyaxis.getMinimum(),Qyaxis.getMaximum()]
            # Q_extents = [Qzaxis.getMinimum(),Qzaxis.getMaximum()]

        signal = mtd['tmpDataMD'].getSignalArray().copy()

        pk, bkg = partial_integration(signal, Qx, Qy, Qz, Q_rot, D_pk, D_bkg_in, D_bkg_out)

        pk_data.append(pk)
        bkg_data.append(bkg)

        mask = (D_bkg_in[0,0]*(Qx-Q_rot[0])**2\
               +D_bkg_in[1,1]*(Qy-Q_rot[1])**2\
               +D_bkg_in[2,2]*(Qz-Q_rot[2])**2 > 1)\
             & (D_bkg_out[0,0]*(Qx-Q_rot[0])**2\
               +D_bkg_out[1,1]*(Qy-Q_rot[1])**2\
               +D_bkg_out[2,2]*(Qz-Q_rot[2])**2 <= 1)

        signal = mtd['tmpNormMD'].getSignalArray().copy()

        pk, bkg = partial_integration(signal, Qx, Qy, Qz, Q_rot, D_pk, D_bkg_in, D_bkg_out)

        pk_norm.append(pk)
        bkg_norm.append(bkg)

        if i == 0:
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

    return pk_data, pk_norm, bkg_data, bkg_norm, dQp

def load_normalization_calibration(facility, spectrum_file, counts_file,
                                   tube_calibration, detector_calibration):

    if not mtd.doesExist('tube_table') and tube_calibration is not None:
        LoadNexus(Filename=tube_calibration, OutputWorkspace='tube_table')

    if not mtd.doesExist('sa') and counts_file is not None:
        if facility == 'SNS':
            LoadNexus(Filename=counts_file, OutputWorkspace='sa')
            NormaliseByCurrent('sa', OutputWorkspace='sa')
        else:
            LoadMD(Filename=counts_file, OutputWorkspace='van')

    if not mtd.doesExist('flux') and spectrum_file is not None:
        LoadNexus(Filename=spectrum_file, OutputWorkspace='flux')

    if mtd.doesExist('tube_table'):
        if mtd.doesExist('sa'):
            ApplyCalibration(Workspace='sa', CalibrationTable='tube_table')
        if mtd.doesExist('flux'):
            ApplyCalibration(Workspace='flux', CalibrationTable='tube_table')

    if detector_calibration is not None:
        ext = os.path.splitext(detector_calibration)[1]
        if mtd.doesExist('sa'):
            if ext == '.xml':
                LoadParameterFile(Workspace='sa', Filename=detector_calibration)
            else:
                LoadIsawDetCal(Workspace='sa', Filename=detector_calibration)
        if mtd.doesExist('flux'):
            if ext == '.xml':
                LoadParameterFile(Workspace='flux', Filename=detector_calibration)
            else:
                LoadIsawDetCal(Workspace='flux', Filename=detector_calibration)

def pre_integration(runs, directory, facility, instrument, ipts, ub_file, reflection_condition,
                    spectrum_file, counts_file, tube_calibration, detector_calibration,
                    mod_vector_1=[0,0,0], mod_vector_2=[0,0,0], mod_vector_3=[0,0,0],
                    max_order=0, cross_terms=False, exp=None):

    min_d_spacing = 0.7
    max_d_spacing= 20

    # peak centroid radius ---------------------------------------------------------
    centroid_radius = 0.125

    # goniometer axis --------------------------------------------------------------
    gon_axis = 'BL9:Mot:Sample:Axis3.RBV'

    load_normalization_calibration(facility, spectrum_file, counts_file,
                                   tube_calibration, detector_calibration)

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
                    CopyInstrumentParameters(InputWorkspace='sa', OutputWorkspace=ows)

                if instrument == 'CORELLI':
                    SetGoniometer(Workspace=ows, Axis0='{},0,1,0,1'.format(gon_axis))
                else:
                    SetGoniometer(Workspace=ows, Goniometers='Universal')

                ConvertToMD(InputWorkspace=ows,
                            OutputWorkspace=omd,
                            QDimensions='Q3D',
                            dEAnalysisMode='Elastic',
                            Q3DFrames='Q_sample',
                            LorentzCorrection=True,
                            MinValues='-20,-20,-20',
                            MaxValues='20,20,20',
                            Uproj='1,0,0',
                            Vproj='0,1,0',
                            Wproj='0,0,1',
                            SplitInto=2,
                            SplitThreshold=50,
                            MaxRecursionDepth=13,
                            MinRecursionDepth=7)   

            else:
                filename = '/HFIR/{}/IPTS-{}/shared/autoreduce/{}_exp{:04}_scan{:04}.nxs'.format(instrument,ipts,instrument,exp,r)
                LoadMD(Filename=filename, OutputWorkspace=ows)

                scale = mtd[ows].getExperimentInfo(0).run().getProperty('monitor').value
                norm = np.sum(mtd[ows].getExperimentInfo(0).run().getProperty('monitor').value)

                scale /= norm

                temp = mtd[ows].getExperimentInfo(0).run().getProperty('coldtip').value
                Q3, Q1 = np.percentile(temp,75), np.percentile(temp,25)

                IQR = Q3-Q1

                mask = (temp > Q3+1.5*IQR) | (temp < Q1-1.5*IQR)

                scale[mask] = 0

                d = mtd[ows].getSignalArray().copy()
                d[...,mask] = 0

                v = mtd['van'].getSignalArray().copy()

                mtd[ows].setSignalArray(d)
                mtd[ows].setErrorSquaredArray(d)

                SetGoniometer(Workspace=ows,
                              Axis0='omega,0,1,0,-1',
                              Axis1='chi,0,0,1,-1',
                              Axis2='phi,0,1,0,-1',
                              Average=False)

                wavelength = float(mtd[ows].getExperimentInfo(0).run().getProperty('wavelength').value)

                ConvertHFIRSCDtoMDE(InputWorkspace=ows,
                                    Wavelength=wavelength,
                                    MinValues='-10,-10,-10',
                                    MaxValues='10,10,10',
                                    SplitInto=5,
                                    SplitThreshold=1000,
                                    MaxRecursionDepth=13,
                                    OutputWorkspace=omd)

                mtd[ows].setSignalArray(v.repeat(d.shape[2]).reshape(*d.shape)*scale)
                mtd[ows].setErrorSquaredArray(v.repeat(d.shape[2]).reshape(*d.shape)*scale)

                ConvertHFIRSCDtoMDE(InputWorkspace=ows,
                                    Wavelength=wavelength,
                                    MinValues='-10,-10,-10',
                                    MaxValues='10,10,10',
                                    SplitInto=5,
                                    SplitThreshold=1000,
                                    MaxRecursionDepth=13,
                                    OutputWorkspace=ows+'_van')

            if type(ub_file) is list:
                LoadIsawUB(InputWorkspace=omd, Filename=ub_file[i])
            elif type(ub_file) is str:
                LoadIsawUB(InputWorkspace=omd, Filename=ub_file)
            else:
                UB = mtd[ows].getExperimentInfo(0).run().getProperty('ubmatrix').value
                UB = [float(ub) for ub in UB.split(' ')]
                UB = np.array(UB).reshape(3,3)
                SetUB(omd, UB=UB)

        else:

            if facility == 'HFIR':
                ws = ows if mtd.doesExist(ows) else omd
                wavelength = float(mtd[ws].getExperimentInfo(0).run().getProperty('wavelength').value)

        # peak prediction parameters ---------------------------------------------------
        if instrument == 'CORELLI':
            min_wavelength = 0.63
            max_wavelength= 2.51
        elif instrument == 'MANDI':
            min_wavelength = 0.4
            max_wavelength = 4
        elif instrument == 'HB3A':
            min_wavelength = 0.95*wavelength
            max_wavelength = 1.05*wavelength

        if not mtd.doesExist(opk):

            if facility == 'SNS':
                PredictPeaks(InputWorkspace=omd,
                             WavelengthMin=min_wavelength,
                             WavelengthMax=max_wavelength,
                             MinDSpacing=min_d_spacing,
                             MaxDSpacing=max_d_spacing,
                             OutputType='Peak',
                             ReflectionCondition=reflection_condition,
                             OutputWorkspace=opk)
            else:
                wavelength = float(mtd[omd].getExperimentInfo(0).run().getProperty('wavelength').value)

                PredictPeaks(InputWorkspace=omd,
                             WavelengthMin=wavelength*0.95,
                             WavelengthMax=wavelength*1.05,
                             MinDSpacing=min_d_spacing,
                             MaxDSpacing=max_d_spacing,
                             ReflectionCondition=reflection_condition,
                             CalculateGoniometerForCW=True,
                             CalculateWavelength=False,
                             Wavelength=wavelength,
                             InnerGoniometer=True,
                             FlipX=True,
                             OutputType='Peak',
                             OutputWorkspace=opk)

                HFIRCalculateGoniometer(Workspace=opk,
                                        Wavelength=wavelength,
                                        OverrideProperty=True,
                                        InnerGoniometer=True,
                                        FlipX=True)

            if max_order > 0:
                PredictSatellitePeaks(Peaks=opk,
                                      SatellitePeaks=opk,
                                      ModVector1=mod_vector_1,
                                      ModVector2=mod_vector_2,
                                      ModVector3=mod_vector_3,
                                      MaxOrder=max_order,
                                      CrossTerms=cross_terms,
                                      IncludeIntegerHKL=True,
                                      IncludeAllPeaksInRange=False)

            # CentroidPeaksMD(InputWorkspace=omd,
            #                 PeakRadius=centroid_radius,
            #                 PeaksWorkspace=opk,
            #                 OutputWorkspace=opk)

            # CentroidPeaksMD(InputWorkspace=omd,
            #                 PeakRadius=centroid_radius,
            #                 PeaksWorkspace=opk,
            #                 OutputWorkspace=opk)

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

            SaveNexus(InputWorkspace=opk, Filename=os.path.join(directory,opk+'.nxs'))

        if facility == 'SNS':
            if mtd.doesExist(ows):
                DeleteWorkspace(ows)
            if mtd.doesExist(omd):
                DeleteWorkspace(omd)
            if mtd.doesExist(opk):
                DeleteWorkspace(opk)

def partial_load(runs, banks, phi, chi, omega, proton_charge,
                 directory, facility, instrument, ipts, Q,
                 sample_radius=0.15, chemical_formula=None,
                 volume=0, z_parameter=1, exp=None):

    for i, (r, b, p, c, o) in enumerate(zip(runs, banks, phi, chi, omega)):
        print('Processing run : {}'.format(r))
        if facility == 'SNS':
            ows = '{}_{}_{}'.format(instrument,r,b)
        else:
            ows = '{}_{}_{}'.format(instrument,exp,r)

        omd = ows+'_md'

        if not mtd.doesExist(ows):

            if facility == 'SNS':
                filename = '/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(instrument,ipts,instrument,r)
                LoadEventNexus(Filename=filename, 
                               BankName='bank{}'.format(b), 
                               SingleBankPixelsOnly=True,
                               LoadLogs=False,
                               LoadNexusInstrumentXML=False,
                               OutputWorkspace=ows)

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
                             
                pc = proton_charge[r]

                AddSampleLog(Workspace=ows,
                             LogName='gd_prtn_chrg', 
                             LogText=str(pc),
                             LogType='Number',
                             LogUnit='uA.hour',
                             NumberType='Double')

                if mtd.doesExist('sa'):
                    CopyInstrumentParameters(InputWorkspace='sa', OutputWorkspace=ows)

                SetGoniometer(Workspace=ows, Goniometers='Universal')

                NormaliseByCurrent(InputWorkspace=ows, OutputWorkspace=ows)

                if chemical_formula is not None:
                    
                    SetSampleMaterial(InputWorkspace=ows,
                                      ChemicalFormula=chemical_formula,
                                      ZParameter=z_parameter,
                                      UnitCellVolume=volume)

                    AnvredCorrection(InputWorkspace=ows,
                                     OnlySphericalAbsorption=True,
                                     Radius=sample_radius,
                                     OutputWorkspace=ows)

                ConvertUnits(InputWorkspace=ows, OutputWorkspace=ows, EMode='Elastic', Target='Momentum')

                if instrument == 'CORELLI':
                    CropWorkspaceForMDNorm(InputWorkspace=ows, XMin=2.5, XMax=10, OutputWorkspace=ows)

        if facility == 'SNS':

            Qmin = [Qi-0.5 for Qi in Q]
            Qmax = [Qi+0.5 for Qi in Q]

            ConvertToMD(InputWorkspace=ows,
                        OutputWorkspace=omd,
                        QDimensions='Q3D',
                        dEAnalysisMode='Elastic',
                        Q3DFrames='Q_sample',
                        LorentzCorrection=False,
                        MinValues='{},{},{}'.format(*Qmin),
                        MaxValues='{},{},{}'.format(*Qmax),
                        PreprocDetectorsWS='-',
                        Uproj='1,0,0',
                        Vproj='0,1,0',
                        Wproj='0,0,1')   

def partial_cleanup(runs, banks, facility, instrument, runs_banks, key):

    for r, b in zip(runs, banks):

        if facility == 'SNS':
            ows = '{}_{}_{}'.format(instrument,r,b)
        else:
            ows = '{}_{}_{}'.format(instrument,exp,r)

        omd = ows+'_md'

        peak_keys = runs_banks[(r,b)]
        peak_keys.remove(key)
        runs_banks[(r,b)] = peak_keys
 
        if facility == 'SNS':
            if mtd.doesExist(omd):
                DeleteWorkspace(omd)

            if len(peak_keys) == 0:
                if mtd.doesExist(ows):
                    DeleteWorkspace(ows)

    return runs_banks

def set_instrument(instrument):

    tof_instruments = ['CORELLI', 'MANDI', 'TOPAZ']

    instrument = instrument.upper()

    if instrument == 'BL9':
        instrument = 'CORELLI'
    if instrument == 'BL11B':
        instrument = 'MANDI'
    if instrument == 'BL12':
        instrument = 'TOPAZ'

    if instrument == 'DEMAND':
        instrument = 'HB3A'
    if instrument == 'WAND2':
        instrument = 'HB2C'

    facility = 'SNS' if instrument in tof_instruments else 'HFIR'

    return facility, instrument

def integration_loop(keys, outname, ref_peak_dictionary, ref_dict, filename,
                     spectrum_file, counts_file, tube_calibration, detector_calibration,
                     directory, facility, instrument, ipts, runs,
                     split_angle, a, b, c, alpha, beta, gamma,
                     mod_vector_1, mod_vector_2, mod_vector_3, max_order,
                     sample_radius, chemical_formula, volume, z_parameter, experiment):

    load_normalization_calibration(facility, spectrum_file, counts_file,
                                   tube_calibration, detector_calibration)
                                             
    if facility == 'HFIR':
        ows = '{}_{}'.format(instrument,experiment)+'_{}'
    else:
        ows = '{}'.format(instrument)+'_{}'

    opk = ows+'_pk'
    omd = ows+'_md'
    
    LoadNexus(Filename=filename, OutputWorkspace='tmp')
    
    for r in runs:
        FilterPeaks(InputWorkspace='tmp', 
                    FilterVariable='RunNumber',
                    FilterValue=r,
                    Operator='=',
                    OutputWorkspace=opk.format(r))
                    
    proton_charge = {}
    
    if facility == 'SNS':
        LoadEmptyInstrument(InstrumentName=instrument, OutputWorkspace=instrument+'_empty')
        for r in runs:
            logfile = '/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(instrument,ipts,instrument,r)
            LoadNexusLogs(Workspace=instrument+'_empty', Filename=logfile, OverwriteLogs=True)
            proton_charge[r] = mtd[instrument+'_empty'].getRun().getPropertyAsSingleValueWithTimeAveragedMean('gd_prtn_chrg')

    peak_dictionary = PeakDictionary(a, b, c, alpha, beta, gamma)
    peak_dictionary.set_satellite_info(mod_vector_1, mod_vector_2, mod_vector_3, max_order)
    peak_dictionary.set_scale_constant(1e+6)

    for r in runs:
        peak_dictionary.add_peaks(opk.format(r))

    peak_dictionary.split_peaks(split_angle)
    peaks = peak_dictionary.to_be_integrated()

    peak_envelope = PeakEnvelope(directory+'/{}.pdf'.format(outname))
    peak_envelope.show_plots(False)

    if mtd.doesExist('sa'):

        banks = list(set(mtd['tmp'].column(13)))
        banks = [int(bank.replace('bank', '')) for bank in banks]

        for b in banks:
            CropToComponent(InputWorkspace='sa', 
                            ComponentNames='bank{}'.format(b),
                            OutputWorkspace='sa_{}'.format(b))
    
    runs_banks = {}
    
    for key in keys:
        
        key = tuple(key)

        peaks_list = peak_dictionary.peak_dict.get(key)

        redundancies = peaks[key]
       
        for j, redundancy in enumerate(redundancies):
            
            runs = peaks_list[j].get_run_numbers()
            banks = peaks_list[j].get_bank_numbers()
            
            for r, b in zip(runs, banks):
                
                if runs_banks.get((r,b)) is None:
                    runs_banks[(r,b)] = [key]
                else:
                    peak_keys = runs_banks[(r,b)]
                    peak_keys.append(key)
                    runs_banks[(r,b)] = peak_keys

    for i, key in enumerate(keys):

        key = tuple(key)

        print('Integrating peak : {}'.format(key))
        print(peak_dictionary.peak_dict.get(key))

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

            banks = peaks_list[j].get_bank_numbers()

            Q0 = peaks_list[j].get_Q()
            
            phi = peaks_list[j].get_phi_angles()
            chi = peaks_list[j].get_chi_angles()
            omega = peaks_list[j].get_omega_angles()

            partial_load(runs, banks, phi, chi, omega, proton_charge,
                         directory, facility, instrument, ipts, Q0,
                         sample_radius, chemical_formula,
                         volume, z_parameter, experiment)

            if fixed:

                ref_peak = ref_peaks[j]
                Q0 = ref_peak.get_Q()
                A = ref_peak.get_A()
                D, W = np.linalg.eig(A)
                D = np.diag(D)

                radii = 1/np.sqrt(np.diagonal(D)) 

                peak_fit, peak_bkg_ratio, peak_score2d = 0, 0, 0

                if np.isclose(np.abs(np.linalg.det(W)),1) and (radii < 0.3).all() and (radii > 0).all():

                    data = norm_integrator(peak_envelope, instrument, runs, banks, Q0, D, W)

                    peak_dictionary.integrated_result(key, Q0, A, peak_fit, peak_bkg_ratio, peak_score2d, data, j)

            else:

                remove = False

                Q, Qx, Qy, Qz, weights = box_integrator(instrument, runs, banks, proton_charge, Q0, binsize=0.005, radius=0.15, exp=experiment)

                center, variance, peak_fit, peak_bkg_ratio, sig_noise_ratio, peak_total_data_ratio = Q_profile(peak_envelope, key, Q, weights, 
                                                                                                               Q0, radius=0.15, bins=31)

                print('Peak-fit Q: {}'.format(peak_fit))
                print('Peak background ratio Q: {}'.format(peak_bkg_ratio))
                print('Signal-noise ratio Q: {}'.format(sig_noise_ratio))
                print('Peak-total to subtrated-data ratio Q: {}'.format(peak_total_data_ratio))

                if (sig_noise_ratio > 3 and 3*np.sqrt(variance) < 0.1 and np.abs(np.linalg.norm(Q0)-center) < 0.1):

                    remove = True

                n, u, v = projection_axes(Q0)

                center2d, covariance2d, peak_score2d, sig_noise_ratio2d = projected_profile(peak_envelope, d, Q, Qx, Qy, Qz, weights,
                                                                                            Q0, u, v, center, variance, radius=0.1,
                                                                                            bins=21, bins2d=21)

                print('Peak-score 2d: {}'.format(peak_score2d))
                print('Signal-noise ratio 2d: {}'.format(sig_noise_ratio2d))

                if (peak_score2d > 2 and not np.isinf(peak_score2d) and not np.isnan(peak_score2d) and np.linalg.norm(center2d) < 0.15 and sig_noise_ratio2d > 3):

                    remove = True

                Qc, A, W, D = ellipsoid(Q0, center, variance, center2d, covariance2d, 
                                        n, u, v, xsigma=4, lscale=5)

                peak_envelope.plot_projection_ellipse(*peak.draw_ellispoid(center2d, covariance2d, lscale=5))

                center, variance, peak_fit, peak_bkg_ratio, sig_noise_ratio, peak_total_data_ratio = extracted_Q_profile(peak_envelope, key, Q, Qx, Qy, Qz, weights, 
                                                                                                                         Q0, u, v, center, variance, center2d, covariance2d, bins=21)

                print('Peak-fit Q second pass: {}'.format(peak_fit))
                print('Peak background ratio Q second pass: {}'.format(peak_bkg_ratio))
                print('Signal-noise ratio Q second pass: {}'.format(sig_noise_ratio))
                print('Peak-total to subtrated-data ratio Q: {}'.format(peak_total_data_ratio))

                if (sig_noise_ratio > 3 and 3*np.sqrt(variance) < 0.1 and np.abs(np.linalg.norm(Qc)-center) < 0.1 and peak_total_data_ratio < 3):

                    remove = True

                if not np.isnan(covariance2d).any():

                    Q0, A, W, D = ellipsoid(Q0, center, variance, center2d, covariance2d, 
                                            n, u, v, xsigma=4, lscale=5)

                    radii = 1/np.sqrt(np.diagonal(D)) 

                    print('Peak-radii: {}'.format(radii))

                    if np.isclose(np.abs(np.linalg.det(W)),1) and (radii < 0.3).all() and (radii > 0).all() and not np.isclose(radii, 0).any():

                        data = norm_integrator(peak_envelope, facility, instrument, runs, banks, Q0, D, W, exp=experiment)

                        peak_dictionary.integrated_result(key, Q0, A, peak_fit, peak_bkg_ratio, peak_score2d, data, j)

                        peak_envelope.write_figure()

                    else:

                        remove = True

                if remove:

                    peak_dictionary.partial_result(key, Q0, A, peak_fit, peak_bkg_ratio, peak_score2d, j)

            runs_banks = partial_cleanup(runs, banks, facility, instrument, runs_banks, key)

        if i % 15 == 0:
            peak_dictionary.save_hkl(directory+'/{}.hkl'.format(outname))       
            peak_dictionary.save(directory+'/{}.pkl'.format(outname))

    peak_dictionary.save_hkl(directory+'/{}.hkl'.format(outname))       
    peak_dictionary.save(directory+'/{}.pkl'.format(outname))
    peak_envelope.create_pdf()