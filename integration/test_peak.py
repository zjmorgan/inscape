import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm

import scipy.special

import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python38.zip')
sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python3.8')
sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python3.8/lib-dynload')
sys.path.append('/opt/anaconda/envs/scd-reduction-tools-dev/lib/python3.8/site-packages')

import numpy as np

import imp

import peak
imp.reload(peak)

from peak import PeakInformation
from fitting import GaussianFit3D

np.random.seed(13)

nx, ny, nz = 11, 10, 9

mu_x, mu_y, mu_z = 0.05, 0.07, -0.02

sigma_x, sigma_y, sigma_z = 0.12, 0.13, 0.2
rho_yz, rho_xz, rho_xy = 0.7, -0.8, -0.9

Qx_min = mu_x-6*sigma_x
Qy_min = mu_y-6*sigma_y
Qz_min = mu_z-6*sigma_z

Qx_max = mu_x+6*sigma_x
Qy_max = mu_y+6*sigma_y
Qz_max = mu_z+6*sigma_z

cov = np.array([[sigma_x**2, rho_xy*sigma_x*sigma_y, rho_xz*sigma_x*sigma_z],
                [rho_xy*sigma_x*sigma_y, sigma_y**2, rho_yz*sigma_y*sigma_z],
                [rho_xz*sigma_x*sigma_z, rho_yz*sigma_y*sigma_z, sigma_z**2]])

Q0 = np.array([mu_x, mu_y, mu_z])

size = 100000

signal = np.random.multivariate_normal(Q0, cov, size=size)

print(signal.shape)

x_bin_edges = np.histogram_bin_edges(signal[:,0], bins=nx, range=(Qx_min,Qx_max))
y_bin_edges = np.histogram_bin_edges(signal[:,1], bins=ny, range=(Qy_min,Qy_max))
z_bin_edges = np.histogram_bin_edges(signal[:,2], bins=nz, range=(Qz_min,Qz_max))

#data, (x_bin_edges, y_bin_edges, z_bin_edges) = np.histogramdd(signal, bins=[nx,ny,nz], range=[(Qx_min, Qx_max), (Qy_min, Qy_max), (Qz_min, Qz_max)])
data, _ = np.histogramdd(signal, bins=[x_bin_edges,y_bin_edges,z_bin_edges])

x_bin_centers = 0.5*(x_bin_edges[1:]+x_bin_edges[:-1])
y_bin_centers = 0.5*(y_bin_edges[1:]+y_bin_edges[:-1])
z_bin_centers = 0.5*(z_bin_edges[1:]+z_bin_edges[:-1])

bin_size = np.diff(x_bin_centers).mean(), np.diff(y_bin_centers).mean(), np.diff(z_bin_centers).mean()

x_bin, y_bin, z_bin = np.meshgrid(x_bin_centers, y_bin_centers, z_bin_centers, indexing='ij')

inv_cov = np.linalg.inv(cov)

norm = np.sqrt(np.linalg.det(2*np.pi*cov))

signal = np.exp(-0.5*(inv_cov[0,0]*(x_bin-mu_x)**2+inv_cov[1,1]*(y_bin-mu_y)**2+inv_cov[2,2]*(z_bin-mu_z)**2\
                  +2*(inv_cov[1,2]*(y_bin-mu_y)*(z_bin-mu_z)+inv_cov[0,2]*(x_bin-mu_x)*(z_bin-mu_z)+inv_cov[0,1]*(x_bin-mu_x)*(y_bin-mu_y))))/norm

#signal = np.exp(-0.5*(inv_cov[1,1]*(y_bin-mu_y)**2+inv_cov[2,2]*(z_bin-mu_z)**2+2*inv_cov[1,2]*(y_bin-mu_y)*(z_bin-mu_z)))
#signal *= np.exp(-0.5*(x_bin-mu_x)**2/sigma_x**2)*(1+scipy.special.erf(-4*(x_bin-mu_x)/sigma_x))

signal *= 10
signal += 3+x_bin+y_bin+2*z_bin

signal += 2*np.random.random(signal.shape)

data = signal**2*10000

data[np.random.random(signal.shape) < 0.1] = np.nan
#data[data.shape[0]//2:,:,:] = np.nan

norm = data/signal

error = np.sqrt(data)/norm

pk = PeakInformation(1)

D = np.diag(1/np.array([3*sigma_x,3*sigma_y,3*sigma_z])**2)
W = np.eye(3)

statistics = 1, 3, 3, 1, 3, 3

print(1/np.sqrt(D.diagonal()), [3*sigma_x,3*sigma_y,3*sigma_z])

D_pk = D/1.26**2
D_bkg_in = D/1.59**2
D_bkg_out = D/2.0**2

mask = D_pk[0,0]*(x_bin-Q0[0])**2\
     + D_pk[1,1]*(y_bin-Q0[1])**2\
     + D_pk[2,2]*(z_bin-Q0[2])**2 <= 1

pk_data = [data[mask].copy()]
pk_norm = [norm[mask].copy()]

pk_Q0, pk_Q1, pk_Q2 = x_bin[mask], y_bin[mask], z_bin[mask]

mask = (D_bkg_in[0,0]*(x_bin-Q0[0])**2\
       +D_bkg_in[1,1]*(y_bin-Q0[1])**2\
       +D_bkg_in[2,2]*(z_bin-Q0[2])**2 > 1)\
     & (D_bkg_out[0,0]*(x_bin-Q0[0])**2\
       +D_bkg_out[1,1]*(y_bin-Q0[1])**2\
       +D_bkg_out[2,2]*(z_bin-Q0[2])**2 <= 1)

bkg_data = [data[mask].copy()]
bkg_norm = [norm[mask].copy()]

bkg_Q0, bkg_Q1, bkg_Q2 = x_bin[mask], y_bin[mask], z_bin[mask]

pk_bkg = pk_data, pk_norm, bkg_data, bkg_norm, bin_size
data_norm = x_bin, y_bin, z_bin, data, norm
cntrs = pk_Q0, pk_Q1, pk_Q2, bkg_Q0, bkg_Q1, bkg_Q2

pk.set_peak_number(1)
pk.add_information(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
pk.add_integration(Q0, D, W, statistics, data_norm, pk_bkg, cntrs)

#print((np.nansum(pk_data[0]/pk_norm[0])-np.isfinite(pk_data).size/np.isfinite(bkg_data).size*np.nansum(bkg_data[0]/bkg_norm[0]))*np.prod(bin_size))

print(pk.get_merged_intensity())
print(pk.get_intensity())

mask = np.isfinite(data)

y = signal[mask]
e = error[mask]

x = (x_bin[mask], y_bin[mask], z_bin[mask])

mu = [mu_x, mu_y, mu_z]
sigma = [sigma_x, sigma_y, sigma_z]

peak_fit_3d = GaussianFit3D(x, y, e, mu, sigma)

A, B, C0, C1, C2, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01, boundary = peak_fit_3d.fit()

fit_1d = [mu_x, sigma_x]
fit_2d = [mu_y, mu_z, sigma_y, sigma_z, 0]
fit_3d = [mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01]

pk.add_fit(fit_1d, fit_2d, fit_3d, 0)

A, B, C0, C1, C2 = pk.integrate()

print(A, B, C0, C1, C2)

print(pk.get_fitted_intensity())
print(pk.get_individual_fitted_intensity())