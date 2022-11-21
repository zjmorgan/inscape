import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm

import numpy as np

import imp

import fitting
imp.reload(fitting)

from fitting import GaussianFit3D

np.random.seed(13)

nx, ny, nz = 101, 101, 101

Qx_min, Qx_max = -0.5, 0.5
Qy_min, Qy_max = -0.5, 0.5
Qz_min, Qz_max = -0.5, 0.5

mu_x, mu_y, mu_z = 0.05, 0.07, -0.02

Qx_min += mu_x
Qy_min += mu_y
Qz_min += mu_z

Qx_max += mu_x
Qy_max += mu_y
Qz_max += mu_z

sigma_x, sigma_y, sigma_z = 0.12, 0.13, 0.1
rho_yz, rho_xz, rho_xy = 0.14, -0.5, -0.25

cov = np.array([[sigma_x**2, rho_xy*sigma_x*sigma_y, rho_xz*sigma_x*sigma_z],
                [rho_xy*sigma_x*sigma_y, sigma_y**2, rho_yz*sigma_y*sigma_z],
                [rho_xz*sigma_x*sigma_z, rho_yz*sigma_y*sigma_z, sigma_z**2]])

Q0 = np.array([mu_x, mu_y, mu_z])

size = 100000

signal = np.random.multivariate_normal(Q0, cov, size=size)

data, (x_bin_edges, y_bin_edges, z_bin_edges) = np.histogramdd(signal, density=False, bins=[nx,ny,nz], range=[(Qx_min, Qx_max), (Qy_min, Qy_max), (Qz_min, Qz_max)])

x_bin_centers = 0.5*(x_bin_edges[1:]+x_bin_edges[:-1])
y_bin_centers = 0.5*(y_bin_edges[1:]+y_bin_edges[:-1])
z_bin_centers = 0.5*(z_bin_edges[1:]+z_bin_edges[:-1])

x_bin, y_bin, z_bin = np.meshgrid(x_bin_centers, y_bin_centers, z_bin_centers, indexing='ij')

data /= data.max()
data += 1

error = np.sqrt(data)

y = data.flatten()
e = error.flatten()

x = (x_bin.flatten(), y_bin.flatten(), z_bin.flatten())

mu = [0, mu_y, mu_z]
sigma = [sigma_x, sigma_y, sigma_z]

peak_fit_3d = GaussianFit3D(x, y, e, mu, sigma)

A, B, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01, boundary = peak_fit_3d.fit()

print(A, B, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01, boundary)

A, B, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01, boundary = peak_fit_3d.estimate()

print(A, B, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01, boundary)

