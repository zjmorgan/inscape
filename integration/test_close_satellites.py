import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm

import numpy as np

import imp

import fitting
imp.reload(fitting)

from fitting import Ellipsoid, Profile, Projection, LineCut

np.random.seed(13)

nx, ny, nz = 201, 201, 201

Qx_min, Qx_max = -0.25, 0.25
Qy_min, Qy_max = -0.25, 0.25
Qz_min, Qz_max = -0.25, 0.25

mu_x, mu_y, mu_z = 1, 1, 0

Qx_min += mu_x
Qy_min += mu_y
Qz_min += mu_z

Qx_max += mu_x
Qy_max += mu_y
Qz_max += mu_z

sigma_x, sigma_y, sigma_z = 0.015, 0.03, 0.04
rho_yz, rho_xz, rho_xy = 0.0, -0.0, -0.0

delta_mu_x, delta_mu_y, delta_mu_z = 0.0, 0.1, 0

d = 2

b = 5.0
c = 0.5

cov = np.array([[sigma_x**2, rho_xy*sigma_x*sigma_y, rho_xz*sigma_x*sigma_z],
                [rho_xy*sigma_x*sigma_y, sigma_y**2, rho_yz*sigma_y*sigma_z],
                [rho_xz*sigma_x*sigma_z, rho_yz*sigma_y*sigma_z, sigma_z**2]])

Q0 = np.array([mu_x, mu_y, mu_z])

size = 1000000

signal = np.random.multivariate_normal(Q0, cov, size=size)

data_norm, (x_bin_edges, y_bin_edges, z_bin_edges) = np.histogramdd(signal, density=False, bins=[nx,ny,nz], range=[(Qx_min, Qx_max), (Qy_min, Qy_max), (Qz_min, Qz_max)])

delta_Q0 = np.array([delta_mu_x, delta_mu_y, delta_mu_z])

signal_sat1 = np.random.multivariate_normal(Q0+delta_Q0, cov, size=size//2)
signal_sat2 = np.random.multivariate_normal(Q0-delta_Q0, cov, size=size//6)

data_norm_sat1, (x_bin_edges, y_bin_edges, z_bin_edges) = np.histogramdd(signal_sat1, density=False, bins=[nx,ny,nz], range=[(Qx_min, Qx_max), (Qy_min, Qy_max), (Qz_min, Qz_max)])
data_norm_sat2, (x_bin_edges, y_bin_edges, z_bin_edges) = np.histogramdd(signal_sat2, density=False, bins=[nx,ny,nz], range=[(Qx_min, Qx_max), (Qy_min, Qy_max), (Qz_min, Qz_max)])

data_norm += data_norm_sat1+data_norm_sat2

Qx = 0.5*(x_bin_edges[1:]+x_bin_edges[:-1])
Qy = 0.5*(y_bin_edges[1:]+y_bin_edges[:-1])
Qz = 0.5*(z_bin_edges[1:]+z_bin_edges[:-1])

data_norm += d*(2*np.random.random(data_norm.shape)-1)

Qx, Qy, Qz = np.meshgrid(Qx, Qy, Qz, indexing='ij')

mask = data_norm > 0

Qx, Qy, Qz = Qx[mask], Qy[mask], Qz[mask]
data_norm = data_norm[mask]

Qp = np.sqrt(Qx**2+Qy**2+Qz**2)

data_norm += b+c*Qp

data_norm *= 0.1

norm = 0.2*(np.ones(data_norm.size)+Qp**2)

#data_norm += d*np.random.random(nx*ny*nz)
#norm += d*np.random.random(nx*ny*nz)

data = data_norm*norm

ellip = Ellipsoid(Qx, Qy, Qz, data, norm, Q0, 0.8)
ellip.reset_axes(delta_Q0)

plt.close('all')

proj = Projection()

int_mask, bkg_mask = ellip.projection_mask()

dQ1, dQ2, data, norm = ellip.dQ1, ellip.dQ2, ellip.data, ellip.norm

stats, params = proj.fit(dQ1, dQ2, data, norm, int_mask, bkg_mask, 0.99)

chi_sq, peak_bkg_ratio, sig_noise_ratio = stats
a, mu_x, mu_y, sigma_x, sigma_y, rho = params

# print(chi_sq, peak_bkg_ratio, sig_noise_ratio)

x = proj.x
y = proj.y
z = proj.z

z_sub = proj.z_sub
z_fit = proj.z_fit

e_sub = proj.e_sub

rx = np.sqrt(1+rho)
ry = np.sqrt(1-rho)

ellipse = Ellipse((0,0), width=2*rx, height=2*ry, edgecolor='r', facecolor='none')

scale_x = 2*sigma_x
scale_y = 2*sigma_y

transf = transforms.Affine2D().rotate_deg(45).scale(scale_x,scale_y).translate(mu_x,mu_y)

fig, ax = plt.subplots(num='Projection1')
im = ax.pcolormesh(x,y,z_sub,linewidth=0,rasterized=True,cmap=plt.cm.viridis)
ax.set_xlabel('$\Delta{Q}_1$')
ax.set_ylabel('$\Delta{Q}_2$')
cb = fig.colorbar(im)
cb.ax.yaxis.set_label_text(r'$I$')
transf = transforms.Affine2D().rotate_deg(45).scale(scale_x,scale_y).translate(mu_x,mu_y)
ellipse.set_transform(transf+ax.transData)
ax.add_patch(ellipse)
plt.show()

prof = LineCut()

int_mask, bkg_mask = ellip.profile_mask()
Qp, data, norm = ellip.Qp, ellip.data, ellip.norm
stats, params = prof.fit(Qp, data, norm, int_mask, bkg_mask, 0.99)

chi_sq, peak_bkg_ratio, sig_noise_ratio = stats
#a, mu, sigma = params

#print(chi_sq, peak_bkg_ratio, sig_noise_ratio)

x = prof.x
y = prof.y

y_sub = prof.y_sub
y_bkg = prof.y_bkg
y_fit = prof.y_fit

e = prof.e
e_sub = prof.e_sub

fig, ax = plt.subplots(num='Profile1')
ax.errorbar(x, y, e, fmt='-o')
ax.errorbar(x, y_sub, e_sub, fmt='-s')
ax.plot(x, y_fit, '--')
ax.plot(x, y_bkg, '--')
ax.set_xlabel('$Q_p$')
ax.set_ylabel('$I$')
plt.show()

#ellip.mu = mu
#ellip.sigma = sigma