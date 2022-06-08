import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm

import numpy as np

import imp

import fitting
imp.reload(fitting)

from fitting import Ellipsoid, Profile, Projection

np.random.seed(13)

nx, ny, nz = 61, 41, 51

Qx_min, Qx_max = 0, 2
Qy_min, Qy_max = -1, 1
Qz_min, Qz_max = 1, 3

mu_x, mu_y, mu_z = 1.3, -0.4, 2.1

sigma_x, sigma_y, sigma_z = 0.1, 0.1, 0.1
rho_yz, rho_xz, rho_xy = 0.2, -0.4, -0.1

d = 5

b = 0.6
c = 0.2

cov = np.array([[sigma_x**2, rho_xy*sigma_x*sigma_y, rho_xz*sigma_x*sigma_z],
                [rho_xy*sigma_x*sigma_y, sigma_y**2, rho_yz*sigma_y*sigma_z],
                [rho_xz*sigma_x*sigma_z, rho_yz*sigma_y*sigma_z, sigma_z**2]])

Q0 = np.array([mu_x, mu_y, mu_z])

signal = np.random.multivariate_normal(Q0, cov, size=10000)
#a*np.sqrt((2*np.pi)**3*np.linalg.det(cov))

data_norm, (x_bin_edges, y_bin_edges, z_bin_edges) = np.histogramdd(signal, density=False, bins=[nx,ny,nz], range=[(Qx_min, Qx_max), (Qy_min, Qy_max), (Qz_min, Qz_max)])

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

data_norm *= 1

norm = 0.2*(np.ones(data_norm.size)+Qp**2)

#data_norm += d*np.random.random(nx*ny*nz)
#norm += d*np.random.random(nx*ny*nz)

data = data_norm*norm

ellip = Ellipsoid(Qx, Qy, Qz, data, norm, Q0, 0.5)
plt.close('all')

prof = Profile()

stats, params = prof.fit(ellip, 0.95)

chi_sq, peak_bkg_ratio, sig_noise_ratio = stats
a, mu, sigma = params

print(chi_sq, peak_bkg_ratio, sig_noise_ratio)

x = prof.x
y = prof.y

y_sub = prof.y_sub
y_fit = prof.y_fit

e = prof.e
e_sub = prof.e_sub

fig, ax = plt.subplots(num='Profile1')
ax.errorbar(x, y, e, fmt='-o')
ax.errorbar(x, y_sub, e_sub, fmt='-s')
ax.plot(x, y_fit, '--')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

ellip.mu = mu
ellip.sigma = sigma

proj = Projection()

stats, params = proj.fit(ellip, 0.99)

chi_sq, peak_bkg_ratio, sig_noise_ratio = stats
a, mu_x, mu_y, sigma_x, sigma_y, rho = params

print(chi_sq, peak_bkg_ratio, sig_noise_ratio)

x = proj.x
y = proj.y
z = proj.z

z_sub = proj.z_sub
z_fit = proj.z_fit

rx = np.sqrt(1+rho)
ry = np.sqrt(1-rho)

ellipse = Ellipse((0,0), width=2*rx, height=2*ry, edgecolor='r', facecolor='none')

scale_x = 2*sigma_x
scale_y = 2*sigma_y

transf = transforms.Affine2D().rotate_deg(45).scale(scale_x,scale_y).translate(mu_x,mu_y)

fig, ax = plt.subplots(num='Projection1')
im = ax.pcolormesh(x,y,z,linewidth=0,rasterized=True,cmap=plt.cm.viridis)
ax.set_xlabel('x')
ax.set_ylabel('y')
cb = fig.colorbar(im)
cb.ax.yaxis.set_label_text(r'$z$')
transf = transforms.Affine2D().rotate_deg(45).scale(scale_x,scale_y).translate(mu_x,mu_y)
ellipse.set_transform(transf+ax.transData)
ax.add_patch(ellipse)
plt.show()

ellip.mu_x, ellip.mu_y = mu_x, mu_y
ellip.sigma_x, ellip.sigma_x = sigma_x, sigma_y
ellip.rho = rho

prof = Profile()

stats, params = prof.fit(ellip, 0.99)

chi_sq, peak_bkg_ratio, sig_noise_ratio = stats
a, mu, sigma = params

print(chi_sq)

x = prof.x
y = prof.y

y_sub = prof.y_sub
y_fit = prof.y_fit

e = prof.e
e_sub = prof.e_sub

fig, ax = plt.subplots(num='Profile2')
ax.errorbar(x, y, e, fmt='-o')
ax.errorbar(x, y_sub, e_sub, fmt='-s')
ax.plot(x, y_fit, '--')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

ellip.mu = mu
ellip.sigma = sigma

proj = Projection()

stats, params = proj.fit(ellip, 0.99)

chi_sq, peak_bkg_ratio, sig_noise_ratio = stats
a, mu_x, mu_y, sigma_x, sigma_y, rho = params\

print(chi_sq)

x = proj.x
y = proj.y
z = proj.z

z_sub = proj.z_sub
z_fit = proj.z_fit

rx = np.sqrt(1+rho)
ry = np.sqrt(1-rho)

ellipse = Ellipse((0,0), width=2*rx, height=2*ry, edgecolor='r', facecolor='none')

scale_x = 2*sigma_x
scale_y = 2*sigma_y

transf = transforms.Affine2D().rotate_deg(45).scale(scale_x,scale_y).translate(mu_x,mu_y)

fig, ax = plt.subplots(num='Projection2')
im = ax.pcolormesh(x,y,z_sub,linewidth=0,rasterized=True,cmap=plt.cm.viridis)
ax.set_xlabel('x')
ax.set_ylabel('y')
cb = fig.colorbar(im)
cb.ax.yaxis.set_label_text(r'$z$')
transf = transforms.Affine2D().rotate_deg(45).scale(scale_x,scale_y).translate(mu_x,mu_y)
ellipse.set_transform(transf+ax.transData)
ax.add_patch(ellipse)
plt.show()