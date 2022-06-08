import numpy as np

class Ellipsoid:

    def __init__(self, Qx, Qy, Qz, data, norm, Q0, size=1, n_std=3, scale=3):

        self.Q0 = Q0

        self.n, self.mu = self.profile_axis(self.Q0)
        self.u, self.v = self.projection_axes(self.n)
        
        self.size=size

        self.n_std, self.scale = n_std, scale

        self.sigma, self.sigma_x, self.sigma_y = size/n_std, size/scale, size/scale

        self.mu_x, self.mu_y, self.rho = 0, 0, 0
        
        u, v, n = self.u, self.v, self.n
        
        self.dQ1 = u[0]*(Qx-Q0[0])+u[1]*(Qy-Q0[1])+u[2]*(Qz-Q0[2])
        self.dQ2 = v[0]*(Qx-Q0[0])+v[1]*(Qy-Q0[1])+v[2]*(Qz-Q0[2])

        self.Qp = Qx*n[0]+Qy*n[1]+Qz*n[2]

        self.Qx, self.Qy, self.Qz = Qx.copy(), Qy.copy(), Qz.copy()

        self.data = data.copy()
        self.norm = norm.copy()
          
    def profile_axis(self, Q0):

        n = Q0/np.linalg.norm(Q0)

        return n, np.dot(Q0, n)

    def projection_axes(self, n):

        n_ind = np.argmin(np.abs(n))

        u = np.zeros(3)
        u[n_ind] = 1

        u = np.cross(n, u)
        u /= np.linalg.norm(u)

        v = np.cross(n, u)
        v *= np.sign(np.dot(np.cross(u, n), v))

        return u, v

    def ellipsoid(self, mask=False):

        mu, sigma = self.mu, self.sigma
        mu_x, mu_y, sigma_x, sigma_y, rho = self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho

        size = self.size

        if mask:
            n_std, scale = self.n_std+2, self.scale+1
        else:
            n_std, scale = self.n_std, self.scale

        n, u, v = self.n, self.u, self.v

        Q0 = self.Q0

        Qp = np.dot(Q0, n)

        Q = Q0+(mu-Qp)*n+u*mu_x+v*mu_y

        rp = n_std*sigma

        cov = np.array([[sigma_x**2, rho*sigma_x*sigma_y],
                        [rho*sigma_x*sigma_y, sigma_y**2]])

        vals, vecs = np.linalg.eig(cov)

        radii = scale*np.sqrt(vals)

        if mask:
            radii[radii > size] = size
            if rp > size: 
                rp = size

        up = u*vecs[0,0]+v*vecs[1,0]
        vp = u*vecs[0,1]+v*vecs[1,1]

        # wu, wv = up, vp
        # ru, rv = radii

        if radii[0] < radii[1]:
            wu, wv = up, vp
            ru, rv = radii
        else:
            wu, wv = vp, up
            ru, rv = radii[::-1]

        W = np.column_stack((wu,wv,n))

        D = np.diag([1/ru**2,1/rv**2,1/rp**2])

        return Q, W, D

    def A(self, W, D):

        return np.dot(np.dot(W, D), W.T)

    def mask(self):

        Q, W, D = self.ellipsoid(mask=True)

        A = self.A(W, D)

        Qx, Qy, Qz = self.Qx, self.Qy, self.Qz

        dQx, dQy, dQz = Qx-Q[0], Qy-Q[1], Qz-Q[2]

        mask = (A[0,0]*dQx+A[0,1]*dQy+A[0,2]*dQz)*dQx+\
               (A[1,0]*dQx+A[1,1]*dQy+A[1,2]*dQz)*dQy+\
               (A[2,0]*dQx+A[2,1]*dQy+A[2,2]*dQz)*dQz <= 1

        return mask

def bin_size(vals, weights):

    bins = 25

    if weights.sum() > 0:

        mu = np.average(vals, weights=weights**2)
        sigma = np.sqrt(np.average((vals-mu)**2, weights=weights**2))

        bin_size = 3.5*sigma/bins
        vrange = vals.max()-vals.min()

        if bin_size > 0 and not np.isclose(vrange,0):

            bins = np.int(np.ceil(vrange/bin_size))

    return bins

class Profile:

    def __init__(self):

        self.a = 0

        self.x, self.y, self.e = None, None, None

        self.y_sub, self.e_sub = None, None
        self.y_fit = None

    def gaussian(self, x, a, mu, sigma):

        return a*np.exp(-0.5*(x-mu)**2/sigma**2)

    def linear(self, x, a, b):

        return a+b*x

    def histogram(self, x, data, norm, mask, bkg_scale=0.95):

        data_norm = data/norm
        data_norm[data_norm < 0] = 0

        bins = bin_size(x, data_norm)
        bin_edges = np.histogram_bin_edges(x, bins=bins)

        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])

        bin_data, _ = np.histogram(x[~mask], bins=bin_edges, weights=data[~mask])
        bin_norm, _ = np.histogram(x[~mask], bins=bin_edges, weights=norm[~mask])

        bin_data_norm = bin_data/bin_norm
        bin_err = np.sqrt(bin_data)/bin_norm

        a, b = self.background(bin_centers, bin_data_norm, bin_err)
 
        bin_bkg = bkg_scale*self.linear(bin_centers, a, b)*bin_norm

        bin_data, _ = np.histogram(x, bins=bin_edges, weights=data)
        bin_norm, _ = np.histogram(x, bins=bin_edges, weights=norm)

        bin_bkg[bin_bkg < 0] = 0

        bin_data_norm = bin_data/bin_norm
        bin_err = np.sqrt(bin_data)/bin_norm

        bin_data_norm_sub = (bin_data-bin_bkg)/bin_norm
        bin_err_sub = np.sqrt(bin_data+bin_bkg)/bin_norm

        bin_data_norm_sub[bin_data_norm_sub < 0] = 0

        return bin_centers, bin_data_norm, bin_err, bin_data_norm_sub, bin_err_sub

    def peak(self, x, y, e, y_fit):

        mask = (y > 0) & (e > 0) & (y_fit > 0) & (y < np.inf) & (e < np.inf)

        A = (np.array([x[mask]*0+1, x[mask], x[mask]**2])*y_fit[mask]**2/e[mask]**2).T
        B = np.log(y[mask])*y_fit[mask]**2/e[mask]**2

        if y[mask].size > 4:
            coeff, r, rank, s = np.linalg.lstsq(A, B)
        else:
            coeff = np.array([0, 0, -0.5])

        return self.parameters(coeff)

    def background(self, x, y, e):

        mask = (y > 0) & (e > 0) & (y < np.inf) & (e < np.inf)

        A = (np.array([x[mask]*0+1, x[mask]])*y[mask]**2/e[mask]**2).T
        B = y[mask]*y[mask]**2/e[mask]**2

        if y[mask].size > 3:
            coeff, r, rank, s = np.linalg.lstsq(A, B)
        else:
            coeff = np.array([0, 0])

        return tuple(coeff)

    def parameters(self, coeff):

        mu = -0.5*coeff[1]/coeff[2]
        sigma = np.sqrt(-0.5/coeff[2])
        a = np.exp(coeff[0]-0.25*coeff[1]**2/coeff[2])

        return a, mu, sigma

    def statistics(self, x, y, e, y_fit):

        mask = (y > 0) & (e > 0) & (y_fit > 0) & (y < np.inf) & (e < np.inf)

        chi_sq = np.sum((y[mask]-y_fit[mask])**2)/(y[mask].size-4) 

        peak_bkg_ratio = np.std(y[mask])/np.median(e[mask])

        sig_noise_ratio = np.sum(y[mask])/np.sqrt(np.sum(e[mask]**2))

        return chi_sq, peak_bkg_ratio, sig_noise_ratio

    def fit(self, ellip, bkg_scale=0.95):

        data = ellip.data.copy()
        norm = ellip.norm.copy()

        x = ellip.Qp.copy()
        y = data/norm
        e = np.sqrt(data)/norm

        mask = ellip.mask()

        xh, yh, eh, y_sub, e_sub = self.histogram(x, data, norm, mask, bkg_scale)

        y_fit = y_sub.copy()

        for _ in range(3):
            a, mu, sigma = self.peak(xh, y_sub, e_sub, y_fit)
            y_fit = self.gaussian(xh, a, mu, sigma)

        self.a = a

        self.x, self.y, self.e = xh.copy(), yh.copy(), eh.copy()

        self.y_sub, self.e_sub = y_sub.copy(), e_sub.copy()
        self.y_fit = y_fit.copy()

        return self.statistics(xh, y_sub, e_sub, y_fit), (a, mu, sigma)

class Projection:

    def __init__(self):

        self.a = 0

        self.x, self.y, self.z, self.e = None, None, None, None

        self.z_sub, self.e_sub = None, None
        self.z_fit = None

    def gaussian(self, x, y, a, mu_x, mu_y, sigma_x, sigma_y, rho):

        return a*np.exp(-0.5/(1-rho**2)*((x-mu_x)**2/sigma_x**2+(y-mu_y)**2/sigma_y**2-2*rho*(x-mu_x)*(y-mu_y)/(sigma_x*sigma_y)))

    def linear(self, x, y, a, b, c):

        return a+b*x+c*y

    def histogram(self, x, y, data, norm, mask, bkg_scale=0.95):
        
        data_norm = data/norm
        data_norm[data_norm < 0] = 0
        
        bins_x = bin_size(x, data_norm)
        bin_edges_x = np.histogram_bin_edges(x, bins=bins_x)    
        bin_centers_x = 0.5*(bin_edges_x[1:]+bin_edges_x[:-1])

        bins_y = bin_size(y, data_norm)
        bin_edges_y = np.histogram_bin_edges(y, bins=bins_y)
        bin_centers_y = 0.5*(bin_edges_y[1:]+bin_edges_y[:-1])

        bin_centers_x, bin_centers_y = np.meshgrid(bin_centers_x, bin_centers_y, indexing='ij')

        bin_data, _, _ = np.histogram2d(x[~mask], y[~mask], bins=[bin_edges_x,bin_edges_y], weights=data[~mask])
        bin_norm, _, _ = np.histogram2d(x[~mask], y[~mask], bins=[bin_edges_x,bin_edges_y], weights=norm[~mask])

        bin_data_norm = bin_data/bin_norm
        bin_err = np.sqrt(bin_data)/bin_norm

        a, b, c = self.background(bin_centers_x, bin_centers_y, bin_data_norm, bin_err)
 
        bin_bkg = bkg_scale*self.linear(bin_centers_x, bin_centers_y, a, b, c)*bin_norm

        bin_data, _, _ = np.histogram2d(x, y, bins=[bin_edges_x,bin_edges_y], weights=data)
        bin_norm, _, _ = np.histogram2d(x, y, bins=[bin_edges_x,bin_edges_y], weights=norm)
        
        bin_bkg[bin_bkg < 0] = 0
        
        bin_data_norm = bin_data/bin_norm
        bin_err = np.sqrt(bin_data)/bin_norm

        bin_data_norm_sub = (bin_data-bin_bkg)/bin_norm
        bin_err_sub = np.sqrt(bin_data+bin_bkg)/bin_norm

        bin_data_norm_sub[bin_data_norm_sub < 0] = 0

        return bin_centers_x, bin_centers_y, bin_data_norm, bin_err, bin_data_norm_sub, bin_err_sub

    def peak(self, x, y, z, e, z_fit):

        mask = (z > 0) & (e > 0) & (z_fit > 0) & (z < np.inf) & (e < np.inf)

        A = (np.array([x[mask]*0+1, x[mask], y[mask], x[mask]*y[mask], x[mask]**2, y[mask]**2])*z_fit[mask]**2/e[mask]**2).T
        B = np.log(z[mask])*z_fit[mask]**2/e[mask]**2
        
        if z[mask].size > 7:
            coeff, r, rank, s = np.linalg.lstsq(A, B)
        else:
            coeff = np.array([0, 0, 0, 0, -0.5, -0.5])

        return self.parameters(coeff)

    def background(self, x, y, z, e):

        mask = (z > 0) & (e > 0) & (z < np.inf) & (e < np.inf)

        A = (np.array([x[mask]*0+1, x[mask], y[mask]])*z[mask]**2/e[mask]**2).T
        B = z[mask]*z[mask]**2/e[mask]**2

        if z[mask].size > 4:
            coeff, r, rank, s = np.linalg.lstsq(A, B)
        else:
            coeff = np.array([0, 0, 0])
            
        return tuple(coeff)

    def parameters(self, coeff):

        sigma_x = np.sqrt(2*np.abs(coeff[5]/(4*coeff[4]*coeff[5]-coeff[3]**2)))
        sigma_y = np.sqrt(2*np.abs(coeff[4]/(4*coeff[4]*coeff[5]-coeff[3]**2)))

        c = sigma_x*sigma_y*coeff[3]

        rho = 0 if np.isclose(c,0) else (np.sqrt(4*c**2+1)-1)/(2*c)

        mu_x = sigma_x*(sigma_x*coeff[1]+rho*sigma_y*coeff[2])
        mu_y = sigma_y*(sigma_y*coeff[2]+rho*sigma_x*coeff[1])

        a = np.exp(coeff[0]+0.5/(1-rho**2)*(mu_x**2/sigma_x**2+mu_y**2/sigma_y**2-2*rho*mu_x*mu_y/sigma_x/sigma_y))

        return a, mu_x, mu_y, sigma_x, sigma_y, rho

    def statistics(self, x, y, z, e, z_fit):

        mask = (z > 0) & (e > 0) & (z_fit > 0) & (z < np.inf) & (e < np.inf)

        chi_sq = np.sum((z[mask]-z_fit[mask])**2)/(z[mask].size-6) 

        peak_bkg_ratio = np.std(z[mask])/np.median(e[mask])

        sig_noise_ratio = np.sum(z[mask])/np.sqrt(np.sum(e[mask]**2))

        return chi_sq, peak_bkg_ratio, sig_noise_ratio

    def fit(self, ellip, bkg_scale=0.95):

        data = ellip.data.copy()
        norm = ellip.norm.copy()

        x = ellip.dQ1.copy()
        y = ellip.dQ2.copy()
        z = data/norm
        e = np.sqrt(data)/norm

        mask = ellip.mask()

        xh, yh, zh, eh, z_sub, e_sub = self.histogram(x, y, data, norm, mask, bkg_scale)

        z_fit = z_sub.copy()

        for _ in range(3):
            a, mu_x, mu_y, sigma_x, sigma_y, rho = self.peak(xh, yh, z_sub, e_sub, z_fit)
            z_fit = self.gaussian(xh, yh, a, mu_x, mu_y, sigma_x, sigma_y, rho)

        self.a = a

        self.x, self.y, self.z, self.e = xh.copy(), yh.copy(), zh.copy(), eh.copy()

        self.z_sub, self.e_sub = z_sub.copy(), e_sub.copy()
        self.z_fit = z_fit.copy()

        return self.statistics(xh, yh, z_sub, e_sub, z_fit), (a, mu_x, mu_y, sigma_x, sigma_y, rho)