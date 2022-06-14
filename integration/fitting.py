import numpy as np
import scipy.optimize

class Ellipsoid:

    def __init__(self, Qx, Qy, Qz, data, norm, Q0, size=1, n_std=3, scale=3, rotation=False):

        self.Q0 = Q0

        self.n, self.mu = self.profile_axis(self.Q0, rotation)
        self.u, self.v = self.projection_axes(self.n)

        self.size = size

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

    def profile_axis(self, Q0, rotation=False):

        if rotation:
            k = np.cross([0,1,0],Q0)
            n = k/np.linalg.norm(k)
        else:
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

        if np.abs(u[1]) > np.abs(v[1]):
            u, v = v, -u

        return u, v

    def ellipsoid(self):

        mu, sigma = self.mu, self.sigma
        mu_x, mu_y, sigma_x, sigma_y, rho = self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho

        size = self.size

        # if mtype == 'profile':
        #     n_std, scale = self.n_std+1, self.scale
        # elif mtype == 'projection':
        #     n_std, scale = self.n_std, self.scale

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

        # if mtype is not None:
        #     radii[radii > size] = size
        #     if rp > size: 
        #         rp = size

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

        D = np.diag(1/np.array([ru,rv,rp])**2)

        return Q, W, D

    def A(self, W, D):

        return np.dot(np.dot(W, D), W.T)

    def mask(self):

        Q, W, D = self.ellipsoid()

        A = self.A(W, D)
        
        size = self.size

        Qx, Qy, Qz = self.Qx, self.Qy, self.Qz

        dQx, dQy, dQz = Qx-Q[0], Qy-Q[1], Qz-Q[2]

        mask = dQx**2+dQy**2+dQz**2 < size**2

        mask = (A[0,0]*dQx+A[0,1]*dQy+A[0,2]*dQz)*dQx+\
               (A[1,0]*dQx+A[1,1]*dQy+A[1,2]*dQz)*dQy+\
               (A[2,0]*dQx+A[2,1]*dQy+A[2,2]*dQz)*dQz <= 1

        return mask

    def profile_mask(self):
        
        size = self.size

        mu, sigma = self.mu, self.sigma

        n_std = self.n_std+1.5

        Qp = self.Qp

        dQp = np.abs(Qp-mu)

        radius = n_std*sigma

        if radius > size:
            radius = size

        mu_x, mu_y, sigma_x, sigma_y, rho = self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho

        scale = self.scale

        dQ1, dQ2 = self.dQ1-mu_x, self.dQ2-mu_y

        cov = np.array([[sigma_x**2, rho*sigma_x*sigma_y],
                        [rho*sigma_x*sigma_y, sigma_y**2]])

        vals, vecs = np.linalg.eig(cov)

        radii = scale*np.sqrt(vals)
        radii[radii > size] = size

        W = vecs.copy()
        D = np.diag(1/radii**2)

        A = np.dot(np.dot(W, D), W.T)

        int_mask = (dQp < radius) & (A[0,0]*dQ1**2+A[1,1]*dQ2**2+2*A[0,1]*dQ1*dQ2 < 1)
        bkg_mask = (dQp > radius) & (A[0,0]*dQ1**2+A[1,1]*dQ2**2+2*A[0,1]*dQ1*dQ2 < 1)

        return int_mask, bkg_mask

    def projection_mask(self):
        
        size = self.size

        mu, sigma = self.mu, self.sigma

        n_std = self.n_std

        Qp = self.Qp

        dQp = np.abs(Qp-mu)

        radius = n_std*sigma

        if radius > size:
            radius = size

        mu_x, mu_y, sigma_x, sigma_y, rho = self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho

        scale = self.scale#+1.5

        dQ1, dQ2 = self.dQ1-mu_x, self.dQ2-mu_y

        cov = np.array([[sigma_x**2, rho*sigma_x*sigma_y],
                        [rho*sigma_x*sigma_y, sigma_y**2]])

        vals, vecs = np.linalg.eig(cov)

        radii = scale*np.sqrt(vals)
        radii[radii > size] = size

        W = vecs.copy()
        D = np.diag(1/radii**2)

        A = np.dot(np.dot(W, D), W.T)
        
        Q = self.Q0
        
        Qx, Qy, Qz = self.Qx, self.Qy, self.Qz

        dQx, dQy, dQz = Qx-Q[0], Qy-Q[1], Qz-Q[2]

        int_mask = (dQp < radius) & (A[0,0]*dQ1**2+A[1,1]*dQ2**2+2*A[0,1]*dQ1*dQ2 < 1)
        bkg_mask = (dQp < radius) & (A[0,0]*dQ1**2+A[1,1]*dQ2**2+2*A[0,1]*dQ1*dQ2 > 1)

        return int_mask, bkg_mask

def bin_size(val, mask, weights):

    bins = 51

    if weights[mask].sum() > 0:

        mu = np.average(val[mask], weights=weights[mask]**2)
        sigma = np.sqrt(np.average((val[mask]-mu)**2, weights=weights[mask]**2))

        bin_size = 3.5*sigma/bins
        val_range = val.max()-val.min()

        if bin_size > 0 and not np.isclose(val_range,0):

            bins = np.min([np.int(np.ceil(val_range/bin_size)),101])

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

    def histogram(self, x, data, norm, ellip, bkg_scale=0.95):
        
        int_mask, bkg_mask = ellip.profile_mask()

        data_norm = data/norm
        data_norm[data_norm < 0] = 0

        bins = bin_size(x, int_mask, data_norm)
        bin_edges = np.histogram_bin_edges(x, bins=bins)

        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])

        bin_data, _ = np.histogram(x[bkg_mask], bins=bin_edges, weights=data[bkg_mask])
        bin_norm, _ = np.histogram(x[bkg_mask], bins=bin_edges, weights=norm[bkg_mask])

        bin_data[np.isclose(bin_data, 0)] = np.nan

        mask = np.isinf(bin_norm) | np.isnan(bin_norm)

        bin_norm[:-1][np.cumsum(mask[1:]) == 1] = np.nan
        bin_norm[1:][np.cumsum(mask[::-1][1:])[::-1] == 1] = np.nan

        bin_data_norm = bin_data/bin_norm
        bin_err = np.sqrt(bin_data)/bin_norm

        a, b = self.background(bin_centers, bin_data_norm, bin_err)

        mask = (bin_data_norm > 0) & (bin_data_norm < np.inf)

        bin_bkg = bkg_scale*self.linear(bin_centers, a, b)
        bin_bkg_err = np.sqrt(np.average((bin_data_norm[mask]-bin_bkg[mask])**2))

        bin_data, _ = np.histogram(x[int_mask], bins=bin_edges, weights=data[int_mask])
        bin_norm, _ = np.histogram(x[int_mask], bins=bin_edges, weights=norm[int_mask])

        bin_norm[np.isclose(bin_norm, 0)] = np.nan

        mask = np.isinf(bin_norm) | np.isnan(bin_norm)
        mask = (np.cumsum(~mask) == 1) | (np.cumsum(~mask[::-1]) == 1)[::-1]

        bin_norm[mask] = np.nan

        bin_data_norm = bin_data/bin_norm
        bin_err = np.sqrt(bin_data)/bin_norm

        bin_data_norm_sub = bin_data/bin_norm-bin_bkg
        bin_err_sub = np.sqrt(bin_data/bin_norm**2+bin_bkg_err**2)

        # bin_data_norm_sub[bin_data_norm_sub < 0] = 0

        return bin_centers, bin_data_norm, bin_err, bin_data_norm_sub, bin_err_sub, bin_bkg

    def sub(self, params, x, y, e):

        a, b = params

        y_fit = self.linear(x, a, b)

        return ((y_fit-y)/e).flatten()

    def background(self, x, y, e):

        mask = (y > -np.inf) & (e > 0) & (y < np.inf) & (e < np.inf)

        A = (np.array([x[mask]*0+1, x[mask]])/e[mask]**2).T
        B = y[mask]/e[mask]**2

        if y[mask].size > 3:
            coeff, r, rank, s = np.linalg.lstsq(A, B)

            args = (x[mask], y[mask], e[mask])
            result = scipy.optimize.least_squares(self.sub, args=args, x0=coeff, loss='soft_l1')

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

        n_df = y[mask].size-4

        chi_sq = np.sum((y[mask]-y_fit[mask])**2/e[mask]**2)/n_df if n_df >= 1 else np.inf

        peak_bkg_ratio = np.std(y[mask])/np.median(e[mask])

        sig_noise_ratio = np.sum(y[mask])/np.sqrt(np.sum(e[mask]**2))

        return chi_sq, peak_bkg_ratio, sig_noise_ratio

    def estimate(self, x, y, e): 

        mask = (y > -np.inf) & (y < np.inf) & (e > 0) & (e < np.inf)
        
        weights = y**2/e**2

        if mask.sum() <= 4 or weights[mask].sum() <= 0:

            a, mu, sigma = 0, 0, 0

            min_bounds = (0,      -np.inf, 0     )
            max_bounds = (np.inf,  np.inf, np.inf)

        else:

            mu = np.average(x[mask], weights=weights[mask])

            sigma = np.sqrt(np.average((x[mask]-mu)**2, weights=weights[mask]))

            a = y[mask].max()
            if a < 0:
                a = 1

            center = 0.5*(x[mask].max()+x[mask].min())
            width = 0.5*(x[mask].max()-x[mask].min())

            min_bounds = (0,      x[mask].min(), 0,           )
            max_bounds = (1.05*a, x[mask].max(), width.max()/3)

            if np.any([mu < min_bounds[1], mu > max_bounds[1], sigma > width/3]):

                mu, sigma = center, width/6

            i = np.argmin(np.abs(mu-x))

            indices = np.argwhere(mask).flatten()

            ind = np.searchsorted(indices, i)

            if ind > 0 and ind < len(indices):
                ind = indices[ind]
                a = y.flatten()[ind]

            if a < min_bounds[0] or a > max_bounds[0]:
                a = 0.5*(min_bounds[0]+max_bounds[0])

        args = (x[mask], y[mask], e[mask])

        params = (a, mu, sigma)

        bounds = (min_bounds, max_bounds)

        return args, params, bounds

    def func(self, params, x, y, e):

        a, mu, sigma = params

        y_fit = self.gaussian(x, a, mu, sigma)

        return ((y_fit-y)*y/e).flatten()

    def fit(self, ellip, bkg_scale=0.95):

        data = ellip.data.copy()
        norm = ellip.norm.copy()

        x = ellip.Qp.copy()

        y = data/norm
        e = np.sqrt(data)/norm

        xh, yh, eh, y_sub, e_sub, y_bkg = self.histogram(x, data, norm, ellip, bkg_scale)

        y_fit = y_sub.copy()

        args, params, bounds = self.estimate(xh, y_sub, e_sub)

        if args[0].size > 4:
            result = scipy.optimize.least_squares(self.func, args=args, x0=params, bounds=bounds, loss='soft_l1')
            params = result.x

        a, mu, sigma = params

        y_fit = self.gaussian(xh, a, mu, sigma)

        self.a = a

        self.x, self.y, self.e = xh.copy(), yh.copy(), eh.copy()

        self.y_sub, self.e_sub = y_sub.copy(), e_sub.copy()
        self.y_bkg, self.y_fit = y_bkg.copy(), y_fit.copy()

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

    def histogram(self, x, y, data, norm, ellip, bkg_scale=0.95):

        int_mask, bkg_mask = ellip.projection_mask()

        data_norm = data/norm
        data_norm[data_norm < 0] = 0

        bins_x = bin_size(x, int_mask, data_norm)
        bin_edges_x = np.histogram_bin_edges(x, bins=bins_x)    
        bin_centers_x = 0.5*(bin_edges_x[1:]+bin_edges_x[:-1])

        bins_y = bin_size(y, int_mask, data_norm)
        bin_edges_y = np.histogram_bin_edges(y, bins=bins_y)
        bin_centers_y = 0.5*(bin_edges_y[1:]+bin_edges_y[:-1])

        bin_centers_x, bin_centers_y = np.meshgrid(bin_centers_x, bin_centers_y, indexing='ij')

        bin_data, _, _ = np.histogram2d(x[bkg_mask], y[bkg_mask], bins=[bin_edges_x,bin_edges_y], weights=data[bkg_mask])
        bin_norm, _, _ = np.histogram2d(x[bkg_mask], y[bkg_mask], bins=[bin_edges_x,bin_edges_y], weights=norm[bkg_mask])

        bin_norm[np.isclose(bin_norm, 0)] = np.nan

        mask = np.isinf(bin_norm) | np.isnan(bin_norm)

        bin_norm[:-1,:][np.cumsum(mask[1:,:], axis=0) == 1] = np.nan
        bin_norm[1:,:][np.cumsum(mask[::-1,:][1:,:], axis=0)[::-1,:] == 1] = np.nan

        bin_norm[:,:-1][np.cumsum(mask[:,1:], axis=1) == 1] = np.nan
        bin_norm[:,1:][np.cumsum(mask[:,::-1][:,1:], axis=1)[:,::-1] == 1] = np.nan

        bin_data_norm = bin_data/bin_norm
        bin_err = np.sqrt(bin_data)/bin_norm

        a, b, c = self.background(bin_centers_x, bin_centers_y, bin_data_norm, bin_err)

        mask = (bin_data_norm > 0) & (bin_data_norm < np.inf)

        bin_bkg = bkg_scale*self.linear(bin_centers_x, bin_centers_y, a, b, c)
        bin_bkg_err = np.sqrt(np.average((bin_data_norm[mask]-bin_bkg[mask])**2))

        bin_data, _, _ = np.histogram2d(x[int_mask], y[int_mask], bins=[bin_edges_x,bin_edges_y], weights=data[int_mask])
        bin_norm, _, _ = np.histogram2d(x[int_mask], y[int_mask], bins=[bin_edges_x,bin_edges_y], weights=norm[int_mask])

        bin_norm[np.isclose(bin_norm, 0)] = np.nan

        mask = np.isinf(bin_norm) | np.isnan(bin_norm)
        mask = (np.cumsum(~mask, axis=0) == 1) | (np.cumsum(~mask[::-1,:], axis=0) == 1)[::-1,:]\
             | (np.cumsum(~mask, axis=1) == 1) | (np.cumsum(~mask[:,::-1], axis=1) == 1)[:,::-1]

        bin_norm[mask] = np.nan

        bin_data_norm = bin_data/bin_norm
        bin_err = np.sqrt(bin_data)/bin_norm
        
        bin_data_norm_sub = bin_data/bin_norm-bin_bkg
        bin_err_sub = np.sqrt(bin_data/bin_norm**2)

        # bin_data_norm_sub[bin_data_norm_sub < 0] = 0

        mask = bin_data_norm_sub > 0
        if mask.sum() > 1:
            med_bkg = np.median(bin_data_norm_sub[mask])
            bin_data_norm_sub -= med_bkg

        return bin_centers_x, bin_centers_y, bin_data_norm, bin_err, bin_data_norm_sub, bin_err_sub, bin_bkg

    def sub(self, params, x, y, z, e):

        a, b, c = params

        z_fit = self.linear(x, y, a, b, c)

        return ((z_fit-z)/e).flatten()

    def background(self, x, y, z, e):

        mask = (z > -np.inf) & (e > 0) & (z < np.inf) & (e < np.inf)

        A = (np.array([x[mask]*0+1, x[mask], y[mask]])/e[mask]**2).T
        B = z[mask]/e[mask]**2

        if z[mask].size > 4:
            coeff, r, rank, s = np.linalg.lstsq(A, B)

            args = (x[mask], y[mask], z[mask], e[mask])
            result = scipy.optimize.least_squares(self.sub, args=args, x0=coeff, loss='soft_l1')

            coeff = result.x
        else:
            coeff = np.array([0, 0, 0])

        return tuple(coeff)

    def statistics(self, x, y, z, e, z_fit):

        mask = (z > 0) & (e > 0) & (z_fit > 0) & (z < np.inf) & (e < np.inf)

        n_df = z[mask].size-6

        chi_sq = np.sum((z[mask]-z_fit[mask])**2/e[mask]**2)/n_df if n_df >= 1 else np.inf

        peak_bkg_ratio = np.std(z[mask])/np.median(e[mask])

        sig_noise_ratio = np.sum(z[mask])/np.sqrt(np.sum(e[mask]**2))

        return chi_sq, peak_bkg_ratio, sig_noise_ratio

    def estimate(self, x, y, z, e): 

        mask = (z > -np.inf) & (z < np.inf) & (e > 0) & (e < np.inf)

        weights = z**2/e**2

        if mask.sum() <= 6 or weights[mask].sum() <= 0:

            a, mu_x, mu_y, sigma_x, sigma_y, rho = 0, 0, 0, 0, 0, 0

            min_bounds = (0,      -np.inf, -np.inf, 0,      0,      -1)
            max_bounds = (np.inf,  np.inf,  np.inf, np.inf, np.inf,  1)

        else:

            mu_x = np.average(x[mask], weights=weights[mask])
            mu_y = np.average(y[mask], weights=weights[mask])

            sigma_x = np.sqrt(np.average((x[mask]-mu_x)**2, weights=weights[mask]))
            sigma_y = np.sqrt(np.average((y[mask]-mu_y)**2, weights=weights[mask]))

            rho = np.average((x[mask]-mu_x)*(y[mask]-mu_y), weights=weights[mask])/sigma_x/sigma_y

            a = z[mask].max()
            if a < 0:
                a = 1

            center = np.array([0.5*(x[mask].max()+x[mask].min()), 0.5*(y[mask].max()+y[mask].min())])
            width = np.array([0.5*(x[mask].max()-x[mask].min()), 0.5*(y[mask].max()-y[mask].min())])

            min_bounds = (0,      x[mask].min(), y[mask].min(), 0.001,      0.001,      -1)
            max_bounds = (1.05*a, x[mask].max(), y[mask].max(), width[0]/3, width[1]/3,  1)

            if np.any([mu_x < min_bounds[1], mu_y < min_bounds[2],
                       mu_x > max_bounds[1], mu_y > max_bounds[2],
                       sigma_x > width[0]/3, sigma_y > width[1]/3]):

                mu_x, mu_y = center
                sigma_x, sigma_y = width/6

                theta = 0

            i = np.argmin(np.abs(mu_x-x[:,0]))
            j = np.argmin(np.abs(mu_y-y[0,:]))

            n = np.ravel_multi_index((i,j), z.shape)
            indices = np.ravel_multi_index(np.argwhere(mask).T, z.shape)

            ind = np.searchsorted(indices, n)

            if ind > 0 and ind < len(indices):
                ind = indices[ind]
                a = z.flatten()[ind]

            if a < min_bounds[0] or a > max_bounds[0]:
                a = 0.5*(min_bounds[0]+max_bounds[0])

        args = (x[mask], y[mask], z[mask], e[mask])

        params = (a, mu_x, mu_y, sigma_x, sigma_y, rho)

        bounds = (min_bounds, max_bounds)

        return args, params, bounds

    def func(self, params, x, y, z, e):

        a, mu_x, mu_y, sigma_x, sigma_y, rho = params

        z_fit = self.gaussian(x, y, a, mu_x, mu_y, sigma_x, sigma_y, rho)

        return ((z_fit-z)*z/e).flatten()

    def fit(self, ellip, bkg_scale=0.95):

        data = ellip.data.copy()
        norm = ellip.norm.copy()

        x = ellip.dQ1.copy()
        y = ellip.dQ2.copy()

        z = data/norm
        e = np.sqrt(data)/norm

        xh, yh, zh, eh, z_sub, e_sub, z_bkg = self.histogram(x, y, data, norm, ellip, bkg_scale)

        args, params, bounds = self.estimate(xh, yh, z_sub, e_sub)

        if args[0].size > 7:
           result = scipy.optimize.least_squares(self.func, args=args, x0=params, bounds=bounds, loss='soft_l1')
           params = result.x

        a, mu_x, mu_y, sigma_x, sigma_y, rho = params

        z_fit = self.gaussian(xh, yh, a, mu_x, mu_y, sigma_x, sigma_y, rho)

        self.a = a

        self.x, self.y, self.z, self.e = xh.copy(), yh.copy(), zh.copy(), eh.copy()

        self.z_sub, self.e_sub = z_sub.copy(), e_sub.copy()
        self.z_bkg, self.z_fit = z_bkg.copy(), z_fit.copy()

        return self.statistics(xh, yh, z_sub, e_sub, z_fit), (a, mu_x, mu_y, sigma_x, sigma_y, rho)