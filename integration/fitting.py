import numpy as np
import scipy.optimize
from lmfit import Minimizer, Parameters, fit_report

class Ellipsoid:

    def __init__(self, Qx, Qy, Qz, data, norm, Q0, size=1, n_std=3, scale=3, rotation=False):

        self.Q0 = Q0

        self.n, self.mu = self.profile_axis(self.Q0, rotation)
        self.u, self.v = self.projection_axes(self.n)

        self.size = size

        self.n_std, self.scale = n_std, scale

        self.sigma, self.sigma_x, self.sigma_y = size/n_std, size/scale, size/scale

        self.mu_x, self.mu_y, self.rho = 0, 0, 0

        self.Qx, self.Qy, self.Qz = Qx.copy(), Qy.copy(), Qz.copy()

        self.transform()

        self.data = data.copy()
        self.norm = norm.copy()

    def transform(self):

        u, v, n = self.u, self.v, self.n

        Q0 = self.Q0

        Qx, Qy, Qz = self.Qx, self.Qy, self.Qz

        self.dQ1 = u[0]*(Qx-Q0[0])+u[1]*(Qy-Q0[1])+u[2]*(Qz-Q0[2])
        self.dQ2 = v[0]*(Qx-Q0[0])+v[1]*(Qy-Q0[1])+v[2]*(Qz-Q0[2])

        self.Qp = Qx*n[0]+Qy*n[1]+Qz*n[2]

    def reset_axes(self, k0):

        self.n = k0/np.linalg.norm(k0)
        self.u, self.v = self.projection_axes(self.n)

        self.transform()

        self.mu = np.dot(self.Q0, self.n)

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

        if np.linalg.det(cov) > 0:
            vals, vecs = np.linalg.eig(cov)
        else:
            vals, vecs = np.zeros(2), np.eye(2)

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
            wu, wv = vp, -up
            ru, rv = radii[::-1]

        W = np.column_stack((wu,wv,n))

        D = np.diag(1/np.array([ru,rv,rp])**2)

        return Q, W, D

    def sig(self):

        sigma, sigma_x, sigma_y, rho = self.sigma, self.sigma_x, self.sigma_y, self.rho

        cov = np.array([[sigma_x**2, rho*sigma_x*sigma_y],
                        [rho*sigma_x*sigma_y, sigma_y**2]])

        if np.linalg.det(cov) > 0:
            vals, vecs = np.linalg.eig(cov)
        else:
            vals, vecs = np.zeros(2), np.eye(2)

        sigma_i = np.sqrt(vals)

        if sigma_i[0] > sigma_i[1]:
            sigma_i = sigma_i[::-1]

        return np.append(sigma_i,sigma)

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

    def profile_mask(self, extend=False):

        size = self.size

        mu, sigma = self.mu, self.sigma

        n_std = self.n_std+1.5

        Qp = self.Qp

        dQp = np.abs(Qp-mu)

        radius = n_std*sigma

        if not extend:
            if radius > 0.75*size:
                radius = 0.75*size
        else:
            radius = 1.75*size

        if radius < 0.075:
            radius = 0.075

        mu_x, mu_y, sigma_x, sigma_y, rho = self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho

        scale = self.scale

        dQ1, dQ2 = self.dQ1-mu_x, self.dQ2-mu_y

        cov = np.array([[sigma_x**2, rho*sigma_x*sigma_y],
                        [rho*sigma_x*sigma_y, sigma_y**2]])

        if np.linalg.det(cov) > 0:
            vals, vecs = np.linalg.eig(cov)
        else:
            vals, vecs = np.zeros(2), np.eye(2)

        radii = scale*np.sqrt(vals)
        radii[radii > 0.75*size] = 0.75*size

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

        if radius > 0.75*size:
            radius = 0.75*size
        elif radius < 0.075:
            radius = 0.075

        mu_x, mu_y, sigma_x, sigma_y, rho = self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho

        scale = self.scale+3

        dQ1, dQ2 = self.dQ1-mu_x, self.dQ2-mu_y

        cov = np.array([[sigma_x**2, rho*sigma_x*sigma_y],
                        [rho*sigma_x*sigma_y, sigma_y**2]])

        if np.linalg.det(cov) > 0:
            vals, vecs = np.linalg.eig(cov)
        else:
            vals, vecs = np.zeros(2), np.eye(2)

        radii = scale*np.sqrt(vals)
        radii[radii > 0.75*size] = 0.75*size

        W = vecs.copy()
        D = np.diag(1/radii**2)

        A = np.dot(np.dot(W, D), W.T)

        Q = self.Q0

        Qx, Qy, Qz = self.Qx, self.Qy, self.Qz

        dQx, dQy, dQz = Qx-Q[0], Qy-Q[1], Qz-Q[2]

        int_mask = (dQp < radius) & (A[0,0]*dQ1**2+A[1,1]*dQ2**2+2*A[0,1]*dQ1*dQ2 < 1)
        bkg_mask = (dQp < radius) & (A[0,0]*dQ1**2+A[1,1]*dQ2**2+2*A[0,1]*dQ1*dQ2 > 1)

        return int_mask, bkg_mask

def estimate_bins(val, mask, weights):

    bins = 41

    if (weights[mask] > 0).sum() > 11:

        weights -= 0.95*weights[mask][weights[mask] > 0].min()

        mu = np.average(val[mask], weights=weights[mask]**2)
        sigma = np.sqrt(np.average((val[mask]-mu)**2, weights=weights[mask]**2))

        bin_size = 3.5*sigma/bins
        val_range = val.max()-val.min()

        if bin_size > 0 and not np.isclose(val_range,0):

            bins = np.min([np.ceil(val_range/bin_size),41])

    return int(bins)

class Profile:

    def __init__(self):

        self.a = 0
        self.mu = 0
        self.sigma = 0

        self.x, self.y, self.e = None, None, None

        self.y_sub, self.e_sub = None, None
        self.y_fit = None

    def gaussian(self, x, a, mu, sigma):

        return a*np.exp(-0.5*(x-mu)**2/sigma**2)

    def linear(self, x, a, b):

        return a+b*x

    def histogram(self, x, data, norm, int_mask, bkg_mask, bkg_scale=0.95):

        data_norm = data/norm
        data_norm[data_norm < 0] = 0

        bins = estimate_bins(x, int_mask, data_norm)
        bin_edges = np.histogram_bin_edges(x, bins=bins)

        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])

        bin_data, _ = np.histogram(x, bins=bin_edges, weights=data)
        bin_norm, _ = np.histogram(x, bins=bin_edges, weights=norm)

        bin_data_norm = bin_data/bin_norm
        bin_err = np.sqrt(bin_data)/bin_norm

        array = np.ones_like(data_norm)

        bin_bkg_array, _ = np.histogram(x[bkg_mask], bins=bin_edges, weights=array[bkg_mask])
        bin_int_array, _ = np.histogram(x[int_mask], bins=bin_edges, weights=array[int_mask])

        bin_bkg_mask = bin_bkg_array <= 2
        bin_int_mask = bin_int_array <= 2

        bin_data, _ = np.histogram(x[bkg_mask], bins=bin_edges, weights=data[bkg_mask])
        bin_norm, _ = np.histogram(x[bkg_mask], bins=bin_edges, weights=norm[bkg_mask])

        bin_norm[0] = np.nan
        bin_norm[-1] = np.nan

        bin_norm[bin_bkg_mask] = np.nan
        bin_norm[~bin_int_mask] = np.nan
        bin_norm[np.isclose(bin_norm, 0)] = np.nan

        mask = np.isinf(bin_norm) | np.isnan(bin_norm)

        bin_norm[:-1][np.cumsum(mask[1:]) == 1] = np.nan
        bin_norm[1:][np.cumsum(mask[::-1][1:])[::-1] == 1] = np.nan

        bin_data_norm_bkg = bin_data/bin_norm
        bin_err_bkg = np.sqrt(bin_data)/bin_norm

        a, b = self.background(bin_centers, bin_data_norm_bkg, bin_err_bkg)

        mask = (bin_data_norm_bkg > 0) & (bin_data_norm_bkg < np.inf)

        bin_bkg = bkg_scale*self.linear(bin_centers, a, b)
        bin_bkg_err = np.sqrt(np.average((bin_data_norm_bkg[mask]-bin_bkg[mask])**2))

        bin_data, _ = np.histogram(x[int_mask], bins=bin_edges, weights=data[int_mask])
        bin_norm, _ = np.histogram(x[int_mask], bins=bin_edges, weights=norm[int_mask])

        bin_norm[bin_int_mask] = np.nan
        bin_norm[~bin_bkg_mask] = np.nan
        bin_norm[np.isclose(bin_norm, 0)] = np.nan

        mask = np.isinf(bin_norm) | np.isnan(bin_norm)
        mask = (np.cumsum(~mask) == 1) | (np.cumsum(~mask[::-1]) == 1)[::-1]

        bin_data_norm_sub = bin_data/bin_norm
        bin_err_sub = np.sqrt(bin_data)/bin_norm

        bin_data_norm[~bin_int_mask] = bin_data_norm_sub[~bin_int_mask]
        bin_data_norm[bin_int_mask] = np.nan

        bin_data_norm_sub -= bin_bkg

        return bin_centers, bin_data_norm, bin_err, bin_data_norm_sub, bin_err_sub, bin_bkg

    def background(self, x, y, e):

        mask = (y > -np.inf) & (e > 0) & (y < np.inf) & (e < np.inf)

        A = (np.array([x[mask]*0+1, x[mask]])/e[mask]).T
        B = y[mask]/e[mask]

        if y[mask].size > 3:
            coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)

        else:
            coeff = np.array([0, 0])

        return tuple(coeff)

    def parameters(self, coeff):

        mu = -0.5*coeff[1]/coeff[2]
        sigma = np.sqrt(-0.5/coeff[2])
        a = np.exp(coeff[0]-0.25*coeff[1]**2/coeff[2])

        return a, mu, sigma

    def statistics(self, x, y, e, y_fit, mu, sigma, n_std=3):

        mask = (y > -np.inf) & (e > 0) & (y_fit > -np.inf) & (y < np.inf) & (e < np.inf)

        n_df = y[mask].size-5

        pk  = (np.abs(x-mu) <= n_std*sigma) & mask
        bkg = (np.abs(x-mu) >  n_std*sigma) & mask

        n_pk, n_bkg = np.sum(pk), np.sum(bkg)

        chi_sq = np.sum((y[mask]-y_fit[mask])**2/e[mask]**2)/n_df if n_df >= 1 else np.inf

        peak_bkg_ratio = np.std(y[pk])/np.median(e[bkg]) if n_bkg >= 1 else np.inf

        sig_noise_ratio = np.sum(y[pk])/np.sqrt(np.sum(e[pk]**2)) if n_pk >= 1 else np.inf

        return chi_sq, peak_bkg_ratio, sig_noise_ratio

    def estimate(self, x, y, e): 

        mask = (y > -np.inf) & (y < np.inf) & (e > 0) & (e < np.inf)

        weights = y**2/e**2

        if mask.sum() <= 6 or weights[mask].sum() <= 0:

            a, mu, sigma, b, c = 0, 0, 0, 0, 0

            min_bounds = (0,      -np.inf, 0,      0,      0     )
            max_bounds = (np.inf,  np.inf, np.inf, np.inf, np.inf)

        else:

            mu = np.average(x[mask], weights=weights[mask])

            sigma = np.sqrt(np.average((x[mask]-mu)**2, weights=weights[mask]))

            x_min, x_max = x[mask].min(), x[mask].max()
            y_min, y_max = y[mask].min(), y[mask].max()

            center = 0.5*(x_max+x_min)
            width  = 0.5*(x_max-x_min)

            x_range = (x_max-x_min)
            y_range = (y_max-y_min)

            min_bounds = (0.01*y_range, x[mask].min(), 0.00001/3,     y_min-0.5*y_range, -100*y_range/x_range)
            max_bounds = ( 100*y_range, x[mask].max(), width.max()/3, y_max+0.5*y_range,  100*y_range/x_range)

            a = y_range
            b = y_min
            c = 0

            if np.any([mu < min_bounds[1], mu > max_bounds[1], sigma > width/3]):

                mu, sigma = center, width/6

        args = (x[mask], y[mask], e[mask])

        params = (a, mu, sigma, b, c)

        bounds = (min_bounds, max_bounds)

        return args, params, bounds

    def func(self, params, x, y, e):

        A, mu, sigma, b, c = params

        y_fit = self.gaussian(x, A, mu, sigma)
        y_bkg = self.linear(x, b, c)

        w = 1/e

        y_fit += y_bkg

        y_fit[~np.isfinite(y_fit)] = 1e+15

        return ((y_fit-y)*w).flatten()

    def jac(self, params, x, y, e):

        A, mu, sigma, b, c = params

        y_fit = self.gaussian(x, A, mu, sigma)
        y_bkg = self.linear(x, b, c)

        w = 1/e

        fp_a = (np.exp(-0.5*(x-mu)**2/sigma**2)*w).flatten()
        fp_mu = (y_fit*(x-mu)/sigma**2*w).flatten()
        fp_sigma = (y_fit*(x-mu)**2/sigma**3*w).flatten()

        fp_b = (w).flatten()
        fp_c = (x*w).flatten()

        J = np.stack((fp_a,fp_mu,fp_sigma,fp_b,fp_c))

        J[np.isnan(J)] = 1e+15
        J[np.isinf(J)] = 1e+15

        return J

    def residual(self, params, x, y, e):

        a = params['a']
        mu = params['mu']
        sigma = params['sigma']

        b = params['b']
        c = params['c']

        x0 = (a, mu, sigma, b, c)

        return self.func(x0, x, y, e)

    def gradient(self, params, x, y, e):

        a = params['a']
        mu = params['mu']
        sigma = params['sigma']

        b = params['b']
        c = params['c']

        x0 = (a, mu, sigma, b, c)

        return self.jac(x0, x, y, e)

    def fit(self, Qp, data, norm, int_mask, bkg_mask, bkg_scale=0.95):

        x = Qp.copy()

        y = data/norm
        e = np.sqrt(data)/norm

        xh, yh, eh, y_sub, e_sub, y_bkg = self.histogram(x, data, norm, int_mask, bkg_mask, bkg_scale)

        y_fit = y_sub.copy()

        args, params, bounds = self.estimate(xh, y_sub, e_sub)

        a, mu, sigma, b, c = params

        self.mu = mu
        self.sigma = sigma

        min_bounds, max_bounds = bounds

        min_a, min_mu, min_sigma, min_b, min_c = min_bounds
        max_a, max_mu, max_sigma, max_b, max_c = max_bounds

        if args[0].size > 6:

            params = Parameters()
            params.add('a', value=a, min=min_a, max=max_a)
            params.add('mu', value=mu, min=min_mu, max=max_mu)
            params.add('sigma', value=sigma, min=min_sigma, max=max_sigma)
            params.add('b', value=b, min=min_b, max=max_b)
            params.add('c', value=c, min=min_c, max=max_c)

            out = Minimizer(self.residual, params, fcn_args=(args), Dfun=self.gradient, col_deriv=True, nan_policy='omit') #

            result = out.minimize(method='leastsq')

            params = result.params['a'].value, \
                     result.params['mu'].value, \
                     result.params['sigma'].value, \
                     result.params['b'].value, \
                     result.params['c'].value

        a, mu, sigma, b, c = params

        y_fit = self.gaussian(xh, a, mu, sigma)

        bkg = self.linear(xh, b, c)

        y_sub -= bkg
        y_bkg += bkg

        self.a = a
        self.b = b
        self.c = c

        self.x, self.y, self.e = xh.copy(), yh.copy(), eh.copy()

        self.y_sub, self.e_sub = y_sub.copy(), e_sub.copy()
        self.y_bkg, self.y_fit = y_bkg.copy(), y_fit.copy()

        return self.statistics(xh, y_sub, e_sub, y_fit, mu, sigma), (a, mu, sigma)

class Projection:

    def __init__(self):

        self.a = 0
        self.mu_x, self.mu_y = 0, 0
        self.sigma_x, self.sigma_y, self.rho = 0, 0, 0

        self.x, self.y, self.z, self.e = None, None, None, None

        self.z_sub, self.e_sub = None, None
        self.z_fit = None

    def gaussian(self, x, y, a, mu_x, mu_y, sigma_x, sigma_y, rho):

        return a*np.exp(-0.5/(1-rho**2)*((x-mu_x)**2/sigma_x**2+(y-mu_y)**2/sigma_y**2-2*rho*(x-mu_x)*(y-mu_y)/(sigma_x*sigma_y)))

    def gaussian_rotated(self, x, y, A, mu_x, mu_y, sigma_1, sigma_2, theta):

        a = 0.5*(np.cos(theta)**2/sigma_1**2+np.sin(theta)**2/sigma_2**2)
        b = 0.5*(np.sin(theta)**2/sigma_1**2+np.cos(theta)**2/sigma_2**2)
        c = 0.5*(1/sigma_1**2-1/sigma_2**2)*np.sin(2*theta)

        return A*np.exp(-(a*(x-mu_x)**2+b*(y-mu_y)**2+c*(x-mu_x)*(y-mu_y)))

    def linear(self, x, y, a, b, c):

        return a+b*x+c*y

    def nonlinear(self, x, y, a, b, c, d):

        return a+b*x+c*y+d*x*y

    def histogram(self, x, y, data, norm, int_mask, bkg_mask, bkg_scale=0.95):

        data_norm = data/norm
        data_norm[data_norm < 0] = 0

        bins_x = estimate_bins(x, int_mask, data_norm)
        bin_edges_x = np.histogram_bin_edges(x, bins=bins_x)    
        bin_centers_x = 0.5*(bin_edges_x[1:]+bin_edges_x[:-1])

        bins_y = estimate_bins(y, int_mask, data_norm)
        bin_edges_y = np.histogram_bin_edges(y, bins=bins_y)
        bin_centers_y = 0.5*(bin_edges_y[1:]+bin_edges_y[:-1])

        bin_centers_x, bin_centers_y = np.meshgrid(bin_centers_x, bin_centers_y, indexing='ij')

        bin_data, _, _ = np.histogram2d(x, y, bins=[bin_edges_x,bin_edges_y], weights=data)
        bin_norm, _, _ = np.histogram2d(x, y, bins=[bin_edges_x,bin_edges_y], weights=norm)

        bin_data_norm = bin_data/bin_norm
        bin_err = np.sqrt(bin_data)/bin_norm

        array = np.ones_like(data_norm)

        bin_bkg_array, _, _ = np.histogram2d(x[bkg_mask], y[bkg_mask], bins=[bin_edges_x,bin_edges_y], weights=array[bkg_mask])
        bin_int_array, _, _ = np.histogram2d(x[int_mask], y[int_mask], bins=[bin_edges_x,bin_edges_y], weights=array[int_mask])

        bin_bkg_mask = bin_bkg_array <= 2
        bin_int_mask = bin_int_array <= 2

        bin_data, _, _ = np.histogram2d(x[bkg_mask], y[bkg_mask], bins=[bin_edges_x,bin_edges_y], weights=data[bkg_mask])
        bin_norm, _, _ = np.histogram2d(x[bkg_mask], y[bkg_mask], bins=[bin_edges_x,bin_edges_y], weights=norm[bkg_mask])

        bin_norm[0,:] = np.nan
        bin_norm[-1,:] = np.nan

        bin_norm[:,0] = np.nan
        bin_norm[:,-1] = np.nan

        bin_norm[bin_bkg_mask] = np.nan
        bin_norm[~bin_int_mask] = np.nan
        bin_norm[np.isclose(bin_norm, 0)] = np.nan

        mask = np.isinf(bin_norm) | np.isnan(bin_norm)

        bin_norm[:-1,:][np.cumsum(mask[1:,:], axis=0) == 1] = np.nan
        bin_norm[1:,:][np.cumsum(mask[::-1,:][1:,:], axis=0)[::-1,:] == 1] = np.nan

        bin_norm[:,:-1][np.cumsum(mask[:,1:], axis=1) == 1] = np.nan
        bin_norm[:,1:][np.cumsum(mask[:,::-1][:,1:], axis=1)[:,::-1] == 1] = np.nan

        bin_data_norm_bkg = bin_data/bin_norm
        bin_err_bkg = np.sqrt(bin_data)/bin_norm

        a, b, c, d = self.background(bin_centers_x, bin_centers_y, bin_data_norm_bkg, bin_err_bkg)

        mask = (bin_data_norm_bkg > 0) & (bin_data_norm_bkg < np.inf)

        bin_bkg = bkg_scale*self.nonlinear(bin_centers_x, bin_centers_y, a, b, c, d)
        bin_bkg_err = np.sqrt(np.average((bin_data_norm_bkg[mask]-bin_bkg[mask])**2))

        bin_data, _, _ = np.histogram2d(x[int_mask], y[int_mask], bins=[bin_edges_x,bin_edges_y], weights=data[int_mask])
        bin_norm, _, _ = np.histogram2d(x[int_mask], y[int_mask], bins=[bin_edges_x,bin_edges_y], weights=norm[int_mask])

        bin_norm[bin_int_mask] = np.nan
        bin_norm[~bin_bkg_mask] = np.nan
        bin_norm[np.isclose(bin_norm, 0)] = np.nan

        mask = np.isinf(bin_norm) | np.isnan(bin_norm)
        mask = (np.cumsum(~mask, axis=0) == 0) | (np.cumsum(~mask[::-1,:], axis=0) == 1)[::-1,:]\
             | (np.cumsum(~mask, axis=1) == 1) | (np.cumsum(~mask[:,::-1], axis=1) == 1)[:,::-1]

        bin_norm[mask] = np.nan

        bin_data_norm_sub = bin_data/bin_norm
        bin_err_sub = np.sqrt(bin_data)/bin_norm

        bin_data_norm[~bin_int_mask] = bin_data_norm_sub[~bin_int_mask]
        bin_data_norm[bin_int_mask] = np.nan

        bin_data_norm_sub -= bin_bkg

        return bin_centers_x, bin_centers_y, bin_data_norm, bin_err, bin_data_norm_sub, bin_err_sub, bin_bkg

    def background(self, x, y, z, e):

        mask = (z > -np.inf) & (e > 0) & (z < np.inf) & (e < np.inf)

        A = (np.array([x[mask]*0+1, x[mask], y[mask], x[mask]*y[mask]])/e[mask]).T
        B = z[mask]/e[mask]

        if z[mask].size > 5:
            coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)

        else:
            coeff = np.array([0, 0, 0, 0])

        return tuple(coeff)

    def statistics(self, x, y, z, e, z_fit, mu_x, mu_y, sigma_x, sigma_y, rho, scale=3):

        cov = np.array([[sigma_x**2, rho*sigma_x*sigma_y],
                        [rho*sigma_x*sigma_y, sigma_y**2]])

        if np.linalg.det(cov) > 0:

            vals, vecs = np.linalg.eig(cov)

            radii = scale*np.sqrt(vals)

            W = vecs.copy()
            D = np.diag(1/radii**2)

            A = np.dot(np.dot(W, D), W.T)

            mask = (z > -np.inf) & (e > 0) & (z_fit > -np.inf) & (z < np.inf) & (e < np.inf)

            n_df = z[mask].size-10

            pk  = (A[0,0]*(x-mu_x)**2+A[1,1]*(y-mu_y)**2+2*A[0,1]*(x-mu_x)*(y-mu_y) <= 1) & mask
            bkg = (A[0,0]*(x-mu_x)**2+A[1,1]*(y-mu_y)**2+2*A[0,1]*(x-mu_x)*(y-mu_y) >  1) & mask

            n_pk, n_bkg = np.sum(pk), np.sum(bkg)

            chi_sq = np.sum((z[mask]-z_fit[mask])**2/e[mask]**2)/n_df if n_df >= 1 else np.inf

            peak_bkg_ratio = np.std(z[pk])/np.median(e[bkg]) if n_bkg >= 1 else np.inf

            sig_noise_ratio = np.sum(np.abs(z[pk]))/np.sqrt(np.sum(e[pk]**2)) if n_pk >= 1 else np.inf

        else:

            chi_sq, peak_bkg_ratio, sig_noise_ratio = np.inf, np.inf, np.inf

        return chi_sq, peak_bkg_ratio, sig_noise_ratio

    def estimate(self, x, y, z, e): 

        mask = (z > 0) & (z < np.inf) & (e > 0) & (e < np.inf)

        weights = z**2/e**2

        if mask.sum() <= 10 or weights[mask].sum() <= 0:

            a, mu_x, mu_y, sigma_1, sigma_2, theta, b, cx, cy, cxy = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

            min_bounds = (1e-9,   -np.inf, -np.inf, 0,      0,      -np.pi/2, 0,      -np.inf, -np.inf, -np.inf)
            max_bounds = (np.inf,  np.inf,  np.inf, np.inf, np.inf,  np.pi/2, np.inf,  np.inf,  np.inf,  np.inf)

        else:

            mu_x = np.average(x[mask], weights=weights[mask])
            mu_y = np.average(y[mask], weights=weights[mask])

            sigma_x = np.sqrt(np.average((x[mask]-mu_x)**2, weights=weights[mask]))
            sigma_y = np.sqrt(np.average((y[mask]-mu_y)**2, weights=weights[mask]))

            rho = np.average((x[mask]-mu_x)*(y[mask]-mu_y), weights=weights[mask])/sigma_x/sigma_y

            cov = np.array([[sigma_x**2, rho*sigma_x*sigma_y],
                            [rho*sigma_x*sigma_y, sigma_y**2]])

            if np.linalg.det(cov) > 0:
                vals, vecs = np.linalg.eig(cov)
            else:
                vals, vecs = np.zeros(2), np.eye(2)

            sigma_1, sigma_2 = np.sqrt(vals)
            theta = np.arctan(vecs[1,0]/vecs[0,0])

            # sigma_1 = np.sqrt(0.5*(sigma_x**2+sigma_y**2)+np.sqrt(0.5*(sigma_x**2-sigma_y**2)**2+(rho*sigma_x*sigma_y)**2))
            # sigma_2 = np.sqrt(0.5*(sigma_x**2+sigma_y**2)-np.sqrt(0.5*(sigma_x**2-sigma_y**2)**2+(rho*sigma_x*sigma_y)**2))
            # theta = 0.5*np.arctan(2*(rho*sigma_x*sigma_y)/(sigma_x**2-sigma_y**2))

            x_min, x_max = x[mask].min(), x[mask].max()
            y_min, y_max = y[mask].min(), y[mask].max()
            z_min, z_max = z[mask].min(), z[mask].max()

            if np.isclose(x_min,x_max):
                x_min -= 0.1
                x_max += 0.1

            if np.isclose(y_min,y_max):
                y_min -= 0.1
                y_max += 0.1

            if np.isclose(z_min,z_max):
                z_min -= 0.1
                z_max += 0.1

            center = np.array([0.5*(x_max+x_min), 0.5*(y_max+y_min)])
            width  = np.array([0.5*(x_max-x_min), 0.5*(y_max-y_min)])

            width[:] = np.max(width)

            x_range = (x_max-x_min)
            y_range = (y_max-y_min)
            z_range = (z_max-z_min)

            min_bounds = (0.01*z_range, x_min, y_min, 0.01/3,     0.01/3,     -np.pi/2, z_min-0.5*z_range, -100*z_range/x_range, -100*z_range/y_range, -100*z_range/x_range/y_range)
            max_bounds = ( 100*z_range, x_max, y_max, width[0]/2, width[1]/2,  np.pi/2, z_max+0.5*z_range,  100*z_range/x_range,  100*z_range/y_range,  100*z_range/x_range/y_range)

            a = z_range
            b = z_min
            cx, cy, cxy = 0, 0, 0

            if np.any([mu_x < min_bounds[1], mu_y < min_bounds[2],
                       mu_x > max_bounds[1], mu_y > max_bounds[2],
                       sigma_1 > width[0]/2, sigma_2 > width[1]/2]):

                mu_x, mu_y = center
                sigma_1, sigma_2 = width/6

                theta = 0

        args = (x[mask], y[mask], z[mask], e[mask])

        params = (a, mu_x, mu_y, sigma_1, sigma_2, theta, b, cx, cy, cxy)

        bounds = (min_bounds, max_bounds)

        return args, params, bounds

    def func(self, params, x, y, z, e):

        A, mu_x, mu_y, sigma_1, sigma_2, theta, b, cx, cy, cxy = params

        z_fit = self.gaussian_rotated(x, y, A, mu_x, mu_y, sigma_1, sigma_2, theta)
        z_bkg = self.nonlinear(x, y, b, cx, cy, cxy)

        z_fit += z_bkg

        w = 1/e

        z_fit[~np.isfinite(z_fit)] = 1e+15

        return ((z_fit-z)*w).flatten()

    def jac(self, params, x, y, z, e):

        A, mu_x, mu_y, sigma_1, sigma_2, theta, b, cx, cy, cxy = params

        z_fit = self.gaussian_rotated(x, y, A, mu_x, mu_y, sigma_1, sigma_2, theta)

        w = 1/e

        a = 0.5*(np.cos(theta)**2/sigma_1**2+np.sin(theta)**2/sigma_2**2)
        b = 0.5*(np.sin(theta)**2/sigma_1**2+np.cos(theta)**2/sigma_2**2)
        c = 0.5*(1/sigma_1**2-1/sigma_2**2)*np.sin(2*theta)

        fp_a = (np.exp(-(a*(x-mu_x)**2+b*(y-mu_y)**2+c*(x-mu_x)*(y-mu_y)))*w).flatten()

        fp_mu_x = (z_fit*(2*a*(x-mu_x)+c*(y-mu_y))*w).flatten()
        fp_mu_y = (z_fit*(2*b*(y-mu_y)+c*(x-mu_x))*w).flatten()

        ap_sigma_1 = -np.cos(theta)**2/sigma_1**3
        ap_sigma_2 = -np.sin(theta)**2/sigma_2**3
        ap_theta = 0.5*(1/sigma_2**2-1/sigma_1**2)*np.sin(2*theta)

        bp_sigma_1 = -np.sin(theta)**2/sigma_1**3
        bp_sigma_2 = -np.cos(theta)**2/sigma_2**3
        bp_theta = 0.5*(1/sigma_1**2-1/sigma_2**2)*np.sin(2*theta)

        cp_sigma_1 = -np.sin(2*theta)**2/sigma_1**3
        cp_sigma_2 =  np.sin(2*theta)**2/sigma_2**3
        cp_theta = (1/sigma_1**2-1/sigma_2**2)*np.cos(2*theta)

        fp_sigma_1 = -(z_fit*(ap_sigma_1*(x-mu_x)**2+bp_sigma_1*(y-mu_y)**2+cp_sigma_1*(x-mu_x)*(y-mu_y))*w).flatten()
        fp_sigma_2 = -(z_fit*(ap_sigma_2*(x-mu_x)**2+bp_sigma_2*(y-mu_y)**2+cp_sigma_2*(x-mu_x)*(y-mu_y))*w).flatten()
        fp_theta = -(z_fit*(ap_theta*(x-mu_x)**2+bp_theta*(y-mu_y)**2+cp_theta*(x-mu_x)*(y-mu_y))*w).flatten()

        fp_b = (w).flatten()
        fp_cx = (x*w).flatten()
        fp_cy = (y*w).flatten()
        fp_cxy = (x*y*w).flatten()

        J = np.stack((fp_a,fp_mu_x,fp_mu_y,fp_sigma_1,fp_sigma_2,fp_theta,fp_b,fp_cx,fp_cy,fp_cxy))

        J[np.isnan(J)] = 1e+15
        J[np.isinf(J)] = 1e+15

        return J

    def residual(self, params, x, y, z, e):

        a = params['a']
        mu_x = params['mu_x']
        mu_y = params['mu_y']
        sigma_1 = params['sigma_1']
        sigma_2 = params['sigma_2']
        theta = params['theta']

        b = params['b']
        cx = params['cx']
        cy = params['cy']
        cxy = params['cxy']

        x0 = (a, mu_x, mu_y, sigma_1, sigma_2, theta, b, cx, cy, cxy)

        return self.func(x0, x, y, z, e)

    def gradient(self, params, x, y, z, e):

        a = params['a']
        mu_x = params['mu_x']
        mu_y = params['mu_y']
        sigma_1 = params['sigma_1']
        sigma_2 = params['sigma_2']
        theta = params['theta']

        b = params['b']
        cx = params['cx']
        cy = params['cy']
        cxy = params['cxy']

        x0 = (a, mu_x, mu_y, sigma_1, sigma_2, theta, b, cx, cy, cxy)

        return self.jac(x0, x, y, z, e)

    def fit(self, dQ1, dQ2, data, norm, int_mask, bkg_mask, bkg_scale=0.95, max_size=None):

        x = dQ1.copy()
        y = dQ2.copy()

        z = data/norm
        e = np.sqrt(data)/norm

        xh, yh, zh, eh, z_sub, e_sub, z_bkg = self.histogram(x, y, data, norm, int_mask, bkg_mask, bkg_scale)

        args, params, bounds = self.estimate(xh, yh, z_sub, e_sub)

        a, mu_x, mu_y, sigma_1, sigma_2, theta, b, cx, cy, cxy = params

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])

        cov = np.dot(R, np.dot(np.diag([sigma_1**2, sigma_2**2]), R.T))

        sigma_x, sigma_y = np.sqrt(np.diag(cov))
        rho = cov[0,1]/(sigma_x*sigma_y)

        self.mu_x, self.mu_y = mu_x, mu_y
        self.sigma_x, self.sigma_y, self.rho = sigma_x, sigma_y, rho

        min_bounds, max_bounds = bounds

        min_a, min_mu_x, min_mu_y, min_sigma_1, min_sigma_2, min_theta, min_b, min_cx, min_cy, min_cxy = min_bounds
        max_a, max_mu_x, max_mu_y, max_sigma_1, max_sigma_2, max_theta, max_b, max_cx, max_cy, max_cxy = max_bounds

        if max_size is not None:
            if 3*max_sigma_1 > max_size:
                max_sigma_1 = max_size/3
            if 3*max_sigma_2 > max_size:
                max_sigma_2 = max_size/3

        # xa, ya, za, ea = np.array([0.005]), np.array([0.006]), np.array([1.1]), np.array([0.9])
        # h = 1e-4
        # for i in range(10):
        #     fargs = list(params)
        #     fargs[i] += h
        #     print(i,(self.func(fargs, xa, ya, za, ea)-self.func(params, xa, ya, za, ea))/h, self.jac(params, xa, ya, za, ea)[i,:])

        if args[0].size > 11:

            params = Parameters()
            params.add('a', value=a, min=min_a, max=max_a)
            params.add('mu_x', value=mu_x, min=min_mu_x, max=max_mu_x)
            params.add('mu_y', value=mu_y, min=min_mu_y, max=max_mu_y)
            params.add('sigma_1', value=sigma_1, min=min_sigma_1, max=max_sigma_1)
            params.add('sigma_2', value=sigma_2, min=min_sigma_2, max=max_sigma_2)
            params.add('theta', value=theta, min=min_theta, max=max_theta, vary=True)
            params.add('b', value=b, min=min_b, max=max_b)
            params.add('cx', value=cx, min=min_cx, max=max_cx)
            params.add('cy', value=cy, min=min_cy, max=max_cy)
            params.add('cxy', value=cxy, min=min_cxy, max=max_cxy)

            # reduce_fcn = None if not robust else self.loss

            out = Minimizer(self.residual, params, fcn_args=(args)) #, Dfun=self.gradient, col_deriv=True, nan_policy='omit'
            result = out.minimize(method='leastsq')

            params = result.params['a'].value, \
                     result.params['mu_x'].value, \
                     result.params['mu_y'].value, \
                     result.params['sigma_1'].value, \
                     result.params['sigma_2'].value, \
                     result.params['theta'].value, \
                     result.params['b'].value, \
                     result.params['cx'].value, \
                     result.params['cy'].value, \
                     result.params['cxy'].value

        a, mu_x, mu_y, sigma_1, sigma_2, theta, b, cx, cy, cxy = params

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])

        cov = np.dot(R, np.dot(np.diag([sigma_1**2, sigma_2**2]), R.T))

        sigma_x, sigma_y = np.sqrt(np.diag(cov))
        rho = cov[0,1]/(sigma_x*sigma_y)

        z_fit = self.gaussian(xh, yh, a, mu_x, mu_y, sigma_x, sigma_y, rho)

        bkg = self.nonlinear(xh, yh, b, cx, cy, cxy)

        z_sub -= bkg
        z_bkg += bkg

        self.a = a
        self.b = b
        self.cx = cx
        self.cy = cy
        self.cxy = cxy

        self.x, self.y, self.z, self.e = xh.copy(), yh.copy(), zh.copy(), eh.copy()

        self.z_sub, self.e_sub = z_sub.copy(), e_sub.copy()
        self.z_bkg, self.z_fit = z_bkg.copy(), z_fit.copy()

        return self.statistics(xh, yh, z_sub, e_sub, z_fit, mu_x, mu_y, sigma_x, sigma_y, rho), (a, mu_x, mu_y, sigma_x, sigma_y, rho)

class LineCut(Profile):
 
    def __init__(self, delta=0):

        self.a0, self.a1, self.a2 = 0, 0, 0

        self.x, self.y, self.e = None, None, None

        self.y_sub, self.e_sub = None, None
        self.y_fit = None

        self.delta = delta

    def estimate(self, x, y, e): 

        mask = (y > -np.inf) & (y < np.inf) & (e > 0) & (e < np.inf)

        weights = y**2/e**2

        if mask.sum() <= 11 or weights[mask].sum() <= 0:

            a0, a1, a2, mu0, mu1, mu2, sigma, b, c = 0, 0, 0, 0, 0, 0, 0, 0, 0

            min_bounds = (0,      0,      0,      -np.inf, -np.inf, -np.inf, 0,      0,      0     )
            max_bounds = (np.inf, np.inf, np.inf,  np.inf,  np.inf,  np.inf, np.inf, np.inf, np.inf)
            
            a, center, width = 0, 0, 0

        else:

            mu = np.average(x[mask], weights=weights[mask])

            sigma = np.sqrt(np.average((x[mask]-mu)**2, weights=weights[mask]))

            b, c = 0, 0

            x_min, x_max = x[mask].min(), x[mask].max()
            y_min, y_max = y[mask].min(), y[mask].max()

            center = 0.5*(x_max+x_min)
            width  = 0.5*(x_max-x_min)

            x_range = x_max-x_min
            y_range = y_max-y_min

            min_bounds = (0.0*y_range, 0.0*y_range, 0.0*y_range, x_min, x_min, x_min, 0.0001/3,      y_min-y_range, -10*y_range/x_range)
            max_bounds = (  2*y_range,   2*y_range,   2*y_range, x_max, x_max, x_max, width.max()/3, y_max+y_range,  10*y_range/x_range)

            a = y_range

            if np.any([mu < min_bounds[3], mu > max_bounds[3], mu < min_bounds[4], mu > max_bounds[4], mu < min_bounds[5], mu > max_bounds[5], sigma > width/3]):

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

        params = (a, a, a, center-width/2, center, center+width/2, sigma/3, b, c)

        bounds = (min_bounds, max_bounds)

        return args, params, bounds

    def statistics(self, x, y, e, y_fit, mu0, mu1, mu2, sigma, n_std=3):

        mask = (y > -np.inf) & (e > 0) & (y_fit > -np.inf) & (y < np.inf) & (e < np.inf)

        n_df = y[mask].size-1

        mu = np.array([mu0,mu1,mu2])

        argmin, argmax = np.argmin(mu), np.argmax(mu)

        mu_min = mu[argmin]
        mu_max = mu[argmax]

        pk  = ((x-mu_min >= -n_std*sigma) & (x-mu_max <= n_std*sigma)) & mask
        bkg = ((x-mu_min <  -n_std*sigma) | (x-mu_max >  n_std*sigma)) & mask

        n_pk, n_bkg = np.sum(pk), np.sum(bkg)

        chi_sq = np.sum((y[mask]-y_fit[mask])**2/e[mask]**2)/n_df if n_df >= 1 else np.inf

        peak_bkg_ratio = np.std(y[pk])/np.median(e[bkg]) if n_bkg >= 1 else np.inf

        sig_noise_ratio = np.sum(y[pk])/np.sqrt(np.sum(e[pk]**2)) if n_pk >= 1 else np.inf

        return chi_sq, peak_bkg_ratio, sig_noise_ratio

    def func(self, params, x, y, e):

        A0, A1, A2, mu0, mu1, mu2, sigma, b, c = params

        y0_fit = self.gaussian(x, A0, mu0, sigma)
        y1_fit = self.gaussian(x, A1, mu1, sigma)
        y2_fit = self.gaussian(x, A2, mu2, sigma)

        y_fit = y0_fit+y1_fit+y2_fit
        y_bkg = self.linear(x, b, c)

        w = 1/e

        y_fit += y_bkg

        y_fit[~np.isfinite(y_fit)] = 1e+15

        return ((y_fit-y)*w).flatten()

    def jac(self, params, x, y, e):

        A0, A1, A2, mu0, mu1, mu2, sigma0, sigma1, sigma2, b, c = params

        y0_fit = self.gaussian(x, A0, mu0, sigma0)
        y1_fit = self.gaussian(x, A1, mu1, sigma1)
        y2_fit = self.gaussian(x, A2, mu2, sigma2)

        w = 1/e

        fp_a0 = (np.exp(-0.5*(x-mu0)**2/sigma0**2)*w).flatten()
        fp_a1 = (np.exp(-0.5*(x-mu1)**2/sigma1**2)*w).flatten()
        fp_a2 = (np.exp(-0.5*(x-mu2)**2/sigma2**2)*w).flatten()

        fp_mu0 = (y0_fit*(x-mu0)/sigma0**2*w).flatten()
        fp_mu1 = (y1_fit*(x-mu1)/sigma1**2*w).flatten()
        fp_mu2 = (y2_fit*(x-mu2)/sigma2**2*w).flatten()

        fp_sigma0 = (y0_fit*(x-mu0)**2/sigma0**3*w).flatten()
        fp_sigma1 = (y1_fit*(x-mu1)**2/sigma1**3*w).flatten()
        fp_sigma2 = (y2_fit*(x-mu2)**2/sigma2**3*w).flatten()

        fp_b = (w).flatten()
        fp_c = (x*w).flatten()

        J = np.stack((fp_a0,fp_a1,fp_a2,fp_mu0,fp_mu1,fp_mu2,fp_sigma0,fp_sigma1,fp_sigma2,fp_b,fp_c))

        #J[np.isnan(J)] = 1e+15
        #J[np.isinf(J)] = 1e+15

        return J

    def residual(self, params, x, y, e):

        a0 = params['a0']
        a1 = params['a1']
        a2 = params['a2']

        mu0 = params['mu0']
        mu1 = params['mu1']
        mu2 = params['mu2']

        sigma = params['sigma']

        b = params['b']
        c = params['c']

        x0 = (a0, a1, a2, mu0, mu1, mu2, sigma, b, c)

        return self.func(x0, x, y, e)

    def gradient(self, params, x, y, e):

        a0 = params['a0']
        a1 = params['a1']
        a2 = params['a2']

        mu0 = params['mu0']
        mu1 = params['mu1']
        mu2 = params['mu2']

        sigma = params['sigma']

        b = params['b']
        c = params['c']

        x0 = (a0, a1, a2, mu0, mu1, mu2, sigma, b, c)

        return self.jac(x0, x, y, e)

    def fit(self, Qp, data, norm, int_mask, bkg_mask, bkg_scale=0.95):

        x = Qp.copy()

        y = data/norm
        e = np.sqrt(data)/norm

        xh, yh, eh, y_sub, e_sub, y_bkg = self.histogram(x, data, norm, int_mask, bkg_mask, bkg_scale)

        y_fit = y_sub.copy()

        args, params, bounds = self.estimate(xh, y_sub, e_sub)

        a0, a1, a2, mu0, mu1, mu2, sigma, b, c = params

        min_bounds, max_bounds = bounds

        min_a0, min_a1, min_a2, min_mu0, min_mu1, min_mu2, min_sigma, min_b, min_c = min_bounds
        max_a0, max_a1, max_a2, max_mu0, max_mu1, max_mu2, max_sigma, max_b, max_c = max_bounds

        # h = 1e-6
        # 
        # for i in range(11):
        #     print(i,h)
        #     p0 = np.array(params)
        #     p = p0.copy()
        #     p[i] += h
        # 
        #     print(np.round((self.func(p, xh, yh, eh)-self.func(p0, xh, yh, eh))/h,2))
        #     print(np.round(self.jac(p0, xh, yh, eh),2)[i])

        errs = np.nan, np.nan, np.nan, np.nan

        if args[0].size > 9:

            params = Parameters()
            params.add('a0', value=a0, min=min_a0, max=max_a0)
            params.add('a1', value=a1, min=min_a1, max=max_a1)
            params.add('mu1', value=mu1, min=min_mu1, max=max_mu1)
            params.add('sigma', value=sigma, min=min_sigma, max=max_sigma)
            params.add('b', value=b, min=min_b, max=max_b)
            params.add('c', value=c, min=min_c, max=max_c)

            delta = self.delta
            if 2*delta <= 1.5*min_sigma:
                delta = 6*min_sigma

            params.add('kappa', value=delta-1.5*sigma, min=0, max=2*delta-1.5*min_sigma, vary=True)
            params.add('delta', expr='kappa+1.5*sigma')
            params.add('a2', expr='a0')

            params.add('mu0', expr='mu1-delta')
            params.add('mu2', expr='mu1+delta')

            out = Minimizer(self.residual, params, fcn_args=(args), nan_policy='omit') #, col_deriv=True, Dfun=self.gradient

            result = out.minimize(method='leastsq')

            params = result.params['a0'].value, \
                     result.params['a1'].value, \
                     result.params['a2'].value, \
                     result.params['mu0'].value, \
                     result.params['mu1'].value, \
                     result.params['mu2'].value, \
                     result.params['sigma'].value, \
                     result.params['b'].value, \
                     result.params['c'].value

            errs = result.params['a0'].stderr, \
                   result.params['a1'].stderr, \
                   result.params['a2'].stderr, \
                   result.params['sigma'].stderr

            if not np.all(errs):

                errs = np.nan, np.nan, np.nan, np.nan

            # print(fit_report(result))

        a0, a1, a2, mu0, mu1, mu2, sigma, b, c = params

        err_a0, err_a1, err_a2, err_s = errs

        y0_fit = self.gaussian(xh, a0, mu0, sigma)
        y1_fit = self.gaussian(xh, a1, mu1, sigma)
        y2_fit = self.gaussian(xh, a2, mu2, sigma)

        y_fit = y0_fit+y1_fit+y2_fit

        bkg = self.linear(xh, b, c)

        y_sub -= bkg
        y_bkg += bkg

        self.a0 = a0
        self.a1 = a1
        self.a2 = a2

        self.b = b
        self.c = c

        self.x, self.y, self.e = xh.copy(), yh.copy(), eh.copy()

        self.y_sub, self.e_sub = y_sub.copy(), e_sub.copy()
        self.y_bkg, self.y_fit = y_bkg.copy(), y_fit.copy()
       
        a = np.array([a0,a1,a2])
        err_a = np.array([err_a0,err_a1,err_a2])
        mu = np.array([mu0,mu1,mu2])

        sort = np.argsort(mu)

        a, err_a, mu = a[sort], err_a[sort], mu[sort]

        a0, a1, a2 = a
        mu0, mu1, mu2 = mu

        self.intensity = a*sigma*np.sqrt(2*np.pi)
        self.error = np.sqrt(2*np.pi*(a**2*err_s**2+sigma**2*err_a**2+2*a*sigma*0))        

        return self.statistics(xh, y_sub, e_sub, y_fit, mu0, mu1, mu2, sigma), (a0, a1, a2, mu0, mu1, mu2, sigma)

class GaussianFit3D:

    def __init__(self, x, y, e, mu, sigma):

        params = Parameters()

        x0_min, x0_max = np.min(x[0]), np.max(x[0])
        x1_min, x1_max = np.min(x[1]), np.max(x[1])
        x2_min, x2_max = np.min(x[2]), np.max(x[2])

        y_min, y_max = np.min(y), np.max(y)

        x0_range = x0_max-x0_min
        x1_range = x1_max-x1_min
        x2_range = x2_max-x2_min

        y_range = y_max-y_min

        params.add('A', value=y_range, min=0.001*y_range, max=1000*y_range)
        params.add('B', value=y_min, min=y_min-100*y_range, max=y_max+100*y_range)

        params.add('C0', value=0, min=-10*y_range/x0_range, max=10*y_range/x0_range)
        params.add('C1', value=0, min=-10*y_range/x1_range, max=10*y_range/x1_range)
        params.add('C2', value=0, min=-10*y_range/x2_range, max=10*y_range/x2_range)

        params.add('mu0', value=mu[0], min=mu[0]-0.1, max=mu[0]+0.1)
        params.add('mu1', value=mu[1], min=mu[1]-0.1, max=mu[1]+0.1)
        params.add('mu2', value=mu[2], min=mu[2]-0.1, max=mu[2]+0.1)

        params.add('sigma0', value=sigma[0], min=0.25*sigma[0], max=2*sigma[0])
        params.add('sigma1', value=sigma[1], min=0.25*sigma[1], max=2*sigma[1])
        params.add('sigma2', value=sigma[2], min=0.25*sigma[2], max=2*sigma[2])

        params.add('phi', value=0, min=-np.pi/2, max=np.pi/2)
        params.add('theta', value=np.pi/2, min=np.pi/4, max=3*np.pi/4)
        params.add('omega', value=0, min=-np.pi/2, max=np.pi/2)

        self.params = params

        self.x = x
        self.y = y
        self.e = e

    def gaussian_3d(self, Q0, Q1, Q2, A, mu0, mu1, mu2, sigma0, sigma1, sigma2, phi=0, theta=0, omega=0):

        x0, x1, x2 = Q0-mu0, Q1-mu1, Q2-mu2

        S = self.S_matrix(sigma0, sigma1, sigma2, phi, theta, omega)

        inv_S = np.linalg.inv(S)

        return A*np.exp(-0.5*(inv_S[0,0]*x0**2+inv_S[1,1]*x1**2+inv_S[2,2]*x2**2\
                          +2*(inv_S[1,2]*x1*x2+inv_S[0,2]*x0*x2+inv_S[0,1]*x0*x1)))

    def gaussian(self, Q0, Q1, Q2, A, mu0, mu1, mu2, sigma0, sigma1, sigma2, phi=0, theta=0, omega=0):

        x0, x1, x2 = Q0-mu0, Q1-mu1, Q2-mu2

        S = self.S_matrix(sigma0, sigma1, sigma2, phi, theta, omega)

        inv_S = np.linalg.inv(S)

        return A*np.exp(-0.5*(inv_S[0,0]*x0**2+inv_S[1,1]*x1**2+inv_S[2,2]*x2**2\
                          +2*(inv_S[1,2]*x1*x2+inv_S[0,2]*x0*x2+inv_S[0,1]*x0*x1)))

    def S_matrix(self, sigma0, sigma1, sigma2, phi=0, theta=0, omega=0):

        U = self.U_matrix(phi, theta, omega)
        V = np.diag(np.dot(np.linalg.inv(U**2), [sigma0**2, sigma1**2, sigma2**2]))

        S = np.dot(np.dot(U,V),U.T)

        return S

    def U_matrix(self, phi=0, theta=0, omega=0):

        ux = np.cos(phi)*np.sin(theta)
        uy = np.sin(phi)*np.sin(theta)
        uz = np.cos(theta)

        U = np.array([[np.cos(omega)+ux**2*(1-np.cos(omega)), ux*uy*(1-np.cos(omega))-uz*np.sin(omega), ux*uz*(1-np.cos(omega))+uy*np.sin(omega)],
                      [uy*ux*(1-np.cos(omega))+uz*np.sin(omega), np.cos(omega)+uy**2*(1-np.cos(omega)), uy*uz*(1-np.cos(omega))-ux*np.sin(omega)],
                      [uz*ux*(1-np.cos(omega))-uy*np.sin(omega), uz*uy*(1-np.cos(omega))+ux*np.sin(omega), np.cos(omega)+uz**2*(1-np.cos(omega))]])

        return U

    def residual(self, params, x, y, e):

        Q0, Q1, Q2 = x

        A = params['A']
        B = params['B']

        C0 = params['C0']
        C1 = params['C1']
        C2 = params['C2']

        mu0 = params['mu0']
        mu1 = params['mu1']
        mu2 = params['mu2']

        sigma0 = params['sigma0']
        sigma1 = params['sigma1']
        sigma2 = params['sigma2']

        phi = params['phi']
        theta = params['theta']
        omega = params['omega']

        args = Q0, Q1, Q2, A, B, C0, C1, C2, mu0, mu1, mu2, sigma0, sigma1, sigma2, phi, theta, omega

        yfit = self.func(*args)

        diff = (y-yfit)/e

        diff[~np.isfinite(diff)] = 1e+15
        diff[~np.isfinite(diff)] = 1e+15

        return diff

    def func(self, Q0, Q1, Q2, A, B, C0, C1, C2, mu0, mu1, mu2, sigma0, sigma1, sigma2, phi, theta, omega):

        args = Q0, Q1, Q2, A, mu0, mu1, mu2, sigma0, sigma1, sigma2, phi, theta, omega

        return self.gaussian(*args)+B+C0*Q0+C1*Q1+C2*Q2

    def gradient(self, params, x, y, e):

        Q0, Q1, Q2 = x

        A = params['A']
        B = params['B']

        C0 = params['C0']
        C1 = params['C1']
        C2 = params['C2']

        mu0 = params['mu0']
        mu1 = params['mu1']
        mu2 = params['mu2']

        sigma0 = params['sigma0']
        sigma1 = params['sigma1']
        sigma2 = params['sigma2']

        phi = params['phi']
        theta = params['theta']
        omega = params['omega']

        args = Q0, Q1, Q2, A, B, C0, C1, C2, mu0, mu1, mu2, sigma0, sigma1, sigma2, phi, theta, omega

        return self.jac(*args)/e

    def fit(self):

        out = Minimizer(self.residual, self.params, fcn_args=(self.x, self.y, self.e), nan_policy='omit') #, Dfun=self.gradient, col_deriv=True, nan_policy='omit'
        result = out.minimize(method='leastsq')

        #result = out.prepare_fit()

        self.params = result.params

        # report_fit(result)

        A = result.params['A'].value
        B = result.params['B'].value

        C0 = result.params['C0'].value
        C1 = result.params['C1'].value
        C2 = result.params['C2'].value

        mu0 = result.params['mu0'].value
        mu1 = result.params['mu1'].value
        mu2 = result.params['mu2'].value

        sigma0 = result.params['sigma0'].value
        sigma1 = result.params['sigma1'].value
        sigma2 = result.params['sigma2'].value

        phi = result.params['phi'].value
        theta = result.params['theta'].value
        omega = result.params['omega'].value

        # Q0, Q1, Q2 = self.x
        # Q0, Q1, Q2 = np.array([-0.06]), np.array([-0.05]), np.array([-0.025])
        # 
        # params = (Q0, Q1, Q2, A, B, mu0, mu1, mu2, sigma0, sigma1, sigma2, phi, theta, omega)
        # 
        # h = 1e-8
        # for i in range(11):
        #     fargs = list(params)
        #     fargs[i+3] += h
        # 
        #     print(fargs[3:])
        #     print(params[3:])
        #     print(i,(self.func(*fargs)-self.func(*params))/h, self.jac(*params)[i,:].round(8))

        # print(result.params['A'])
        # print(result.params['B'])
        # print(result.params['mu0'])
        # print(result.params['mu1'])
        # print(result.params['mu2'])
        # print(result.params['sigma0'])
        # print(result.params['sigma1'])
        # print(result.params['sigma2'])
        # print(result.params['phi'])
        # print(result.params['theta'])
        # print(result.params['omega'])

        boundary = self.check_boundary(A, B, mu0, mu1, mu2, sigma0, sigma1, sigma2, result.params)

        S = self.S_matrix(sigma0, sigma1, sigma2, phi, theta, omega)

        var = np.diag(S)
        sig = np.sqrt(var)

        sig_inv = np.diag(1/sig)

        rho = np.dot(np.dot(sig_inv, S), sig_inv)

        sig0, sig1, sig2 = sig[0], sig[1], sig[2]
        rho12, rho02, rho01 = rho[1,2], rho[0,2], rho[0,1]

        return A, B, C0, C1, C2, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01, boundary

    def eigendecomposition(self):

        params = self.params

        sigma0 = params['sigma0'].value
        sigma1 = params['sigma1'].value
        sigma2 = params['sigma2'].value

        phi = params['phi'].value
        theta = params['theta'].value
        omega = params['omega'].value

        U = self.U_matrix(phi, theta, omega)
        V = np.dot(np.linalg.inv(U**2), [sigma0**2, sigma1**2, sigma2**2])

        return V, U

    def estimate(self):

        Q0, Q1, Q2 = self.x
        y, e = self.y, self.e

        params = self.params

        bkg = np.percentile(y, 25)

        mask = (y > 0) & (y < bkg) & (e > 0) & (e < np.inf)

        if np.sum(mask) > 6:

            A = (np.array([np.ones_like(Q0[mask]), Q0[mask], Q1[mask], Q2[mask]])/e[mask]).T
            b = y[mask]/e[mask]

            coeff, r, rank, s = np.linalg.lstsq(A, b, rcond=None)

        else:

            coeff = 0, 0, 0, 0

        B, C0, C1, C2 = coeff

        z = y.copy()

        z -= B+C0*Q0+C1*Q1+C2*Q2
        z[z < 0] = 0

        weights = z**2/e**2

        mask = (z > 0) & (z < np.inf) & (e > 0) & (e < np.inf)

        if np.sum(mask) > 3:

            mu0 = np.average(Q0[mask], weights=weights[mask])
            mu1 = np.average(Q1[mask], weights=weights[mask])
            mu2 = np.average(Q2[mask], weights=weights[mask])

            x0, x1, x2 = Q0-mu0, Q1-mu1, Q2-mu2

            sig0 = np.sqrt(np.average(x0[mask]**2, weights=weights[mask]))
            sig1 = np.sqrt(np.average(x1[mask]**2, weights=weights[mask]))
            sig2 = np.sqrt(np.average(x2[mask]**2, weights=weights[mask]))

            rho12 = np.average(x1[mask]*x2[mask], weights=weights[mask])/sig1/sig2
            rho02 = np.average(x0[mask]*x2[mask], weights=weights[mask])/sig0/sig2
            rho01 = np.average(x0[mask]*x1[mask], weights=weights[mask])/sig0/sig1

            S = self.covariance_matrix(sig0, sig1, sig2, rho12, rho02, rho01)

            inv_S = np.linalg.inv(S)

            y = np.exp(-0.5*(inv_S[0,0]*x0**2+inv_S[1,1]*x1**2+inv_S[2,2]*x2**2\
                         +2*(inv_S[1,2]*x1*x2+inv_S[0,2]*x0*x2+inv_S[0,1]*x0*x1)))

            A = (np.array([y, np.ones_like(y)])/e).T
            b = z/e

            coeff, r, rank, s = np.linalg.lstsq(A, b, rcond=None)

        else:

            coeff = 0, 0

        A, b = coeff

        B += b

        boundary = self.check_outside(A, B, mu0, mu1, mu2, sig0, sig1, sig2, params)

        return A, B, C0, C1, C2, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01, boundary

    def check_outside(self, A, B, mu0, mu1, mu2, sigma0, sigma1, sigma2, params):

        A_min, A_max = params['A'].min, params['A'].max
        B_min, B_max = params['B'].min, params['B'].max

        mu0_min, mu0_max = params['mu0'].min, params['mu0'].max
        mu1_min, mu1_max = params['mu1'].min, params['mu1'].max
        mu2_min, mu2_max = params['mu2'].min, params['mu2'].max

        sigma0_min, sigma0_max = params['sigma0'].min, params['sigma0'].max
        sigma1_min, sigma1_max = params['sigma1'].min, params['sigma1'].max
        sigma2_min, sigma2_max = params['sigma2'].min, params['sigma2'].max

        boundary =  (mu0 <= mu0_min) or (mu0 >= mu0_max)\
                 or (mu1 <= mu1_min) or (mu1 >= mu1_max)\
                 or (mu2 <= mu2_min) or (mu2 >= mu2_max)\
                 or (sigma0 <= sigma0_min) or (sigma0 >= sigma0_max)\
                 or (sigma1 <= sigma1_min) or (sigma1 >= sigma1_max)\
                 or (sigma2 <= sigma2_min) or (sigma2 >= sigma2_max)

        return boundary

    def check_boundary(self, A, B, mu0, mu1, mu2, sigma0, sigma1, sigma2, params):

        A_min, A_max = params['A'].min, params['A'].max
        B_min, B_max = params['B'].min, params['B'].max

        mu0_min, mu0_max = params['mu0'].min, params['mu0'].max
        mu1_min, mu1_max = params['mu1'].min, params['mu1'].max
        mu2_min, mu2_max = params['mu2'].min, params['mu2'].max

        sigma0_min, sigma0_max = params['sigma0'].min, params['sigma0'].max
        sigma1_min, sigma1_max = params['sigma1'].min, params['sigma1'].max
        sigma2_min, sigma2_max = params['sigma2'].min, params['sigma2'].max

        boundary = np.isclose(mu0, mu0_min, rtol=1e-3) | np.isclose(mu0, mu0_max, rtol=1e-3) \
                 | np.isclose(mu1, mu1_min, rtol=1e-3) | np.isclose(mu1, mu1_max, rtol=1e-3) \
                 | np.isclose(mu2, mu2_min, rtol=1e-3) | np.isclose(mu2, mu2_max, rtol=1e-3) \
                 | np.isclose(sigma0, sigma0_min, rtol=1e-3) | np.isclose(sigma0, sigma0_max, rtol=1e-3) \
                 | np.isclose(sigma1, sigma1_min, rtol=1e-3) | np.isclose(sigma1, sigma1_max, rtol=1e-3) \
                 | np.isclose(sigma2, sigma2_min, rtol=1e-3) | np.isclose(sigma2, sigma2_max, rtol=1e-3)

        return boundary

    def covariance_matrix(self, sig0, sig1, sig2, rho12, rho02, rho01):

        sig = np.diag([sig0, sig1, sig2])

        rho = np.array([[1, rho01, rho02],
                        [rho01, 1, rho12],
                        [rho02, rho12, 1]])

        S = np.dot(np.dot(sig, rho), sig)

        return S

    def model(self, x, A, B, C0, C1, C2, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01):

        S = self.covariance_matrix(sig0, sig1, sig2, rho12, rho02, rho01)

        inv_S = np.linalg.inv(S)

        x0, x1, x2 = x[0]-mu0, x[1]-mu1, x[2]-mu2

        norm = np.sqrt(np.linalg.det(2*np.pi*S))

        return A*np.exp(-0.5*(inv_S[0,0]*x0**2+inv_S[1,1]*x1**2+inv_S[2,2]*x2**2\
                          +2*(inv_S[1,2]*x1*x2+inv_S[0,2]*x0*x2+inv_S[0,1]*x0*x1)))/norm+B+C0*x0+C1*x1+C2*x2

class SatelliteGaussianFit3D(GaussianFit3D):

    def __init__(self, x, y, e, mu, sigma, delta):

        if np.isclose(sigma[2]-delta/3, 0):
            delta = sigma[2]/2

        params = Parameters()

        x0_min, x0_max = np.min(x[0]), np.max(x[0])
        x1_min, x1_max = np.min(x[1]), np.max(x[1])
        x2_min, x2_max = np.min(x[2]), np.max(x[2])

        y_min, y_max = np.min(y), np.max(y)

        x0_range = x0_max-x0_min
        x1_range = x1_max-x1_min
        x2_range = x2_max-x2_min

        y_range = y_max-y_min

        params.add('A0', value=y_range, min=0.001*y_range, max=1000*y_range)
        params.add('A1', value=y_range, min=0.001*y_range, max=1000*y_range)
        params.add('A2', expr='A0')

        params.add('B', value=y_min, min=y_min-100*y_range, max=y_max+100*y_range)

        params.add('C0', value=0, min=-10*y_range/x0_range, max=10*y_range/x0_range)
        params.add('C1', value=0, min=-10*y_range/x1_range, max=10*y_range/x1_range)
        params.add('C2', value=0, min=-10*y_range/x2_range, max=10*y_range/x2_range)

        params.add('mu0', value=mu[0], min=mu[0]-0.1, max=mu[0]+0.1)
        params.add('mu1', value=mu[1], min=mu[1]-0.1, max=mu[1]+0.1)
        params.add('mu2', value=mu[2], min=mu[2]-0.1, max=mu[2]+0.1)

        params.add('delta', value=delta, min=0.5*delta, max=2*delta)

        params.add('sigma0', value=sigma[0], min=0.25*sigma[0], max=2*sigma[0])
        params.add('sigma1', value=sigma[1], min=0.25*sigma[1], max=2*sigma[1])
        params.add('sigma2', value=(sigma[2]-delta/3), min=0.25*(sigma[2]-delta/3), max=2*(sigma[2]-delta/3))

        params.add('phi', value=0, min=-np.pi/2, max=np.pi/2)
        params.add('theta', value=np.pi/2, min=np.pi/4, max=3*np.pi/4)
        params.add('omega', value=0, min=-np.pi/2, max=np.pi/2)

        self.params = params

        self.x = x
        self.y = y
        self.e = e

    def residual(self, params, x, y, e):

        Q0, Q1, Q2 = x

        A0 = params['A0']
        A1 = params['A1']
        A2 = params['A2']

        B = params['B']

        C0 = params['C0']
        C1 = params['C1']
        C2 = params['C2']

        mu0 = params['mu0']
        mu1 = params['mu1']
        mu2 = params['mu2']

        delta = params['delta']

        sigma0 = params['sigma0']
        sigma1 = params['sigma1']
        sigma2 = params['sigma2']

        phi = params['phi']
        theta = params['theta']
        omega = params['omega']

        args = Q0, Q1, Q2, A0, A1, A2, B, C0, C1, C2, mu0, mu1, mu2, delta, sigma0, sigma1, sigma2, phi, theta, omega

        yfit = self.func(*args)

        obj = (y-yfit)/e

        obj[~np.isfinite(obj)] = 1e+15
        obj[~np.isfinite(obj)] = 1e+15

        return obj

    def func(self, Q0, Q1, Q2, A0, A1, A2, B, C0, C1, C2, mu0, mu1, mu2, delta, sigma0, sigma1, sigma2, phi, theta, omega):

        args0 = Q0, Q1, Q2, A0, mu0, mu1, mu2-delta, sigma0, sigma1, sigma2, phi, theta, omega
        args1 = Q0, Q1, Q2, A1, mu0, mu1, mu2,       sigma0, sigma1, sigma2, phi, theta, omega
        args2 = Q0, Q1, Q2, A2, mu0, mu1, mu2+delta, sigma0, sigma1, sigma2, phi, theta, omega

        return self.gaussian(*args0)+self.gaussian(*args1)+self.gaussian(*args2)+B+C0*Q0+C1*Q1+C2*Q2

    def fit(self):

        out = Minimizer(self.residual, self.params, fcn_args=(self.x, self.y, self.e), nan_policy='omit') #, Dfun=self.gradient, col_deriv=True, nan_policy='omit'
        result = out.minimize(method='leastsq')

        #result = out.prepare_fit()

        self.params = result.params

        # report_fit(result)

        A0 = result.params['A0'].value
        A1 = result.params['A1'].value
        A2 = result.params['A2'].value

        B = result.params['B'].value

        C0 = result.params['C0'].value
        C1 = result.params['C1'].value
        C2 = result.params['C2'].value

        mu0 = result.params['mu0'].value
        mu1 = result.params['mu1'].value
        mu2 = result.params['mu2'].value

        delta = result.params['delta'].value

        sigma0 = result.params['sigma0'].value
        sigma1 = result.params['sigma1'].value
        sigma2 = result.params['sigma2'].value

        phi = result.params['phi'].value
        theta = result.params['theta'].value
        omega = result.params['omega'].value

        boundary = self.check_boundary(A0, A1, A2, B, mu0, mu1, mu2, sigma0, sigma1, sigma2, result.params)

        S = self.S_matrix(sigma0, sigma1, sigma2, phi, theta, omega)

        var = np.diag(S)
        sig = np.sqrt(var)

        sig_inv = np.diag(1/sig)

        rho = np.dot(np.dot(sig_inv, S), sig_inv)

        sig0, sig1, sig2 = sig[0], sig[1], sig[2]
        rho12, rho02, rho01 = rho[1,2], rho[0,2], rho[0,1]

        return A0, A1, A2, B, C0, C1, C2, mu0, mu1, mu2, delta, sig0, sig1, sig2, rho12, rho02, rho01, boundary

    def check_outside(self, A0, A1, A2, B, mu0, mu1, mu2, sigma0, sigma1, sigma2, params):

        A0_min, A0_max = params['A0'].min, params['A0'].max
        A1_min, A1_max = params['A1'].min, params['A1'].max
        A2_min, A2_max = params['A2'].min, params['A2'].max

        B_min, B_max = params['B'].min, params['B'].max

        mu0_min, mu0_max = params['mu0'].min, params['mu0'].max
        mu1_min, mu1_max = params['mu1'].min, params['mu1'].max
        mu2_min, mu2_max = params['mu2'].min, params['mu2'].max

        sigma0_min, sigma0_max = params['sigma0'].min, params['sigma0'].max
        sigma1_min, sigma1_max = params['sigma1'].min, params['sigma1'].max
        sigma2_min, sigma2_max = params['sigma2'].min, params['sigma2'].max

        boundary =  (mu0 <= mu0_min) or (mu0 >= mu0_max)\
                 or (mu1 <= mu1_min) or (mu1 >= mu1_max)\
                 or (mu2 <= mu2_min) or (mu2 >= mu2_max)\
                 or (sigma0 <= sigma0_min) or (sigma0 >= sigma0_max)\
                 or (sigma1 <= sigma1_min) or (sigma1 >= sigma1_max)\
                 or (sigma2 <= sigma2_min) or (sigma2 >= sigma2_max)

        return boundary

    def check_boundary(self, A0, A1, A2, B, mu0, mu1, mu2, sigma0, sigma1, sigma2, params):

        A0_min, A0_max = params['A0'].min, params['A0'].max
        A1_min, A1_max = params['A1'].min, params['A1'].max
        A2_min, A2_max = params['A2'].min, params['A2'].max

        B_min, B_max = params['B'].min, params['B'].max

        mu0_min, mu0_max = params['mu0'].min, params['mu0'].max
        mu1_min, mu1_max = params['mu1'].min, params['mu1'].max
        mu2_min, mu2_max = params['mu2'].min, params['mu2'].max

        sigma0_min, sigma0_max = params['sigma0'].min, params['sigma0'].max
        sigma1_min, sigma1_max = params['sigma1'].min, params['sigma1'].max
        sigma2_min, sigma2_max = params['sigma2'].min, params['sigma2'].max

        boundary = np.isclose(mu0, mu0_min, rtol=1e-3) | np.isclose(mu0, mu0_max, rtol=1e-3) \
                 | np.isclose(mu1, mu1_min, rtol=1e-3) | np.isclose(mu1, mu1_max, rtol=1e-3) \
                 | np.isclose(mu2, mu2_min, rtol=1e-3) | np.isclose(mu2, mu2_max, rtol=1e-3) \
                 | np.isclose(sigma0, sigma0_min, rtol=1e-3) | np.isclose(sigma0, sigma0_max, rtol=1e-3) \
                 | np.isclose(sigma1, sigma1_min, rtol=1e-3) | np.isclose(sigma1, sigma1_max, rtol=1e-3) \
                 | np.isclose(sigma2, sigma2_min, rtol=1e-3) | np.isclose(sigma2, sigma2_max, rtol=1e-3)

        return boundary

    def model(self, x, A0, A1, A2, B, C0, C1, C2, mu0, mu1, mu2, delta, sig0, sig1, sig2, rho12, rho02, rho01):

        S = self.covariance_matrix(sig0, sig1, sig2, rho12, rho02, rho01)

        inv_S = np.linalg.inv(S)

        x0, x1, x2_1 = x[0]-mu0, x[1]-mu1, x[2]-mu2

        x2_0 = x2_1-delta
        x2_2 = x2_1+delta

        norm = np.sqrt(np.linalg.det(2*np.pi*S))

        return (A0*np.exp(-0.5*(inv_S[0,0]*x0**2  +inv_S[1,1]*x1**2  +inv_S[2,2]*x2_0**2\
                            +2*(inv_S[1,2]*x1*x2_0+inv_S[0,2]*x0*x2_0+inv_S[0,1]*x0*x1)))\
               +A1*np.exp(-0.5*(inv_S[0,0]*x0**2  +inv_S[1,1]*x1**2  +inv_S[2,2]*x2_1**2\
                            +2*(inv_S[1,2]*x1*x2_1+inv_S[0,2]*x0*x2_1+inv_S[0,1]*x0*x1)))\
               +A2*np.exp(-0.5*(inv_S[0,0]*x0**2  +inv_S[1,1]*x1**2  +inv_S[2,2]*x2_2**2\
                            +2*(inv_S[1,2]*x1*x2_2+inv_S[0,2]*x0*x2_2+inv_S[0,1]*x0*x1))))/norm+B+C0*x0+C1*x1+C2*x2_1