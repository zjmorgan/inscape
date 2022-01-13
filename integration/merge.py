# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

from mantid.kernel import V3D
from mantid.geometry import Goniometer

from sklearn import mixture

import pprint
import pickle
#pickle.settings['recurse'] = True

def _pprint_dict(self, object, stream, indent, allowance, context, level):
    write = stream.write
    write('{')
    if self._indent_per_level > 1:
        write((self._indent_per_level - 1) * ' ')
    length = len(object)
    if length:
        self._format_dict_items(object.items(), stream, indent, allowance + 1,
                                context, level)
    write('}')
pprint.PrettyPrinter._dispatch[dict.__repr__] = _pprint_dict

from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib import patheffects
import matplotlib.gridspec as gridspec

plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 8

def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,a,b],
                     [0,0,0,zback]])

def set_axes_equal(ax: plt.Axes):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])
    
class PeakEnvelope:
    
    def __init__(self, pdf_file):
        
        self.pp = PdfPages(pdf_file)
                
        plt.close('peak-envelope')

        self.fig = plt.figure(num='peak-envelope', figsize=(12,4))
        gs = gridspec.GridSpec(1, 3, figure=self.fig, wspace=0.333, width_ratios=[0.2,0.4,0.4])
        
        gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.25)
        gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1], width_ratios=[0.8,0.2], height_ratios=[0.2,0.8])
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[2], width_ratios=[0.5,0.5], height_ratios=[0.5,0.5], wspace=0.166, hspace=0.166)
        
        self.ax_Q = self.fig.add_subplot(gs0[0,0])
        self.ax_Q2 = self.fig.add_subplot(gs0[1,0])
        
        self.ax_Q.minorticks_on()
        self.ax_Q2.minorticks_on()
        
        self.ax_p_scat = self.fig.add_subplot(gs1[1,0])
        self.ax_pu = self.fig.add_subplot(gs1[0,0])
        self.ax_pv = self.fig.add_subplot(gs1[1,1])
        self.ax_pe = self.fig.add_subplot(gs1[0,1])
        
        self.ax_p_scat.minorticks_on()
        self.ax_pu.minorticks_on()
        self.ax_pv.minorticks_on()
        
        self.ax_Qu = self.fig.add_subplot(gs2[0,0])
        self.ax_Qv = self.fig.add_subplot(gs2[1,0])
        self.ax_uv = self.fig.add_subplot(gs2[1,1])
        self.ax_ = self.fig.add_subplot(gs2[0,1])
        
        self.ax_Qu.minorticks_on()
        self.ax_Qv.minorticks_on()
        self.ax_uv.minorticks_on()
        
        self.ax_pu.get_xaxis().set_visible(False)
        self.ax_pv.get_yaxis().set_visible(False)
                                                           
        self.ax_pe.set_aspect('equal')
        self.ax_p_scat.set_aspect('equal')
        
        self.ax_pe.axis('off')
        
        self.ax_p_scat.set_xlabel('Q\u2081 (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        self.ax_p_scat.set_ylabel('Q\u2082 (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        
        self.ax_pu.set_ylim([-0.05,1.05])
        self.ax_pv.set_xlim([-0.05,1.05])

        self.ax_Q.set_rasterization_zorder(100)
        self.ax_Q.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        
        self.line_Q, self.caps_Q, self.bars_Q = self.ax_Q.errorbar([0], [0], yerr=[0], fmt='.-', rasterized=True, zorder=2)
        self.norm_Q = self.ax_Q.plot([0], [0], '--', rasterized=True, zorder=1)
        self.Q = self.ax_Q.plot([0], [0], ':.', rasterized=True, zorder=0)

        self.scat_p = self.ax_p_scat.scatter([0,1], [0,1], c=[1,2], s=5, marker='o', zorder=1, cmap=plt.cm.viridis, norm=mpl.colors.LogNorm(), rasterized=True) 
        self.elli_p = self.ax_p_scat.plot([0,1], [0,1], color='C3', zorder=10000000, rasterized=True)
        
        self.norm_p0 = self.ax_pu.plot([0,1], [0,1], '-', color='C3', rasterized=True)
        self.line_p0 = self.ax_pu.plot([0,1], [0,1], '.', color='C4', rasterized=True)
        self.norm_p1 = self.ax_pv.plot([0,1], [0,1], '-', color='C3', rasterized=True)
        self.line_p1 = self.ax_pv.plot([0,1], [0,1], '.', color='C4', rasterized=True)
        
        self.ax_Q2.set_rasterization_zorder(100)
        self.ax_Q2.set_xlabel('Q (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        self.ax_Q2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        self.line_Q2, self.caps_Q2, self.bars_Q2 = self.ax_Q2.errorbar([0], [0], yerr=[0], fmt='.-', rasterized=True, zorder=2)
        self.norm_Q2 = self.ax_Q2.plot([0], [0], '--', rasterized=True, zorder=1)
        self.Q2 = self.ax_Q2.plot([0], [0], ':.', rasterized=True, zorder=0)
        
        self.im_Qu = self.ax_Qu.imshow([[0,1],[0,1]], interpolation='nearest',
                                        origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())
                                        
        self.im_Qv = self.ax_Qv.imshow([[0,1],[0,1]], interpolation='nearest',
                                        origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())
                                        
        self.im_uv = self.ax_uv.imshow([[0,1],[0,1]], interpolation='nearest',
                                        origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())
                                        
        self.ax_Qu.set_rasterization_zorder(100)
        self.ax_Qv.set_rasterization_zorder(100)
        self.ax_uv.set_rasterization_zorder(100)

        self.peak_pu = self.ax_Qu.plot([0,1], [0,1], '-', color='C3', zorder=10, rasterized=True)
        self.inner_pu = self.ax_Qu.plot([0,1], [0,1], ':', color='C3', zorder=10, rasterized=True)
        self.outer_pu = self.ax_Qu.plot([0,1], [0,1], '--', color='C3', zorder=10, rasterized=True)
        
        self.peak_pv = self.ax_Qv.plot([0,1], [0,1], '-', color='C3', zorder=10, rasterized=True)
        self.inner_pv = self.ax_Qv.plot([0,1], [0,1], ':', color='C3', zorder=10, rasterized=True)
        self.outer_pv = self.ax_Qv.plot([0,1], [0,1], '--', color='C3', zorder=10, rasterized=True)
        
        self.peak_uv = self.ax_uv.plot([0,1], [0,1], '-', color='C3', zorder=10, rasterized=True)
        self.inner_uv = self.ax_uv.plot([0,1], [0,1], ':', color='C3', zorder=10, rasterized=True)
        self.outer_uv = self.ax_uv.plot([0,1], [0,1], '--', color='C3', zorder=10, rasterized=True)
        
        self.ax_Qv.set_xlabel('Q (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        self.ax_uv.set_xlabel('Q\u2081\u2032 (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        
        self.ax_Qu.set_ylabel('Q\u2081\u2032 (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        self.ax_Qv.set_ylabel('Q\u2082\u2032 (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        
        self.ax_Qu.get_xaxis().set_visible(False)
        self.ax_uv.get_yaxis().set_visible(False)
        
        self.ax_uv.get_xaxis().set_visible(False)
        
        self.ax_.set_aspect('equal')
        
        self.ax_.axis('off')
        
        self.__show_plots = True
        
    def clear_plots(self):
        
        self.ax_Q.set_title('')
        
        barsy, = self.bars_Q
                
        self.line_Q.set_data([0],[0])
        
        barsy.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip([0], [0], [0])])
        
        self.norm_Q[0].set_data([0],[0])
        
        self.Q[0].set_data([0],[0])

        self.ax_Q.relim()
        self.ax_Q.autoscale()

        barsy, = self.bars_Q2
        
        # ---
                
        self.line_Q2.set_data([0],[0])
        
        barsy.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip([0], [0], [0])])
        
        self.norm_Q2[0].set_data([0],[0])
        
        self.Q2[0].set_data([0],[0])
        
        self.ax_Q2.relim()
        self.ax_Q2.autoscale()

        # ---
        
        self.ax_pu.set_title('')
                
        self.scat_p.set_array(np.array([1,2]))
        self.scat_p.autoscale()
        
        self.scat_p.set_offsets(np.c_[[0,1],[0,1]])

        self.line_p0[0].set_data([0,1],[0,0])
        self.line_p1[0].set_data([0,0],[0,1])
        
        self.norm_p0[0].set_data([0,1],[0,0])
        self.norm_p1[0].set_data([0,0],[0,1])
                            
        self.elli_p[0].set_data([0,1],[0,1])
        
        self.ax_p_scat.set_xlim([0,1])
        self.ax_p_scat.set_ylim([0,1])

        self.ax_pu.set_xlim([0,1])
        self.ax_pv.set_ylim([0,1])
        
        # ---
        
        self.im_Qu.set_array(np.c_[[0,1],[1,0]])
        self.im_Qv.set_array(np.c_[[0,1],[1,0]])
        self.im_uv.set_array(np.c_[[0,1],[1,0]])
        
        self.im_Qu.autoscale()
        self.im_Qv.autoscale()
        self.im_uv.autoscale()
        
        self.im_Qu.set_extent([0,1,0,1])
        self.im_Qv.set_extent([0,1,0,1])
        self.im_uv.set_extent([0,1,0,1])
        
        self.ax_Qu.set_aspect(1)
        self.ax_Qv.set_aspect(1)
        self.ax_uv.set_aspect(1)
        
        self.peak_pu[0].set_data([0,1],[0,1])
        self.peak_pv[0].set_data([0,1],[0,1])
        self.peak_uv[0].set_data([0,1],[0,1])
        
        self.inner_pu[0].set_data([0,1],[0,1])
        self.inner_pv[0].set_data([0,1],[0,1])
        self.inner_uv[0].set_data([0,1],[0,1])
        
        self.outer_pu[0].set_data([0,1],[0,1])
        self.outer_pv[0].set_data([0,1],[0,1])
        self.outer_uv[0].set_data([0,1],[0,1])
        
    def show_plots(self, show):
        
        self.__show_plots = show
        
    def create_pdf(self):
        
        self.pp.close()
               
    def plot_Q(self, key, x, y, y0, yerr, X, Y):
        
        self.ax_Q.set_title(key)
        
        barsy, = self.bars_Q
                
        self.line_Q.set_data(x,y)
        
        barsy.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(x, y+yerr, y-yerr)])
        
        self.norm_Q[0].set_data(X,Y)
        
        self.Q[0].set_data(x,y0)

        self.ax_Q.relim()
        self.ax_Q.autoscale()

        if self.__show_plots: self.fig.show()
        
        # self.pp.savefig(self.fig, dpi=144)
        
    def plot_extracted_Q(self, key, x, y, y0, yerr, X, Y):
                
        barsy, = self.bars_Q2
                
        self.line_Q2.set_data(x,y)
        
        barsy.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(x, y+yerr, y-yerr)])
        
        self.norm_Q2[0].set_data(X,Y)
        
        self.Q2[0].set_data(x,y0)
        
        self.ax_Q2.relim()
        self.ax_Q2.autoscale()
        
        if self.__show_plots: self.fig.show()
                
    def plot_projection(self, key, xs, ys, weights, limits, xu, yu, xv, yv, Xu, Yu, Xv, Yv):
        
        xlim = [limits[0][0], limits[0][1]]
        ylim = [limits[1][0], limits[1][1]]
        
        self.ax_pu.set_title('d = {:.4f} \u212B'.format(key))
        
        sort = np.argsort(weights)
        
        self.scat_p.set_array(weights[sort])
        self.scat_p.autoscale()
        
        self.scat_p.set_offsets(np.c_[xs[sort],ys[sort]])

        self.line_p0[0].set_data(xu,yu)
        self.line_p1[0].set_data(yv,xv)
        
        self.norm_p0[0].set_data(Xu,Yu)
        self.norm_p1[0].set_data(Yv,Xv)

        self.ax_p_scat.set_xlim(xlim)
        self.ax_p_scat.set_ylim(ylim)

        self.ax_pu.set_xlim(xlim)
        self.ax_pv.set_ylim(ylim)
        
        if self.__show_plots: self.fig.show()
                
    def plot_projection_ellipse(self, xe, ye, x0, y0, x1, y1):
            
        self.elli_p[0].set_data(xe,ye)
        
        if self.__show_plots: self.fig.show()
        
    def plot_integration(self, signal, u_extents, v_extents, Q_extents, 
                         x_pk_Qu, y_pk_Qu, x_pk_Qv, y_pk_Qv, x_pk_uv, y_pk_uv, 
                         x_in_Qu, y_in_Qu, x_in_Qv, y_in_Qv, x_in_uv, y_in_uv, 
                         x_out_Qu, y_out_Qu, x_out_Qv, y_out_Qv, x_out_uv, y_out_uv):
    
        Qu = np.nansum(signal, axis=1)
        Qv = np.nansum(signal, axis=0)
        uv = np.nansum(signal, axis=2)
    
        self.im_Qu.set_array(Qu)
        self.im_Qv.set_array(Qv)
        self.im_uv.set_array(uv)
        
        self.im_Qu.autoscale()
        self.im_Qv.autoscale()
        self.im_uv.autoscale()

        Qu_extents = [Q_extents[0],Q_extents[2],u_extents[0],u_extents[2]]
        Qv_extents = [Q_extents[0],Q_extents[2],v_extents[0],v_extents[2]]
        uv_extents = [u_extents[0],u_extents[2],v_extents[0],v_extents[2]]
        
        self.im_Qu.set_extent(Qu_extents)
        self.im_Qv.set_extent(Qv_extents)
        self.im_uv.set_extent(uv_extents)
        
        Qu_aspect = Q_extents[1]/u_extents[1]
        Qv_aspect = Q_extents[1]/v_extents[1]
        uv_aspect = u_extents[1]/v_extents[1]
        
        self.ax_Qu.set_aspect(Qu_aspect)
        self.ax_Qv.set_aspect(Qv_aspect)
        self.ax_uv.set_aspect(uv_aspect)
        
        self.peak_pu[0].set_data(x_pk_Qu, y_pk_Qu)
        self.peak_pv[0].set_data(x_pk_Qv, y_pk_Qv)
        self.peak_uv[0].set_data(x_pk_uv, y_pk_uv)
        
        self.inner_pu[0].set_data(x_in_Qu, y_in_Qu)
        self.inner_pv[0].set_data(x_in_Qv, y_in_Qv)
        self.inner_uv[0].set_data(x_in_uv, y_in_uv)

        self.outer_pu[0].set_data(x_out_Qu, y_out_Qu)
        self.outer_pv[0].set_data(x_out_Qv, y_out_Qv)
        self.outer_uv[0].set_data(x_out_uv, y_out_uv)
        
        if self.__show_plots: self.fig.show()

        self.pp.savefig(self.fig, dpi=144)

class PeakInformation:
    
    def __init__(self, scale_constant):
        
        self.__peak_num = 0
        self.__run_num = []
        self.__bank_num = []
        self.__peak_ind = []
        self.__row = []
        self.__col = []

        self.__A = np.eye(3)
        self.__bin_size = np.zeros(3)
        self.__Q = np.zeros(3)

        self.__peak_fit = 0.0
        self.__peak_bkg_ratio = 0.0
        self.__peak_score = 0.0

        self.__pk_data = None
        self.__pk_norm = None
        
        self.__bkg_data = None
        self.__bkg_norm = None
        
        self.__norm_scale = np.array([])
        
        self.__wl = []
        self.__two_theta = []
        self.__az_phi = []

        self.__phi = []
        self.__chi = []
        self.__omega = []
        
        self.__est_int = []
        self.__est_int_err = []
    
        self.__scale_constant = scale_constant
        
    def get_Q(self):
        
        return self.__Q
        
    def get_A(self):
        
        return self.__A
        
    def update_scale_constant(self, scale_constant):
        
        self.__scale_constant = scale_constant
        
    def update_norm_scale(self, bank_scale):
        
        norm_scale = []
        for bank in self.__bank_num:
            norm_scale.append(bank_scale[bank])
        
        self.__norm_scale = np.array(norm_scale)
    
    def set_peak_number(self, peak_num):
        
        self.__peak_num = peak_num
        
    def get_peak_number(self):
        
        return self.__peak_num
        
    def get_run_numbers(self):
        
        return self.__run_num
        
    def get_peak_indices(self):
        
        return self.__peak_ind
        
    def get_merged_intensity(self):
        
        return self.__merge_intensity()
        
    def get_merged_intensity_error(self):
        
        return self.__merge_intensity_error()
        
    def get_goniometers(self):
        
        R = []
        
        for phi, chi, omega in zip(self.__phi,self.__chi,self.__omega):
            R.append(np.dot(self.__R_y(omega), np.dot(self.__R_z(chi), self.__R_y(phi))))
            
        return R
        
    def __R_x(self, angle):
        
        t = np.deg2rad(angle)
        
        return np.array([[1,0,0],[0,np.cos(t),-np.sin(t)],[0,np.sin(t),np.cos(t)]])
        
    def __R_y(self, angle):
        
        t = np.deg2rad(angle)
        
        return np.array([[np.cos(t),0,np.sin(t)],[0,1,0],[-np.sin(t),0,np.cos(t)]])

    def __R_z(self, angle):
        
        t = np.deg2rad(angle)
        
        return np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
        
    def dictionary(self):
        
        return { 'PeakNumber': self.__peak_num,
                 'RunNumber': self.__run_num,
                 'BankNumber': self.__bank_num,
                 'PeakIndex': self.__peak_ind,
                 'Row': self.__row,
                 'Col': self.__col,
                 'MergedIntensity': np.round(self.__merge_intensity(),2),
                 'MergedIntensitySigma': np.round(self.__merge_intensity_error(),2),
                 'MergedPeakBackgroundRatio': np.round(self.__merge_pk_bkg_ratio(),2),
                 'Ellispoid': np.round(self.__A[np.triu_indices(3)],2).tolist(),
                 'BinSize': np.round(self.__bin_size.tolist(),3).tolist(),
                 'Q': np.round(self.__Q,3).tolist(),
                 'PeakQFit': np.round(self.__peak_fit,2),
                 'PeakBackgroundRatio': np.round(self.__peak_bkg_ratio,2),
                 'PeakScore-2D-Std(int)/Std(bg)': np.round(self.__peak_score,2),
                 'Intensity': np.round(self.__intensity(),2).tolist(),
                 'IntensitySigma': np.round(self.__intensity_error(),2).tolist(),
                 'PeakBackgroundRatio': np.round(self.__pk_bkg_ratio(),2).tolist(),
                 'NormalizationScale': np.round(self.__norm_scale,2).tolist(),
                 'Wavelength': np.round(self.__wl,2).tolist(),
                 'ScatteringAngle': np.round(np.rad2deg(self.__two_theta),2).tolist(),
                 'AzimuthalAngle': np.round(np.rad2deg(self.__az_phi),2).tolist(),
                 'GoniometerPhiAngle': np.round(self.__phi,2).tolist(),
                 'GoniometerChiAngle': np.round(self.__chi,2).tolist(),
                 'GoniometerOmegaAngle': np.round(self.__omega,2).tolist(),
                 'EstimatedIntensity': np.round(self.__est_int,2).tolist(),
                 'EstimatedIntensitySigma': np.round(self.__est_int_err,2).tolist() }

    def add_information(self, run, bank, ind, row, col, wl, two_theta, az_phi, 
                              phi, chi, omega, intens, sig):
        
        self.__run_num.append(run)
        self.__bank_num.append(bank)
        self.__peak_ind.append(ind)
        self.__row.append(row)
        self.__col.append(col)
        self.__wl.append(wl)
        self.__two_theta.append(two_theta)
        self.__az_phi.append(az_phi)
        self.__phi.append(phi)
        self.__chi.append(chi)
        self.__omega.append(omega)
        self.__est_int.append(intens)
        self.__est_int_err.append(sig)
                
    def add_integration(self, Q, A, peak_fit, peak_bkg_ratio, peak_score, data):
        
        self.__Q = Q
        self.__A = A
        
        pk_data, pk_norm, bkg_data, bkg_norm, bin_size = data
        
        self.__bin_size = bin_size
        
        self.__peak_fit = peak_fit
        self.__peak_bkg_ratio = peak_bkg_ratio
        self.__peak_score = peak_score
        
        self.__pk_data = pk_data
        self.__pk_norm = pk_norm
        
        self.__bkg_data = bkg_data
        self.__bkg_norm = bkg_norm
        
        self.__norm_scale = np.ones(len(pk_data))
        
    def __merge_pk_bkg_ratio(self):

        if self.__peak_num == 0 or len(self.__norm_scale) == 0:

            return 0.0

        else:

            pk_vol = np.sum(~np.isnan([self.__pk_data,self.__pk_norm]).any(axis=0))
            bkg_vol = np.sum(~np.isnan([self.__bkg_data,self.__bkg_norm]).any(axis=0))

            return pk_vol/bkg_vol

    def __merge_intensity(self):

        if self.__peak_num == 0 or len(self.__norm_scale) == 0:

            return 0.0

        else:
        
            data = self.__pk_data
            norm = self.__pk_norm
            
            bkg_data = self.__bkg_data
            bkg_norm = self.__bkg_norm
            
            scale_norm = self.__norm_scale[:,np.newaxis]
            volume_ratio = self.__merge_pk_bkg_ratio()
            
            bin_size = self.__bin_size 
            constant = self.__scale_constant*np.prod(bin_size)
            
            data_norm = np.nansum(data, axis=0)/np.nansum(np.multiply(norm, scale_norm), axis=0)
            bkg_data_norm = np.nansum(bkg_data, axis=0)/np.nansum(np.multiply(bkg_norm, scale_norm), axis=0)
            
            data_norm[np.isinf(data_norm)] = np.nan
            bkg_data_norm[np.isinf(bkg_data_norm)] = np.nan

            intens = np.nansum(data_norm)
            bkg_intens = np.nanmedian(bkg_data_norm)
            
            intensity = (intens-bkg_intens*volume_ratio)*constant
            
            return intensity
        
    def __merge_intensity_error(self):
        
        if self.__peak_num == 0 or len(self.__norm_scale) == 0:
            
            return 0.0
            
        else:
            
            data = self.__pk_data
            norm = self.__pk_norm
            
            bkg_data = self.__bkg_data
            bkg_norm = self.__bkg_norm
                        
            scale_norm = self.__norm_scale[:,np.newaxis]
            volume_ratio = self.__merge_pk_bkg_ratio()
            
            bin_size = self.__bin_size 
            constant = self.__scale_constant*np.prod(bin_size)
            
            norm_sum = np.nansum(np.multiply(norm, scale_norm), axis=0)
            bkg_norm_sum = np.nansum(np.multiply(bkg_norm, scale_norm), axis=0)
                        
            var_data_norm = np.nansum(data, axis=0)/norm_sum
            var_bkg_data_norm = np.nansum(bkg_data, axis=0)/bkg_norm_sum
            
            top_5_percent = np.sort(norm_sum)[-norm_sum.size//20:-1].mean()
            top_5_percent_bkg = np.sort(bkg_norm_sum)[-bkg_norm_sum.size//20:-1].mean()
            
            sig_data_norm = np.sqrt(np.nansum(var_data_norm*top_5_percent))/top_5_percent
            sig_data_norm_bkg = np.sqrt(np.nansum(var_data_norm*top_5_percent))/top_5_percent
            
            sig_intensity = np.sqrt(sig_data_norm**2+np.multiply(sig_data_norm_bkg,volume_ratio)**2)*constant
            
            return sig_intensity
            
    def __pk_bkg_ratio(self):
        
        if self.__peak_num == 0 or len(self.__norm_scale) == 0:
            
            return np.array([])
            
        else:
                        
            pk_vol = np.sum(~np.isnan([self.__pk_data,self.__pk_norm]).any(axis=0),axis=1)
            bkg_vol = np.sum(~np.isnan([self.__bkg_data,self.__bkg_norm]).any(axis=0),axis=1)
                                       
            return pk_vol/bkg_vol
        
    def __intensity(self):
        
        if self.__peak_num == 0 or len(self.__norm_scale) == 0:
            
            return np.array([])
            
        else:
            
            data = self.__pk_data
            norm = self.__pk_norm
            
            bkg_data = self.__bkg_data
            bkg_norm = self.__bkg_norm
            
            scale_norm = self.__norm_scale[:,np.newaxis]
            volume_ratio = self.__pk_bkg_ratio()
            
            bin_size = self.__bin_size 
            constant = self.__scale_constant*np.prod(bin_size)
            
            data_norm = data/np.multiply(norm, scale_norm)
            bkg_data_norm = bkg_data/np.multiply(bkg_norm, scale_norm)
            
            data_norm[np.isinf(data_norm)] = 0
            bkg_data_norm[np.isinf(bkg_data_norm)] = 0
            
            intens = np.nansum(data_norm, axis=1)
            bkg_intens = np.nansum(bkg_data_norm, axis=1)
                        
            intensity = (intens-np.multiply(bkg_intens,volume_ratio))*constant
            
            return intensity
        
    def __intensity_error(self):
        
        if self.__peak_num == 0 or len(self.__norm_scale) == 0:
            
            return np.array([])
            
        else:
                
            data = self.__pk_data
            norm = self.__pk_norm
            
            bkg_data = self.__bkg_data
            bkg_norm = self.__bkg_norm
            
            scale_norm = self.__norm_scale[:,np.newaxis]
            volume_ratio = self.__pk_bkg_ratio()
            
            bin_size = self.__bin_size 
            constant = self.__scale_constant*np.prod(bin_size)
                        
            sig_data_norm = np.sqrt(data)/np.multiply(norm, scale_norm)
            sig_bkg_data_norm = np.sqrt(bkg_data)/np.multiply(bkg_norm, scale_norm)
            
            sig_data_norm[np.isinf(sig_data_norm)] = 0
            sig_bkg_data_norm[np.isinf(sig_bkg_data_norm)] = 0
            
            sig_intens = np.nansum(sig_data_norm, axis=1)
            sig_bkg_intens = np.nansum(sig_bkg_data_norm, axis=1)

            sig_intensity = (sig_intens+np.multiply(sig_bkg_intens,volume_ratio))*constant
            
            return sig_intensity

class PeakDictionary:

    def __init__(self, a, b, c, alpha, beta, gamma):

        self.peak_dict = { }

        self.set_constants(a, b, c, alpha, beta, gamma)
        
        self.scale_constant = 1e+8
        
        self.g = Goniometer()
    
    def __call_peak(self, h, k, l):
        
        key = (h,k,l)
        
        d_spacing = self.__get_d_spacing(h,k,l)
        
        if d_spacing is None:
            
            print('Peak does not exist')
        
        else:
            
            print('{} {:2.4f} (\u212B)'.format(key, d_spacing))
            pprint.pprint(self.peak_dict.get(key).dictionary())

    __call__ = __call_peak
    
    def get_d(self, h, k, l):
        
        ol = self.iws.sample().getOrientedLattice()
        
        d_spacing = ol.d(V3D(h,k,l))
    
        return d_spacing
    
    def set_scale_constant(self, constant):
        
        self.scale_constant = constant
        
        for key in self.peak_dict.keys():
            
            peak = self.peak_dict.get(key)
            
            peak.update_scale_constant(constant)
     
    def set_bank_constant(self, bank_scale):
        
        for key in self.peak_dict.keys():
            
            peak = self.peak_dict.get(key)
            
            peak.update_norm_scale(bank_scale)
            
    def set_constants(self, a, b, c, alpha, beta, gamma):
        
        CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace='pws')
        CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace='iws')
        CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace='cws')
                
        self.pws = mtd['pws']
        self.iws = mtd['iws']
        self.cws = mtd['cws']
        
        SetUB(Workspace=self.pws, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        SetUB(Workspace=self.iws, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        SetUB(Workspace=self.cws, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

        self.__reset_peaks()
        
    def __get_d_spacing(self, h, k, l):

        peak = self.__get_peak(h, k, l)
        
        if peak is not None:
            
            peak_num = peak.get_peak_number()
                                    
            return self.pws.getPeak(peak_num-1).getDSpacing()

    def __get_peak(self, h, k, l):
                
        return self.peak_dict.get((h,k,l))
        
    def __reset_peaks(self):
                
        for key in self.peak_dict.keys():
            
            peak = self.peak_dict.get(key)
            
            peak_num = peak.get_peak_number()
            intens = peak.get_merged_intensity()
            sig_intens = peak.get_merged_intensity_error()
            
            h, k, l = key
    
            pk = self.pws.createPeakHKL(V3D(h,k,l))
            pk.setIntensity(intens)
            pk.setSigmaIntensity(sig_intens)
            pk.setPeakNumber(peak_num)
            self.pws.addPeak(pk)
            
    def add_peaks(self, ws):
        
        if not mtd.doesExist('pws'):
            print('pws does not exist')
        
        if mtd.doesExist(ws):
            
            pws = mtd[ws]

            for p in range(pws.getNumberPeaks()):
                
                bank = pws.row(p)['BankName']
                
                peak = pws.getPeak(p)
                
                intens = peak.getIntensity()
                sig_intens = peak.getSigmaIntensity()
                
                if bank != 'None' and intens > 0 and sig_intens > 0:
                            
                    h, k, l = peak.getHKL()
                    
                    h, k, l = int(h), int(k), int(l)
                
                    key = (h,k,l)
                    
                    if self.peak_dict.get(key) is None:
                        
                        peak_num = self.pws.getNumberPeaks()+1
                    
                        self.peak_dict[key] = PeakInformation(self.scale_constant)
                        self.peak_dict[key].set_peak_number(peak_num)
                                        
                        pk = self.pws.createPeakHKL(V3D(h,k,l))
                        pk.setPeakNumber(peak_num)
                        
                        self.pws.addPeak(pk)
                    
                    run = peak.getRunNumber()
                    if run == 0 and ws.split('_')[-2].isnumeric():
                        _, exp, run, _ = ws.split('_')
                    
                    bank = 1 if bank == 'panel' else int(bank.strip('bank'))
                    ind = peak.getPeakNumber()
                    row = int(pws.row(p)['Row'])
                    col = int(pws.row(p)['Col'])
                    
                    wl = peak.getWavelength()
                    two_theta = peak.getScattering()
                    az_phi = peak.getAzimuthal()
                    
                    R = peak.getGoniometerMatrix()
                    self.g.setR(R)
                    phi, chi, omega = self.g.getEulerAngles('YZY') 
                    
                    if np.isclose(chi,0) and np.isclose(omega,0):
                        phi = np.rad2deg(np.arctan2(R[0,2],R[0,0]))
                        
                    self.peak_dict[key].add_information(run, bank, ind, row, col,  
                                                        wl, two_theta, az_phi, 
                                                        phi, chi, omega, 
                                                        intens, sig_intens)
        else:
            print('{} does not exist'.format(ws))
                
    def to_be_integrated(self):
                
        SortPeaksWorkspace(self.pws, ColumnNameToSortBy='DSpacing', SortAscending=False, OutputWorkspace=self.pws)  
        
        peak_dict = {}
        
        for p in range(self.pws.getNumberPeaks()):
            
            peak = self.pws.getPeak(p)
        
            h, k, l = peak.getHKL()
            
            h, k, l = int(h), int(k), int(l)
        
            key = (h,k,l)
            
            pk = self.peak_dict.get(key)
            
            peak_dict[key] = pk.get_run_numbers(), pk.get_peak_indices()
            
        return peak_dict
        
    def integrated_result(self, key, Q, A, peak_fit, peak_bkg_ratio, peak_score, data):

        peak = self.peak_dict[key]
        peak.add_integration(Q, A, peak_fit, peak_bkg_ratio, peak_score, data)
                        
        h, k, l = key
        Qx, Qy, Qz = Q

        peak_num = self.iws.getNumberPeaks()+1
        intens = peak.get_merged_intensity()
        sig_intens = peak.get_merged_intensity_error()
        
        pk = self.iws.createPeakHKL(V3D(h,k,l))
        pk.setPeakNumber(peak_num)
        pk.setIntensity(intens)
        pk.setSigmaIntensity(sig_intens)
        self.iws.addPeak(pk)
        
        peak_num = self.cws.getNumberPeaks()+1

        pk = self.cws.createPeakHKL(V3D(h,k,l))
        pk.setQSampleFrame(V3D(Qx,Qy,Qz))
        pk.setPeakNumber(peak_num)
        pk.setIntensity(intens)
        pk.setSigmaIntensity(sig_intens)
        self.cws.addPeak(pk)
        
    def calibrated_result(self, key, run_num, Q):
        
        peak = self.peak_dict[key]

        h, k, l = key
        Qx, Qy, Qz = Q

        peak_num = self.cws.getNumberPeaks()+1
        
        runs = peak.get_run_numbers()
        goniometer = peak.get_goniometers()[runs.index(run_num)]

        pk = self.cws.createPeakHKL(V3D(h,k,l))
        pk.setGoniometerMatrix(goniometer)
        pk.setQSampleFrame(V3D(Qx,Qy,Qz))
        pk.setPeakNumber(peak_num)
        pk.setRunNumber(run_num)
        self.cws.addPeak(pk)
            
    def save_hkl(self, filename, magnetic=False):
        
        SortPeaksWorkspace(self.iws, ColumnNameToSortBy='DSpacing', SortAscending=False, OutputWorkspace=self.iws)  
        
        ol = self.iws.sample().getOrientedLattice()
        
        if magnetic:
            hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:8.4f}\n'
        else:
            hkl_format = '{:4.0f}{:4.0f}{:4d}{:8.2f}{:8.2f}{:8.4f}\n'
        
        with open(filename, 'w') as f:
            
            for p in range(self.iws.getNumberPeaks()):
                
                pk = self.iws.getPeak(p)
                intens, sig_intens = pk.getIntensity(), pk.getSigmaIntensity()
                
                if (intens > 0 and sig_intens > 0 and intens/sig_intens > 3):

                    h, k, l = pk.getH(), pk.getK(), pk.getL()
                    d_spacing = ol.d(V3D(h,k,l))
                    
                    if magnetic:
                        f.write(hkl_format.format(h, k, l, 1, intens, sig_intens, d_spacing))
                    else:
                        f.write(hkl_format.format(h, k, l, intens, sig_intens, d_spacing))
                        
    def save_calibration(self, filename):
        
        SaveNexus(self.cws, Filename=filename)
                        
    def save(self, filename):
        
        with open(filename, 'wb') as f:
            
            pickle.dump(self.peak_dict, f)
            
    def load(self, filename):
        
        ol = self.pws.sample().getOrientedLattice()
        
        a, b, c, alpha, beta, gamma = ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma()
        
        self.set_constants(a, b, c, alpha, beta, gamma)

        with open(filename, 'rb') as f:
            
            self.peak_dict = pickle.load(f)
            
        self.__reset_peaks()
        
        for key in self.peak_dict.keys():
            
            peak = self.peak_dict.get(key)
            
            peak_num = peak.get_peak_number()
            intens = peak.get_merged_intensity()
            sig_intens = peak.get_merged_intensity_error()
            
            h, k, l = key
            Qx, Qy, Qz = peak.get_Q()

            peak_num = self.iws.getNumberPeaks()+1
            
            pk = self.iws.createPeakHKL(V3D(h,k,l))
            pk.setPeakNumber(peak_num)
            pk.setIntensity(intens)
            pk.setSigmaIntensity(sig_intens)
            self.iws.addPeak(pk)
            
            pk = self.cws.createPeakHKL(V3D(h,k,l))
            pk.setQSampleFrame(V3D(Qx,Qy,Qz))
            pk.setPeakNumber(peak_num)
            pk.setIntensity(intens)
            pk.setSigmaIntensity(sig_intens)
            self.cws.addPeak(pk)
            
        SortPeaksWorkspace(self.pws, ColumnNameToSortBy='DSpacing', SortAscending=False, OutputWorkspace=self.pws)  
        SortPeaksWorkspace(self.iws, ColumnNameToSortBy='DSpacing', SortAscending=False, OutputWorkspace=self.iws)  
            
def box_integrator(runs, indices, binsize=0.001, radius=0.15, exp=None):

    for i, r in enumerate(runs):
        
        if exp is None:
            ows = 'COR_'+str(r)
        else:
            ows = 'HB3A_'+str(exp)+'_'+str(r)
            
        omd = ows+'_md'
        opk = ows+'_pks'

        if i == 0:            
            
            Q0 = mtd[opk].getPeak(indices[i]).getQSampleFrame()
            #UB = mtd[opk].sample().getOrientedLattice().getUB()
                        
            Qr = np.array([radius,radius,radius])
                                                            
            nQ = np.round(2*Qr/binsize).astype(int)+1
            
            Qmin, Qmax = Q0-Qr, Q0+Qr
                                  
        BinMD(InputWorkspace=omd,
              AlignedDim0='Q_sample_x,{},{},{}'.format(Qmin[0],Qmax[0],nQ[0]),
              AlignedDim1='Q_sample_y,{},{},{}'.format(Qmin[1],Qmax[1],nQ[1]),
              AlignedDim2='Q_sample_z,{},{},{}'.format(Qmin[2],Qmax[2],nQ[2]),
              OutputWorkspace='__tmp')
           
        if i == 0:   
            
            box = mtd['__tmp']*0
                                      
        PlusMD('box', '__tmp', OutputWorkspace='box')                                  
    
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
    
    return Q, Qx[mask], Qy[mask], Qz[mask], weights, Q0
    
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
            return 0, np.nan, np.nan, np.nan
            
        if (pos_v_data.sum() > 0):
            v_data -= v_data[pos_v_data].min()
        else:
            return 0, np.nan, np.nan, np.nan
        
        u_data[u_data < 0] = 0
        v_data[v_data < 0] = 0

        if (u_data > 0).any() and (v_data > 0).any():
            u_center = np.average(u_bin_centers, weights=u_data**2)
            v_center = np.average(v_bin_centers, weights=v_data**2)
            
            u_variance = np.average((u_bin_centers-u_center)**2, weights=u_data**2)
            v_variance = np.average((v_bin_centers-v_center)**2, weights=v_data**2)
        else:
            print('First pass failure 2d covariance')
            return 0, np.nan, np.nan, np.nan
                    
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
            return 0, np.nan, np.nan, np.nan
            
        if (pos_v_data.sum() > 0):
            v_data -= v_data[pos_v_data].min()
        else:
            return 0, np.nan, np.nan, np.nan
        
        u_data[u_data < 0] = 0
        v_data[v_data < 0] = 0

        if (u_data > 0).any() and (v_data > 0).any():
            u_center = np.average(u_bin_centers, weights=u_data**2)
            v_center = np.average(v_bin_centers, weights=v_data**2)
            
            u_variance = np.average((u_bin_centers-u_center)**2, weights=u_data**2)
            v_variance = np.average((v_bin_centers-v_center)**2, weights=v_data**2)
        else:
            print('Second pass failure for 2d covariance')
            return 0, np.nan, np.nan, np.nan

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
            return 0, np.nan, np.nan, np.nan
            
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
            return 0, np.nan, np.nan, np.nan
        
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
            return 0, np.nan, np.nan, np.nan
            
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
            return 0, np.nan, np.nan, np.nan
        
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
        return 0, np.nan, np.nan
        
def ellipsoid(peak_envelope, Q0, center, variance, center2d, covariance2d, n, u, v, xsigma=3, lscale=5.99, plot='first'):
    
    Q_offset_1d = (center-np.linalg.norm(Q0))*n
    Q_offset_2d = u*center2d[0]+v*center2d[1]
  
    Q = Q0+Q_offset_1d+Q_offset_2d 
    
    radius = xsigma*np.sqrt(variance)

    eigenvalues, eigenvectors = np.linalg.eig(covariance2d)
    
    radii = lscale*np.sqrt(eigenvalues)
            
    t = np.linspace(0,2*np.pi,100)
    x, y = radii[0]*np.cos(t), radii[1]*np.sin(t)
    
    xe = eigenvectors[0,0]*x+eigenvectors[0,1]*y+center2d[0]
    ye = eigenvectors[1,0]*x+eigenvectors[1,1]*y+center2d[1]
    
    dx0, dy0 = eigenvectors[0,0]*radii[0], eigenvectors[1,0]*radii[0]
    dx1, dy1 = eigenvectors[0,1]*radii[1], eigenvectors[1,1]*radii[1]
    
    x0, y0 = [center2d[0],center2d[0]+dx0], [center2d[1],center2d[1]+dy0]
    x1, y1 = [center2d[0],center2d[0]+dx1], [center2d[1],center2d[1]+dy1]
       
    if plot == 'second': 
        peak_envelope.plot_extracted_projection_ellipse(xe, ye, x0, y0, x1, y1)
    elif plot == 'first':
        peak_envelope.plot_projection_ellipse(xe, ye, x0, y0, x1, y1)

    Qu = (u*eigenvectors[0,0]+v*eigenvectors[1,0])*radii[0]
    Qv = (u*eigenvectors[0,1]+v*eigenvectors[1,1])*radii[1]
    
    # Defining eigenvectors
    W = np.zeros((3,3))
    W[:,0] = Qu/np.linalg.norm(Qu)
    W[:,1] = Qv/np.linalg.norm(Qv)
    W[:,2] = n
    
    # Defining eigenvectors
    D = np.zeros((3,3))
    D[0,0] = 1/radii[0]**2
    D[1,1] = 1/radii[1]**2
    D[2,2] = 1/radius**2
    
    A = (np.dot(np.dot(W, D), W.T))
    
    return Q, A, W, D
    
def partial_integration(signal, Qx, Qy, Qz, Q_rot, D_pk, D_bkg_in, D_bkg_out):
             
    mask = D_pk[0,0]*(Qx-Q_rot[0])**2\
         + D_pk[1,1]*(Qy-Q_rot[1])**2\
         + D_pk[2,2]*(Qz-Q_rot[2])**2 <= 1

    pk = signal[mask].astype('f')
    
    mask = (D_bkg_in[0,0]*(Qx-Q_rot[0])**2\
           +D_bkg_in[1,1]*(Qy-Q_rot[1])**2\
           +D_bkg_in[2,2]*(Qz-Q_rot[2])**2 >= 1)\
         & (D_bkg_out[0,0]*(Qx-Q_rot[0])**2\
           +D_bkg_out[1,1]*(Qy-Q_rot[1])**2\
           +D_bkg_out[2,2]*(Qz-Q_rot[2])**2 <= 1)

    bkg = signal[mask].astype('f')
    
    return pk, bkg
    
def norm_integrator(peak_envelope, runs, Q0, D, W, bin_size=0.013, box_size=1.65, 
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
    
    dQp[0] = np.min([Q0_bin_size,bin_size])
    dQp[1] = np.min([Q1_bin_size,bin_size])
    dQp[2] = Q_bin_size
    
    pk_data, pk_norm = [], []
    bkg_data, bkg_norm = [], []
    
    if mtd.doesExist('dataMD'): DeleteWorkspace('dataMD')
    if mtd.doesExist('normMD'): DeleteWorkspace('normMD')
        
    for i, r in enumerate(runs):
        
        if exp is None:
            ows = 'COR_'+str(r)
        else:
            ows = 'HB3A_'+str(exp)+'_'+str(r)

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
            
            extents = [Q_rot[0]-dQ[0],Q_rot[0]+dQ[0],Q_rot[1]-dQ[1],Q_rot[1]+dQ[1],Q_rot[2]-dQ[2],Q_rot[2]+dQ[2]]
            bins = [int(round(2*dQ[0]/dQp[0]))+1,int(round(2*dQ[1]/dQp[1]))+1,int(round(2*dQ[2]/dQp[2]))+1]
                
        if exp is None:
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
                  
            DivideMD(LHSWorkspace='tmpDataMD', RHSWorkspace='tmpNormMD', OutputWorkspace='__normDataMD')

        if i == 0:
            Qxaxis = mtd['__normDataMD'].getXDimension()
            Qyaxis = mtd['__normDataMD'].getYDimension()
            Qzaxis = mtd['__normDataMD'].getZDimension()

            Qx = np.linspace(Qxaxis.getMinimum(), Qxaxis.getMaximum(), Qxaxis.getNBins()+1)
            Qy = np.linspace(Qyaxis.getMinimum(), Qyaxis.getMaximum(), Qyaxis.getNBins()+1)
            Qz = np.linspace(Qzaxis.getMinimum(), Qzaxis.getMaximum(), Qzaxis.getNBins()+1)

            Qx, Qy, Qz = 0.5*(Qx[:-1]+Qx[1:]), 0.5*(Qy[:-1]+Qy[1:]), 0.5*(Qz[:-1]+Qz[1:])

            Qx, Qy, Qz = np.meshgrid(Qx, Qy, Qz, indexing='ij', copy=False)

            u_extents = [Qxaxis.getMinimum(),Qxaxis.getMaximum()]
            v_extents = [Qyaxis.getMinimum(),Qyaxis.getMaximum()]
            Q_extents = [Qzaxis.getMinimum(),Qzaxis.getMaximum()]

        signal = mtd['tmpDataMD'].getSignalArray().copy()

        pk, bkg = partial_integration(signal, Qx, Qy, Qz, Q_rot, D_pk, D_bkg_in, D_bkg_out)

        pk_data.append(pk)
        bkg_data.append(bkg)

        mask = (D_bkg_in[0,0]*(Qx-Q_rot[0])**2\
               +D_bkg_in[1,1]*(Qy-Q_rot[1])**2\
               +D_bkg_in[2,2]*(Qz-Q_rot[2])**2 >= 1)\
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

        DeleteWorkspace('tmpDataMD')
        DeleteWorkspace('tmpNormMD')

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

    signal[mask] = np.nan
    mtd['normDataMD'].setSignalArray(signal)      

    return pk_data, pk_norm, bkg_data, bkg_norm, dQp
    
def pre_integration(ipts, runs, ub_file, spectrum_file, counts_file, 
                    tube_calibration, detector_calibration, 
                    reflection_condition):
    
    # peak prediction parameters ---------------------------------------------------
    min_wavelength = 0.63
    max_wavelength= 2.51
    min_d_spacing = 0.7
    max_d_spacing= 20
                     
    # peak centroid radius ---------------------------------------------------------
    centroid_radius = 0.125

    # goniometer axis --------------------------------------------------------------
    gon_axis = 'BL9:Mot:Sample:Axis3.RBV'

    if not mtd.doesExist('tube_table'):
        LoadNexus(Filename=tube_calibration, OutputWorkspace='tube_table')
    
    if not mtd.doesExist('sa'):
        LoadNexus(Filename=counts_file, OutputWorkspace='sa')
        
    if not mtd.doesExist('flux'):
        LoadNexus(Filename=spectrum_file, OutputWorkspace='flux')

    merge_md = []
    merge_pk = []

    for r in runs:
        print('Processing run : {}'.format(r))
        ows = 'COR_'+str(r)
        omd = ows+'_md'
        opk = ows+'_pks'
        merge_md.append(omd)
        merge_pk.append(opk)

        if not mtd.doesExist(omd):
            # ipts = GetIPTS(RunNumber=r, Instrument='CORELLI')
            # filename = '{}/nexus/CORELLI_{}.nxs.h5'.format(ipts,r)
            filename = '/SNS/CORELLI/IPTS-{}/nexus/CORELLI_{}.nxs.h5'.format(ipts,r)
            LoadEventNexus(Filename=filename, OutputWorkspace=ows)
            
            ApplyCalibration(Workspace=ows, CalibrationTable='tube_table')
            MaskDetectors(Workspace=ows, MaskedWorkspace='sa')
            LoadParameterFile(Workspace=ows, Filename=detector_calibration)
            
            proton_charge = sum(mtd[ows].getRun().getLogData('proton_charge').value)/1e12
            print('The current proton charge : {}'.format(proton_charge))
            
            #NormaliseByCurrent(ows, OutputWorkspace=ows)

            SetGoniometer(ows, Axis0='{},0,1,0,1'.format(gon_axis)) 
                            
            ConvertUnits(InputWorkspace=ows, OutputWorkspace=ows, EMode='Elastic', Target='Momentum')
            CropWorkspaceForMDNorm(InputWorkspace=ows, XMin=2.5, XMax=10, OutputWorkspace=ows)
            
            #ConvertUnits(InputWorkspace=ows, OutputWorkspace=ows, EMode='Elastic', Target='dSpacing')
                    
            ConvertToMD(InputWorkspace=ows, 
                        OutputWorkspace=omd, 
                        QDimensions="Q3D",
                        dEAnalysisMode="Elastic",
                        Q3DFrames="Q_sample",
                        LorentzCorrection=0,
                        MinValues="-20,-20,-20",
                        MaxValues="20,20,20",
                        Uproj='1,0,0',
                        Vproj='0,1,0',
                        Wproj='0,0,1', 
                        SplitInto=2, 
                        SplitThreshold=50, 
                        MaxRecursionDepth=13, 
                        MinRecursionDepth=7)
                                                
            LoadIsawUB(InputWorkspace=omd, Filename=ub_file)

        if not mtd.doesExist(opk):               
            PredictPeaks(InputWorkspace=omd, 
                         WavelengthMin=min_wavelength, 
                         WavelengthMax=max_wavelength, 
                         MinDSpacing=min_d_spacing, 
                         MaxDSpacing=max_d_spacing, 
                         OutputType='Peak',
                         ReflectionCondition=reflection_condition, 
                         OutputWorkspace=opk)
                         
            CentroidPeaksMD(InputWorkspace=omd,
                            PeakRadius=centroid_radius, 
                            PeaksWorkspace=opk, 
                            OutputWorkspace=opk)
                            
            CentroidPeaksMD(InputWorkspace=omd,
                            PeakRadius=centroid_radius, 
                            PeaksWorkspace=opk, 
                            OutputWorkspace=opk)

            IntegratePeaksMD(InputWorkspace=omd, 
                             PeakRadius=centroid_radius,
                             BackgroundInnerRadius=centroid_radius+0.01,
                             BackgroundOuterRadius=centroid_radius+0.02,
                             PeaksWorkspace=opk,
                             OutputWorkspace=opk)

        # delete ows save memory
        if mtd.doesExist(ows):
            DeleteWorkspace(ows)        
            # md = GroupWorkspaces(merge_md)
            # pk = GroupWorkspaces(merge_pk)
    
def pre_merging(ipts, exp, runs, ub_file, counts_file, reflection_condition):
    
    # peak centroid radius ---------------------------------------------------------
    centroid_radius = 0.125
    
    min_d_spacing = 0.7
    max_d_spacing= 20
    
    if not mtd.doesExist('van'):
        LoadMD(Filename=counts_file, OutputWorkspace='van')
        
    merge_md = []
    merge_pk = []

    for r in runs:
        print('Processing experiment : {}, scan : {}'.format(exp,r))
        ows = 'HB3A_'+str(exp)+'_'+str(r)
        omd = ows+'_md'
        opk = ows+'_pks'
        merge_md.append(omd)
        merge_pk.append(opk)
        
        if not mtd.doesExist(omd):
            filename = '/HFIR/HB3A/IPTS-{}/shared/autoreduce/HB3A_exp{:04}_scan{:04}.nxs'.format(ipts,exp,r)
            LoadMD(Filename=filename, OutputWorkspace=ows)
            
            wavelength = float(mtd[ows].getExperimentInfo(0).run().getProperty('wavelength').value)
            
            SetGoniometer(Workspace=ows, 
                          Axis0='omega,0,1,0,-1',
                          Axis1='chi,0,0,1,-1',
                          Axis2='phi,0,1,0,-1',
                          Average=False)
            
            ConvertHFIRSCDtoMDE(InputWorkspace=ows, 
                                Wavelength=wavelength, 
                                MinValues='-10,-10,-10',
                                MaxValues='10,10,10', 
                                OutputWorkspace=omd)   

        if not mtd.doesExist(ows+'_van'):

            wavelength = float(mtd[ows].getExperimentInfo(0).run().getProperty('wavelength').value)
            
            d = mtd[ows].getSignalArray()
            v = mtd['van'].getSignalArray().copy()
            
            mtd[ows].setSignalArray(v.repeat(d.shape[2]).reshape(*d.shape))
            
            ConvertHFIRSCDtoMDE(InputWorkspace=ows, 
                                Wavelength=wavelength, 
                                MinValues='-10,-10,-10',
                                MaxValues='10,10,10', 
                                OutputWorkspace=ows+'_van') 
                                
            if ub_file is None:
                UB = mtd[ows].getExperimentInfo(0).run().getProperty('ubmatrix').value
                UB = [float(ub) for ub in UB.split(' ')]
                UB = np.array(UB).reshape(3,3)
                
                SetUB(omd, UB=UB)
            else:
                LoadIsawUB(InputWorkspace=omd, Filename=ub_file)

        if not mtd.doesExist(opk):
        
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
                         
            CentroidPeaksMD(InputWorkspace=omd,
                            PeakRadius=centroid_radius, 
                            PeaksWorkspace=opk, 
                            OutputWorkspace=opk)
                            
            CentroidPeaksMD(InputWorkspace=omd,
                            PeakRadius=centroid_radius, 
                            PeaksWorkspace=opk, 
                            OutputWorkspace=opk)

            IntegratePeaksMD(InputWorkspace=omd, 
                             PeakRadius=centroid_radius,
                             BackgroundInnerRadius=centroid_radius+0.01,
                             BackgroundOuterRadius=centroid_radius+0.02,
                             PeaksWorkspace=opk,
                             OutputWorkspace=opk)

        # delete ows save memory
        if mtd.doesExist(ows):
            DeleteWorkspace(ows)        
            # md = GroupWorkspaces(merge_md)
            # pk = GroupWorkspaces(merge_pk)
            
# class GaussianFit3D:
#     
#     def __init__(self):
#         
#         pass
#             
#     def Gaussian3D(self, Q0, Q1, Q2, A, B, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01):
# 
#         sigma = np.array([[sig0**2,rho01*sig0*sig1,rho02*sig0*sig2],
#                           [rho01*sig0*sig1,sig1**2,rho12*sig1*sig2],
#                           [rho02*sig0*sig2,rho12*sig1*sig2,sig2**2]])
#                           
#         inv_sig = np.linalg.inv(sigma)
#         
#         x0, x1, x2 = Q0-mu0, Q1-mu1, Q2-mu2
#         
#         return A*np.exp(-0.5*(inv_sig[0,0]*x0**2+inv_sig[1,1]*x1**2+inv_sig[2,2]*x2**2\
#                           +2*(inv_sig[1,2]*x1*x2+inv_sig[0,2]*x0*x2+inv_sig[0,1]*x0*x1))+B
# 
#     def residual(self, params, Q0, Q1, Q2, y, e):
#         
#         A = params['A']
#         B = params['B']
#         
#         mu0 = params['mu0']
#         mu1 = params['mu1']
#         mu2 = params['mu2']
#         
#         sig0 = params['sig0']
#         sig1 = params['sig1']
#         sig2 = params['sig2']
#         
#         rho12 = params['rho12']
#         rho02 = params['rho02']
#         rho01 = params['rho01']
# 
#         yfit = Gaussian3D(Q0, Q1, Q2, A, B, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01)
#         
#         return (y-yfit)/e
        
#     def fit(self):
#         
#         print(y.max())
#     
#         e /= y.max()
#         y /= y.max()
#         
#         y *= 4/np.exp(2)
#         e *= 4/np.exp(2)
#         
#         x = x[y > 0.005]
#         e = e[y > 0.005]
#         y = y[y > 0.005]
#     
#         params = Parameters()
#         params.add('a', value=50.) #2/x[np.argmax(y)]
#         #params.add('b', value=b)
#         #params.add('r', value=r)
#         params.add('s', value=1.)
#         params.add('qmax', value=Qr)
# 
#         out = Minimizer(self.residual, params, fcn_args=(x, y, e))
#         result = out.minimize(method='leastsq') 
#         
#         a = result.params['a'].value
#         #b = result.params['b'].value
#         #r = result.params['r'].value
#         s = result.params['s'].value
#         qmax = result.params['qmax'].value
# 
#         #results = [a, b, r, s, t0]
#         results = [a, s, qmax]
#         
#         popt.append(results)
#         chisq.append(result.redchi)
#         mt.append(Qr)
#         
#         path_length.append(L1+L2)
#         scattering_angle.append(two_theta)
#         mom.append(2*np.pi/lamda)
# 
#         print(key)
#         report_fit(result)
#             
#         X = np.linspace(x.min(), x.max(), 500)
#         Y = IC(X, *popt[-1])
# 
#         plt.figure()
#         plt.errorbar(x, y, yerr=1.96*e, fmt='-o')
#         plt.plot(X, Y)
#         #plt.plot(popt[-1][2]-0.76124/popt[-1][0], popt[-1][1]*2/np.exp(2), 's')
#         #plt.plot(popt[-1][2]-4.15592/popt[-1][0], popt[-1][1]*2/np.exp(2), 's')
#         #plt.plot(popt[-1][2]+0.421416/popt[-1][0], popt[-1][1]*2/np.exp(2), 's')
#         plt.title(key+' {:.2e} [eV]'.format(energy[i]))
#         plt.xlabel(r'$Q$ [$\AA^{-1}$]')