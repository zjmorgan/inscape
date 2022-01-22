from mantid.simpleapi import CreatePeaksWorkspace, SortPeaksWorkspace
from mantid.simpleapi import SetUB, SaveNexus
from mantid.simpleapi import mtd

from mantid.kernel import V3D
from mantid.geometry import Goniometer

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import pprint
import pickle
#pickle.settings['recurse'] = True

def _pprint_dict(self, object, stream, indent, allowance, context, level):
    write = stream.write
    write('{')
    if self._indent_per_level > 1:
        write((self._indent_per_level-1) * ' ')
    length = len(object)
    if length:
        self._format_dict_items(object.items(), stream, indent,
                                allowance+1, context, level)
    write('}')
pprint.PrettyPrinter._dispatch[dict.__repr__] = _pprint_dict

from matplotlib.backends.backend_pdf import PdfPages
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

def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5*np.max(np.abs(limits[:,1]-limits[:,0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x-radius, x+radius])
    ax.set_ylim3d([y-radius, y+radius])
    ax.set_zlim3d([z-radius, z+radius])
   
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
        
        #self.ax_uv.get_xaxis().set_visible(False)
        
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
        
        h, k, l, m, n, p = key
      
        if m**2+n**2+p**2 > 0:
            
            sat = ''
            
            if m > 0: sat += '+k\u2081'
            if n > 0: sat += '+k\u2082'
            if p > 0: sat += '+k\u2083'
            
            if m < 0: sat += '-k\u2081'
            if n < 0: sat += '-k\u2082'
            if p < 0: sat += '-k\u2083'
            
            self.ax_Q.set_title('({} {} {}){}'.format(h,k,l,sat))
            
        else:
            
            self.ax_Q.set_title('({} {} {})'.format(h,k,l))
        
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

    def get_rows(self):
        
        return np.array(self.__row)
        
    def set_rows(self, rows):
        
        self.__row = list(rows)
        
    def get_cols(self):
        
        return np.array(self.__col)

    def set_cols(self, cols):
        
        self.__col = list(cols)

    def get_run_numbers(self):
        
        return np.array(self.__run_num)
        
    def set_run_numbers(self, runs):
        
        self.__run_num = list(runs)
        
    def get_bank_numbers(self):
        
        return np.array(self.__bank_num)
        
    def set_bank_numbers(self, banks):
        
        self.__bank_num = list(banks)
        
    def get_peak_indices(self):
        
        return np.array(self.__peak_ind)
        
    def set_peak_indices(self, indices):
        
        self.__peak_ind = list(indices)
        
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

    def get_wavelengths(self):
        
        return np.array(self.__wl)
        
    def set_wavelengths(self, wl):
        
        self.__wl = list(wl)
        
    def get_scattering_angles(self):
        
        return np.array(self.__two_theta)
        
    def set_scattering_angles(self, two_theta):
        
        self.__two_theta = list(two_theta)
        
    def get_azimuthal_angles(self):
    
        return np.array(self.__az_phi)
        
    def set_azimuthal_angles(self, az_phi):
    
        self.__az_phi = list(az_phi)
        
    def get_phi_angles(self):
        
        return np.array(self.__phi)
        
    def set_phi_angles(self, phi):
        
        self.__phi = list(phi)
        
    def get_chi_angles(self):
        
        return np.array(self.__chi)
        
    def set_chi_angles(self, chi):
        
        self.__chi = list(chi)
        
    def get_omega_angles(self):
        
        return np.array(self.__omega)
        
    def set_omega_angles(self, omega):
        
        self.__omega = list(omega)
        
    def get_estimated_intensities(self):
        
        return np.array(self.__est_int)
        
    def set_estimated_intensities(self, intens):
        
        self.__est_int = list(intens)

    def get_estimated_intensity_errors(self):
        
        return np.array(self.__est_int_err)
        
    def set_estimated_intensity_errors(self, sig):
        
        self.__est_int_err = list(sig)
        
    def __round(self, a, d):
        
        return np.round(a, d).tolist()
        
    def dictionary(self):
        
        d = { 'PeakNumber': self.__peak_num,
              'RunNumber': self.__run_num,
              'BankNumber': self.__bank_num,
              'PeakIndex': self.__peak_ind,
              'Row': self.__row,
              'Col': self.__col,
              'MergedIntensity': self.__round(self.__merge_intensity(),2),
              'MergedSigma': self.__round(self.__merge_intensity_error(),2),
              'MergedVolumeRatio': self.__round(self.__merge_pk_bkg_ratio(),2),
              'Ellispoid': self.__round(self.__A[np.triu_indices(3)],2),
              'BinSize': self.__round(self.__bin_size.tolist(),3),
              'Q': self.__round(self.__Q,3).tolist(),
              'PeakQFit': self.__round(self.__peak_fit,2),
              'PeakBackgroundRatio': self.__round(self.__peak_bkg_ratio,2),
              'PeakScore2D': self.__round(self.__peak_score,2),
              'Intensity': self.__round(self.__intensity(),2),
              'IntensitySigma': self.__round(self.__intensity_error(),2),
              'VolumeRatio': self.__round(self.__pk_bkg_ratio(),2).tolist(),
              'NormalizationScale': self.__round(self.__norm_scale,2),
              'Wavelength': self.__round(self.__wl,2).tolist(),
              'ScatteringAngle': self.__round(np.rad2deg(self.__two_theta),2),
              'AzimuthalAngle': self.__round(np.rad2deg(self.__az_phi),2),
              'GoniometerPhiAngle': self.__round(self.__phi,2),
              'GoniometerChiAngle': self.__round(self.__chi,2),
              'GoniometerOmegaAngle': self.__round(self.__omega,2),
              'EstimatedIntensity': self.__round(self.__est_int,2),
              'EstimatedSigma': self.__round(self.__est_int_err,2) }
        
        return d

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
        
    def add_partial_integration(self, Q, A, peak_fit, peak_bkg_ratio, peak_score):
        
        self.__Q = Q
        self.__A = A
                        
        self.__peak_fit = peak_fit
        self.__peak_bkg_ratio = peak_bkg_ratio
        self.__peak_score = peak_score
        
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
    
    def __call_peak(self, h, k, l, m=0, n=0, p=0):
        
        key = (h,k,l,m,n,p)
        
        d_spacing = self.get_d(h,k,l,m,n,p)
        
        if d_spacing is None:
            
            print('Peak does not exist')
        
        else:
            
            peak_key = (h,k,l,m,n,p) if m**2+n**2+p**2 > 0 else (h,k,l)
            
            print('{} {:2.4f} (\u212B)'.format(peak_key, d_spacing))
            
            peaks = self.peak_dict.get(key)
            
            for peak in peaks:
                pprint.pprint(peak.dictionary())

    __call__ = __call_peak
    
    def get_d(self, h, k, l, m=0, n=0, p=0):
        
        ol = self.iws.sample().getOrientedLattice()
        
        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)
        
        dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)

        d_spacing = ol.d(V3D(h+dh,k+dk,l+dl))
    
        return d_spacing
    
    def set_scale_constant(self, constant):
        
        self.scale_constant = constant
        
        for key in self.peak_dict.keys():
            
            peaks = self.peak_dict.get(key)
            
            for peak in peaks:
                
                peak.update_scale_constant(constant)
     
    def set_bank_constant(self, bank_scale):
        
        for key in self.peak_dict.keys():
            
            peaks = self.peak_dict.get(key)
            
            for peak in peaks:
                
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
        
    def set_satellite_info(self, mod_vector_1, mod_vector_2, mod_vector_3, max_order):
        
        self.__update_satellite_info(self.pws, mod_vector_1, mod_vector_2, mod_vector_3, max_order)
        self.__update_satellite_info(self.iws, mod_vector_1, mod_vector_2, mod_vector_3, max_order)
        self.__update_satellite_info(self.cws, mod_vector_1, mod_vector_2, mod_vector_3, max_order)
        
    def __update_satellite_info(self, pws, mod_vector_1, mod_vector_2, mod_vector_3, max_order):
        
       ol = pws.sample().getOrientedLattice()
        
       ol.setMaxOrder(max_order)
       
       ol.setModVec1(V3D(*mod_vector_1))
       ol.setModVec2(V3D(*mod_vector_2))
       ol.setModVec3(V3D(*mod_vector_3))
       
       UB = ol.getUB()
       
       mod_HKL = np.column_stack((mod_vector_1,mod_vector_2,mod_vector_3))
       mod_UB = np.dot(UB, mod_HKL)
       
       ol.setModUB(mod_UB)

    def __reset_peaks(self):

        for key in self.peak_dict.keys():

            peaks = self.peak_dict.get(key)
            
            h, k, l, m, n, p = key
            
            for peak in peaks:

                peak_num = peak.get_peak_number()
                intens = peak.get_merged_intensity()
                sig_intens = peak.get_merged_intensity_error()

                R = peak.get_goniometers()[0]

                pk = self.pws.createPeakHKL(V3D(h,k,l))
                pk.setIntMNP(V3D(m,n,p))
                pk.setIntensity(intens)
                pk.setSigmaIntensity(sig_intens)
                pk.setPeakNumber(peak_num)
                pk.setGoniometerMatrix(R)
                self.pws.addPeak(pk)
            
    def add_peaks(self, ws):
        
        if not mtd.doesExist('pws'):
            print('pws does not exist')
        
        if mtd.doesExist(ws):
            
            pws = mtd[ws]

            for pn in range(pws.getNumberPeaks()):
                
                bank = pws.row(pn)['BankName']
                
                peak = pws.getPeak(pn)
                
                intens = peak.getIntensity()
                sig_intens = peak.getSigmaIntensity()
                
                if bank != 'None' and intens > 0 and sig_intens > 0:
                            
                    h, k, l = peak.getIntHKL()
                    m, n, p = peak.getIntMNP()
                    
                    h, k, l, m, n, p = int(h), int(k), int(l), int(m), int(n), int(p)
                    
                    key = (h,k,l,m,n,p)
                    
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

                    #if np.isclose(chi,0) and np.isclose(omega,0):
                    #    phi, omega = 0, np.rad2deg(np.arctan2(R[0,2],R[0,0]))

                    if self.peak_dict.get(key) is None:

                        peak_num = self.pws.getNumberPeaks()+1
                        
                        new_peak = PeakInformation(self.scale_constant)
                        new_peak.set_peak_number(peak_num)

                        self.peak_dict[key] = [new_peak]

                        pk = self.pws.createPeakHKL(V3D(h,k,l))
                        pk.setIntMNP(V3D(m,n,p))
                        pk.setPeakNumber(peak_num)
                        pk.setGoniometerMatrix(R)

                        self.pws.addPeak(pk)
                        
                    self.peak_dict[key][0].add_information(run, bank, ind, row, col, wl, two_theta, az_phi, 
                                                           phi, chi, omega, intens, sig_intens)
        else:
            print('{} does not exist'.format(ws))
            
    def __dbscan_1d(self, array, eps):

        clusters = []

        index = np.argsort(array)

        i = index[0]
        curr_cluster = [i]
        for j in index[1:]:
            diff = array[j]-array[i]
            if min([diff,360-diff]) <= eps:
                curr_cluster.append(j)
            else:
                clusters.append(curr_cluster)
                curr_cluster = [j]
            i = j
        clusters.append(curr_cluster)
        
        if len(clusters) > 1:
            i, j = index[0], index[-1]
            diff = array[j]-array[i]
            if min([diff,360-diff]) <= eps:
                clusters[0] += clusters.pop(-1)

        return clusters

    def split_peaks(self, eps=5):

        for key in self.peak_dict.keys():

            peaks = self.peak_dict.get(key)

            h, k, l, m, n, p = key

            split = []

            for peak in peaks:

                peak_num = peak.get_peak_number()
                rows = peak.get_rows()
                cols = peak.get_cols()
                runs = peak.get_run_numbers()
                banks = peak.get_bank_numbers()
                indices = peak.get_peak_indices()
                wl = peak.get_wavelengths()
                two_theta = peak.get_scattering_angles()
                az = peak.get_azimuthal_angles()
                phi = peak.get_phi_angles()
                chi = peak.get_chi_angles()
                omega = peak.get_omega_angles()
                intens = peak.get_estimated_intensities()
                sig_intens = peak.get_estimated_intensity_errors()
                R = peak.get_goniometers()[0]

                clusters = self.__dbscan_1d(omega, eps)
                
                if len(clusters) > 1:
                    
                    print(clusters, len(clusters))
                    self.pws.removePeak(peak_num)

                    for cluster in clusters:
                        
                        cluster = np.array(cluster)

                        peak_num = self.pws.getNumberPeaks()+1

                        new_peak = PeakInformation(self.scale_constant)
                        new_peak.set_peak_number(peak_num)

                        pk = self.pws.createPeakHKL(V3D(h,k,l))
                        pk.setIntMNP(V3D(m,n,p))
                        pk.setPeakNumber(peak_num)
                        pk.setGoniometerMatrix(R)

                        self.pws.addPeak(pk)
                        
                        new_peak.set_peak_number(peak_num)
                        
                        new_peak.set_rows(rows[cluster])
                        new_peak.set_cols(cols[cluster])
                        new_peak.set_run_numbers(runs[cluster])
                        new_peak.set_bank_numbers(banks[cluster])
                        new_peak.set_peak_indices(indices[cluster])
                        new_peak.set_wavelengths(wl[cluster])
                        new_peak.set_scattering_angles(two_theta[cluster])
                        new_peak.set_azimuthal_angles(az[cluster])
                        new_peak.set_phi_angles(phi[cluster])        
                        new_peak.set_chi_angles(chi[cluster])
                        new_peak.set_omega_angles(omega[cluster])
                        new_peak.set_estimated_intensities(intens[cluster])
                        new_peak.set_estimated_intensity_errors(sig_intens[cluster])

                        split.append(new_peak)
                
                    self.peak_dict[key] = split
           
    def to_be_integrated(self):

        SortPeaksWorkspace(self.pws, ColumnNameToSortBy='DSpacing', SortAscending=False, OutputWorkspace=self.pws)  

        peak_dict = { }

        for pn in range(self.pws.getNumberPeaks()):

            pk = self.pws.getPeak(pn)

            h, k, l = pk.getIntHKL()
            m, n, p = pk.getIntMNP()

            h, k, l, m, n, p = int(h), int(k), int(l), int(m), int(n), int(p)

            key = (h,k,l,m,n,p)

            peaks = self.peak_dict.get(key)
            
            redudancies = []
            
            for peak in peaks:
                
                redudancy = peak.get_run_numbers().tolist(), peak.get_peak_indices().tolist()
                redudancies.append(redudancy)

            peak_dict[key] = redudancies

        return peak_dict
        
    def integrated_result(self, key, Q, A, peak_fit, peak_bkg_ratio, peak_score, data, index=0):

        peaks = self.peak_dict[key]

        peak = peaks[index]
        peak.add_integration(Q, A, peak_fit, peak_bkg_ratio, peak_score, data)

        h, k, l, m, n, p = key
        Qx, Qy, Qz = Q

        peak_num = self.iws.getNumberPeaks()+1
        intens = peak.get_merged_intensity()
        sig_intens = peak.get_merged_intensity_error()
        
        R = peak.get_goniometers()[0]

        pk = self.iws.createPeakHKL(V3D(h,k,l))
        pk.setGoniometerMatrix(R)
        pk.setPeakNumber(peak_num)
        pk.setIntensity(intens)
        pk.setSigmaIntensity(sig_intens)
        self.iws.addPeak(pk)

        peak_num = self.cws.getNumberPeaks()+1

        pk = self.cws.createPeakHKL(V3D(h,k,l))
        pk.setGoniometerMatrix(R)
        pk.setQSampleFrame(V3D(Qx,Qy,Qz))
        pk.setPeakNumber(peak_num)
        pk.setIntensity(intens)
        pk.setSigmaIntensity(sig_intens)
        self.cws.addPeak(pk)
        
    def partial_result(self, key, Q, A, peak_fit, peak_bkg_ratio, peak_score, index=0):

        peaks = self.peak_dict[key]

        peak = peaks[index]
        peak.add_partial_integration(Q, A, peak_fit, peak_bkg_ratio, peak_score)
        
    def calibrated_result(self, key, run_num, Q, index=0):

        peaks = self.peak_dict[key]

        peak = peaks[index]

        h, k, l, m, n, p = key
        Qx, Qy, Qz = Q

        peak_num = self.cws.getNumberPeaks()+1

        runs = peak.get_run_numbers()
        R = peak.get_goniometers()[runs.index(run_num)]

        pk = self.cws.createPeakHKL(V3D(h,k,l))
        pk.setGoniometerMatrix(R)
        pk.setQSampleFrame(V3D(Qx,Qy,Qz))
        pk.setPeakNumber(peak_num)
        pk.setRunNumber(run_num)
        self.cws.addPeak(pk)
            
    def save_hkl(self, filename, cross_terms=False):

        SortPeaksWorkspace(InputWorkspace=self.iws,
                           ColumnNameToSortBy='DSpacing',
                           SortAscending=False,
                           OutputWorkspace=self.iws)  

        ol = self.iws.sample().getOrientedLattice()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

        # max_order = ol.getMaxOrder()

        mod_vecs = []

        for i in range(3):
            x, y, z = ol.getModVec(i)
            if np.linalg.norm([x,y,z]) > 0:
                mod_vecs.append([x,y,z])
        n_mod = len(mod_vecs)

        satellite = True if n_mod > 0 else False

        if satellite:
            hkl_format = '{:4.0f}{:4.0f}{:4.0f}'\
                       + '{:4.0f}'*n_mod\
                       + '{:8.2f}{:8.2f}{:8.4f}\n'
        else:
            hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:8.4f}\n'

        with open(filename, 'w') as f:

            if satellite:
                f.write('# Structural propagation vectors used\n')
                f.write('           {}\n'.format(n_mod))
                for i, mod_vec in enumerate(mod_vecs):
                    x, y, z = mod_vec
                    f.write('       {}{: >13.6f}{: >13.6f}{: >13.6f}\n'.format(i+1,x,y,z))

            for pn in range(self.iws.getNumberPeaks()):

                pk = self.iws.getPeak(pn)
                intens, sig_intens = pk.getIntensity(), pk.getSigmaIntensity()

                if (intens > 0 and sig_intens > 0 and intens/sig_intens > 3):

                    h, k, l = pk.getIntHKL()
                    m, n, p = pk.getIntMNP()

                    h, k, l, m, n, p = int(h), int(k), int(l), int(m), int(n), int(p)

                    dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)

                    d_spacing = ol.d(V3D(h+dh,k+dk,l+dl))

                    items = [h, k, l]
                    if satellite:
                        mnp = []
                        if n_mod > 0: mnp.append(m)
                        if n_mod > 1: mnp.append(n)
                        if n_mod > 2: mnp.append(p)
                        items.extend(mnp)
                    items.extend([intens, sig_intens, d_spacing])

                    f.write(hkl_format.format(*items))

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

        peak_dict = { }

        for key in self.peak_dict.keys():

            # legacy format
            if len(key) == 3:
                h, k, l = key
                m, n, p = 0, 0, 0
            else:
                h, k, l, m, n, p = key

            # legacy format  
            if type(self.peak_dict[key]) is not list:
                peak_dict[(h,k,l,m,n,p)] = [self.peak_dict[key]]
            else:
                peak_dict[(h,k,l,m,n,p)] = self.peak_dict[key]

        self.peak_dict = peak_dict

        self.__reset_peaks()

        for key in self.peak_dict.keys():

            peaks = self.peak_dict.get(key)

            for peak in peaks:

                peak_num = peak.get_peak_number()
                intens = peak.get_merged_intensity()
                sig_intens = peak.get_merged_intensity_error()

                h, k, l, m, n, p = key
                Qx, Qy, Qz = peak.get_Q()

                peak_num = self.iws.getNumberPeaks()+1

                pk = self.iws.createPeakHKL(V3D(h,k,l))
                pk.setIntMNP(V3D(m,n,p))
                pk.setPeakNumber(peak_num)
                pk.setIntensity(intens)
                pk.setSigmaIntensity(sig_intens)
                self.iws.addPeak(pk)

                pk = self.cws.createPeakHKL(V3D(h,k,l))
                pk.setIntMNP(V3D(m,n,p))
                pk.setQSampleFrame(V3D(Qx,Qy,Qz))
                pk.setPeakNumber(peak_num)
                pk.setIntensity(intens)
                pk.setSigmaIntensity(sig_intens)
                self.cws.addPeak(pk)

        SortPeaksWorkspace(self.pws, ColumnNameToSortBy='DSpacing', SortAscending=False, OutputWorkspace=self.pws)
        SortPeaksWorkspace(self.iws, ColumnNameToSortBy='DSpacing', SortAscending=False, OutputWorkspace=self.iws)
    
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