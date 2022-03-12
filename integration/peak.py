from mantid.simpleapi import CreateSingleValuedWorkspace, CreatePeaksWorkspace
from mantid.simpleapi import CloneWorkspace, DeleteWorkspace
from mantid.simpleapi import SortPeaksWorkspace, FilterPeaks
from mantid.simpleapi import SetUB, SetSampleMaterial, SaveNexus
from mantid.simpleapi import mtd

from mantid.kernel import V3D
from mantid.geometry import Goniometer

g = Goniometer()

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import scipy.interpolate

import os
import pprint
import dill as pickle

from lmfit import Parameters, Minimizer, report_fit

#pickle.settings['recurse'] = True

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'PeakInformation':
            from peak import PeakInformation
            return PeakInformation
        return super().find_class(module, name)

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

        self.fig = plt.figure(num='peak-envelope', figsize=(18,6))
        gs = gridspec.GridSpec(1, 3, figure=self.fig, wspace=0.333, width_ratios=[0.2,0.4,0.4])

        gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.25)
        gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1], width_ratios=[0.8,0.2], height_ratios=[0.2,0.8])
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[2], width_ratios=[0.333,0.3333,0.333], height_ratios=[0.5,0.5], wspace=0.667, hspace=0.125)

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
        self.ax_Qv = self.fig.add_subplot(gs2[0,1])
        self.ax_uv = self.fig.add_subplot(gs2[0,2])
        
        self.ax_Qu_fit = self.fig.add_subplot(gs2[1,0])
        self.ax_Qv_fit = self.fig.add_subplot(gs2[1,1])
        self.ax_uv_fit = self.fig.add_subplot(gs2[1,2])

        self.ax_Qu.minorticks_on()
        self.ax_Qv.minorticks_on()
        self.ax_uv.minorticks_on()
        
        self.ax_Qu_fit.minorticks_on()
        self.ax_Qv_fit.minorticks_on()
        self.ax_uv_fit.minorticks_on()
        
        self.ax_pu.get_xaxis().set_visible(False)
        self.ax_pv.get_yaxis().set_visible(False)

        self.ax_pe.set_aspect('equal')
        self.ax_p_scat.set_aspect('equal')

        self.ax_pe.axis('off')

        self.ax_p_scat.set_xlabel('Q\u2081 (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        self.ax_p_scat.set_ylabel('Q\u2082 (\u212B\u207B\u00B9)') # [$\AA^{-1}$]

        self.ax_p_scat.ticklabel_format(axis='both', style='sci', scilimits=(0,0))

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
        
        self.im_Qu_fit = self.ax_Qu_fit.imshow([[0,1],[0,1]], interpolation='nearest',
                                               origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.im_Qv_fit = self.ax_Qv_fit.imshow([[0,1],[0,1]], interpolation='nearest',
                                               origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.im_uv_fit = self.ax_uv_fit.imshow([[0,1],[0,1]], interpolation='nearest',
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

        self.peak_pu_fit = self.ax_Qu_fit.plot([0,1], [0,1], '-', color='C3', zorder=10, rasterized=True)
        self.inner_pu_fit = self.ax_Qu_fit.plot([0,1], [0,1], ':', color='C3', zorder=10, rasterized=True)
        self.outer_pu_fit = self.ax_Qu_fit.plot([0,1], [0,1], '--', color='C3', zorder=10, rasterized=True)

        self.peak_pv_fit = self.ax_Qv_fit.plot([0,1], [0,1], '-', color='C3', zorder=10, rasterized=True)
        self.inner_pv_fit = self.ax_Qv_fit.plot([0,1], [0,1], ':', color='C3', zorder=10, rasterized=True)
        self.outer_pv_fit = self.ax_Qv_fit.plot([0,1], [0,1], '--', color='C3', zorder=10, rasterized=True)

        self.peak_uv_fit = self.ax_uv_fit.plot([0,1], [0,1], '-', color='C3', zorder=10, rasterized=True)
        self.inner_uv_fit = self.ax_uv_fit.plot([0,1], [0,1], ':', color='C3', zorder=10, rasterized=True)
        self.outer_uv_fit = self.ax_uv_fit.plot([0,1], [0,1], '--', color='C3', zorder=10, rasterized=True)

        self.ax_Qu.set_ylabel('Q\u2081\u2032 (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        self.ax_Qv.set_ylabel('Q\u2082\u2032 (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        self.ax_uv.set_ylabel('Q (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        
        self.ax_Qv_fit.set_xlabel('Q (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        self.ax_Qu_fit.set_xlabel('Q (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        self.ax_uv_fit.set_xlabel('Q\u2081\u2032 (\u212B\u207B\u00B9)') # [$\AA^{-1}$]

        self.ax_Qu_fit.set_ylabel('Q\u2081\u2032 (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        self.ax_Qv_fit.set_ylabel('Q\u2082\u2032 (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        self.ax_uv_fit.set_ylabel('Q\u2082\u2032 (\u212B\u207B\u00B9)') # [$\AA^{-1}$]
        
        self.ax_Qu.get_xaxis().set_visible(False)
        self.ax_Qv.get_xaxis().set_visible(False)
        self.ax_uv.get_xaxis().set_visible(False)

        #self.ax_Qu_fit.get_xaxis().set_visible(False)
        #self.ax_uv_fit.get_yaxis().set_visible(False)
        
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

        self.peak_pu_fit[0].set_data([0,1],[0,1])
        self.peak_pv_fit[0].set_data([0,1],[0,1])
        self.peak_uv_fit[0].set_data([0,1],[0,1])

        self.inner_pu_fit[0].set_data([0,1],[0,1])
        self.inner_pv_fit[0].set_data([0,1],[0,1])
        self.inner_uv_fit[0].set_data([0,1],[0,1])

        self.outer_pu_fit[0].set_data([0,1],[0,1])
        self.outer_pv_fit[0].set_data([0,1],[0,1])
        self.outer_uv_fit[0].set_data([0,1],[0,1])

    def show_plots(self, show):

        self.__show_plots = show

    def create_pdf(self):

        self.pp.close()

    def plot_Q(self, key, x, y, y0, yerr, X, Y):

        h, k, l, m, n, p = key

        if m**2+n**2+p**2 > 0:

            sat = ''

            if m > 0: sat += '+q\u2081'
            if n > 0: sat += '+q\u2082'
            if p > 0: sat += '+q\u2083'

            if m < 0: sat += '-q\u2081'
            if n < 0: sat += '-q\u2082'
            if p < 0: sat += '-q\u2083'

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

    def plot_integration_fit(self, signal, u_extents, v_extents, Q_extents,
                             x_pk_Qu, y_pk_Qu, x_pk_Qv, y_pk_Qv, x_pk_uv, y_pk_uv,
                             x_in_Qu, y_in_Qu, x_in_Qv, y_in_Qv, x_in_uv, y_in_uv,
                             x_out_Qu, y_out_Qu, x_out_Qv, y_out_Qv, x_out_uv, y_out_uv):

        Qu = np.nansum(signal, axis=1)
        Qv = np.nansum(signal, axis=0)
        uv = np.nansum(signal, axis=2)

        self.im_Qu_fit.set_array(Qu)
        self.im_Qv_fit.set_array(Qv)
        self.im_uv_fit.set_array(uv)

        self.im_Qu_fit.autoscale()
        self.im_Qv_fit.autoscale()
        self.im_uv_fit.autoscale()

        Qu_extents = [Q_extents[0],Q_extents[2],u_extents[0],u_extents[2]]
        Qv_extents = [Q_extents[0],Q_extents[2],v_extents[0],v_extents[2]]
        uv_extents = [u_extents[0],u_extents[2],v_extents[0],v_extents[2]]

        self.im_Qu_fit.set_extent(Qu_extents)
        self.im_Qv_fit.set_extent(Qv_extents)
        self.im_uv_fit.set_extent(uv_extents)

        Qu_aspect = Q_extents[1]/u_extents[1]
        Qv_aspect = Q_extents[1]/v_extents[1]
        uv_aspect = u_extents[1]/v_extents[1]

        self.ax_Qu_fit.set_aspect(Qu_aspect)
        self.ax_Qv_fit.set_aspect(Qv_aspect)
        self.ax_uv_fit.set_aspect(uv_aspect)

        self.peak_pu_fit[0].set_data(x_pk_Qu, y_pk_Qu)
        self.peak_pv_fit[0].set_data(x_pk_Qv, y_pk_Qv)
        self.peak_uv_fit[0].set_data(x_pk_uv, y_pk_uv)

        self.inner_pu_fit[0].set_data(x_in_Qu, y_in_Qu)
        self.inner_pv_fit[0].set_data(x_in_Qv, y_in_Qv)
        self.inner_uv_fit[0].set_data(x_in_uv, y_in_uv)

        self.outer_pu_fit[0].set_data(x_out_Qu, y_out_Qu)
        self.outer_pv_fit[0].set_data(x_out_Qv, y_out_Qv)
        self.outer_uv_fit[0].set_data(x_out_uv, y_out_uv)

        if self.__show_plots: self.fig.show()

    def write_figure(self):
        
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

        self.__data_scale = np.array([])
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

    def set_Q(self, Q):

        self.__Q = Q

    def get_A(self):

        return self.__A

    def set_A(self, A):

        self.__A = A

    def set_peak_constant(self, scale_constant):

        self.__scale_constant = scale_constant
        
    def get_peak_constant(self):

        return self.__scale_constant
        
    def get_bin_size(self):
        
        return self.__bin_size

    def set_data_scale(self, corr_scale):

        self.__data_scale = np.array(corr_scale)

    def get_data_scale(self):

        if not hasattr(self, '_PeakInformation__data_scale'):
            self.__data_scale = np.ones_like(self.__norm_scale)

        return self.__data_scale

    def set_norm_scale(self, corr_scale):

        self.__norm_scale = np.array(corr_scale)

    def get_norm_scale(self):

        return self.__norm_scale

    def set_bank_scale(self, bank_scale):

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

    def get_merged_peak_volume_fraction(self):

        return self.__merge_pk_vol_fract()

    def get_merged_intensity(self):

        return self.__merge_intensity()

    def get_merged_intensity_error(self):

        return self.__merge_intensity_error()

    def get_peak_volume_fraction(self):

        return self.__pk_vol_fract()

    def get_intensity(self):

        return self.__intensity()

    def get_intensity_error(self):

        return self.__intensity_error()

    def get_goniometers(self):

        R = []

        for phi, chi, omega in zip(self.__phi,self.__chi,self.__omega):
            R.append(np.dot(self.__R_y(omega), np.dot(self.__R_z(chi), self.__R_y(phi))))

        return R

    def get_rotation_angle(self):

        varphi = []

        for phi, chi, omega in zip(self.__phi,self.__chi,self.__omega):
            R = np.dot(self.__R_y(omega), np.dot(self.__R_z(chi), self.__R_y(phi)))
            varphi.append(np.rad2deg(np.arccos((np.trace(R)-1)/2)))

        return varphi

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
              'MergedVolumeFraction': self.__round(self.__merge_pk_vol_fract(),2),
              'Ellispoid': self.__round(self.__A[np.triu_indices(3)],2),
              'BinSize': self.__round(self.__bin_size,3),
              'Q': self.__round(self.__Q,3),
              'PeakQFit': self.__round(self.__peak_fit,2),
              'PeakBackgroundRatio': self.__round(self.__peak_bkg_ratio,2),
              'PeakScore2D': self.__round(self.__peak_score,2),
              'Intensity': self.__round(self.__intensity(),2),
              'IntensitySigma': self.__round(self.__intensity_error(),2),
              'VolumeRatio': self.__round(self.__pk_bkg_ratio(),2),
              'VolumeFraction': self.__round(self.__pk_vol_fract(),2),
              'NormalizationScale': self.__round(self.__norm_scale,2),
              'Wavelength': self.__round(self.__wl,2),
              'CorrectionFactor': self.__round(self.__data_scale,3),
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
        self.__data_scale = np.ones(len(pk_data))

    def add_partial_integration(self, Q, A, peak_fit, peak_bkg_ratio, peak_score):

        self.__Q = Q
        self.__A = A

        self.__peak_fit = peak_fit
        self.__peak_bkg_ratio = peak_bkg_ratio
        self.__peak_score = peak_score
        
    def __merge_pk_vol_fract(self):

        if not self.__is_peak_integrated() or len(self.__good_intensities()) == 0:

            return 0.0

        else:
            
            data = self.__get_merged_peak_data_arrays()
            norm = self.__get_merged_peak_norm_arrays()

            pk_data = np.sum(data, axis=0)
            pk_norm = np.sum(norm, axis=0)
            
            pk_vol_fract = np.sum(~(np.isnan(pk_data/pk_norm)))/len(data[0])

            return pk_vol_fract

    def __merge_pk_bkg_ratio(self):

        if not self.__is_peak_integrated() or len(self.__good_intensities()) == 0:

            return 0.0

        else:
            
            data = self.__get_merged_peak_data_arrays()
            norm = self.__get_merged_peak_norm_arrays()

            bkg_data = self.__get_merged_background_data_arrays()
            bkg_norm = self.__get_merged_background_norm_arrays()

            pk_vol = np.sum(~np.isnan([data,norm]).any(axis=0))
            bkg_vol = np.sum(~np.isnan([bkg_data,bkg_norm]).any(axis=0))

            return pk_vol/bkg_vol

    def __merge_intensity(self):

        if not self.__is_peak_integrated() or len(self.__good_intensities()) == 0:

            return 0.0

        else:

            data = self.__get_merged_peak_data_arrays()
            norm = self.__get_merged_peak_norm_arrays()

            bkg_data = self.__get_merged_background_data_arrays()
            bkg_norm = self.__get_merged_background_norm_arrays()

            scale_data = self.get_merged_data_scale()[:,np.newaxis]
            scale_norm = self.get_merged_norm_scale()[:,np.newaxis]

            volume_ratio = self.__merge_pk_bkg_ratio()

            constant = self.get_peak_constant()*np.prod(self.get_bin_size())

            data_norm = np.nansum(np.multiply(data, scale_data), axis=0)/np.nansum(np.multiply(norm, scale_norm), axis=0)
            bkg_data_norm = np.nansum(np.multiply(bkg_data, scale_data), axis=0)/np.nansum(np.multiply(bkg_norm, scale_norm), axis=0)

            data_norm[np.isinf(data_norm)] = np.nan
            bkg_data_norm[np.isinf(bkg_data_norm)] = np.nan

            Q1, Q2, Q3 = np.nanpercentile(bkg_data_norm, [25,50,75])
            IQR = Q3-Q1
            mask = (bkg_data_norm > Q3+1.5*IQR) | (bkg_data_norm < Q1-1.5*IQR)

            bkg_data_norm[mask] = Q2

            intens = np.nansum(data_norm)
            bkg_intens = np.nansum(bkg_data_norm)

            intensity = (intens-bkg_intens*volume_ratio)*constant

            return intensity

    def __merge_intensity_error(self):

        if not self.__is_peak_integrated() or len(self.__good_intensities()) == 0:

            return 0.0

        else:

            data = self.__get_merged_peak_data_arrays()
            norm = self.__get_merged_peak_norm_arrays()

            bkg_data = self.__get_merged_background_data_arrays()
            bkg_norm = self.__get_merged_background_norm_arrays()
            
            scale_data = self.get_merged_data_scale()[:,np.newaxis]
            scale_norm = self.get_merged_norm_scale()[:,np.newaxis]

            volume_ratio = self.__merge_pk_bkg_ratio()

            constant = self.get_peak_constant()*np.prod(self.get_bin_size())

            data_norm = np.nansum(np.multiply(data, scale_data), axis=0)/np.nansum(np.multiply(norm, scale_norm), axis=0)**2*(1+np.nansum(np.multiply(data, scale_data), axis=0)/np.nansum(np.multiply(norm, scale_norm), axis=0))
            bkg_data_norm = np.nansum(np.multiply(bkg_data, scale_data), axis=0)/np.nansum(np.multiply(bkg_norm, scale_norm), axis=0)**2*(1+np.nansum(np.multiply(bkg_data, scale_data), axis=0)/np.nansum(np.multiply(bkg_norm, scale_norm), axis=0))

            data_norm[np.isinf(data_norm)] = np.nan
            bkg_data_norm[np.isinf(bkg_data_norm)] = np.nan

            intens = np.nansum(data_norm)
            bkg_intens = np.nansum(bkg_data_norm)

            intensity = np.sqrt(intens+bkg_intens*volume_ratio**2)*constant

            return intensity

    def __pk_vol_fract(self):
        
        if not self.__is_peak_integrated():

            return np.array([])

        else:
            
            data = self.__get_peak_data_arrays()
            norm = self.__get_peak_norm_arrays()

            pk_vol_fract = np.sum(~(np.isnan(np.array(data)/np.array(norm))), axis=1)/len(data[0])
    
            return pk_vol_fract

    def __pk_bkg_ratio(self):

        if not self.__is_peak_integrated():

            return np.array([])

        else:
            
            data = self.__get_peak_data_arrays()
            norm = self.__get_peak_norm_arrays()

            bkg_data = self.__get_background_data_arrays()
            bkg_norm = self.__get_background_norm_arrays()

            pk_vol = np.sum(~np.isnan([data,norm]).any(axis=0),axis=1)
            bkg_vol = np.sum(~np.isnan([bkg_data,bkg_norm]).any(axis=0),axis=1)

            return pk_vol/bkg_vol
           
    def __intensity(self):

        if not self.__is_peak_integrated():

            return np.array([])

        else:

            data = self.__get_peak_data_arrays()
            norm = self.__get_peak_norm_arrays()

            bkg_data = self.__get_background_data_arrays()
            bkg_norm = self.__get_background_norm_arrays()

            scale_data = self.__data_scale[:,np.newaxis]
            scale_norm = self.__norm_scale[:,np.newaxis]

            volume_ratio = self.__pk_bkg_ratio()

            constant = self.get_peak_constant()*np.prod(self.get_bin_size())

            data_norm = np.multiply(data, scale_data)/np.multiply(norm, scale_norm)
            bkg_data_norm = np.multiply(bkg_data, scale_data)/np.multiply(bkg_norm, scale_norm)

            data_norm[np.isinf(data_norm)] = np.nan
            bkg_data_norm[np.isinf(bkg_data_norm)] = np.nan

            intens = np.nansum(data_norm, axis=1)
            bkg_intens = np.nansum(bkg_data_norm, axis=1)

            intensity = (intens-np.multiply(bkg_intens,volume_ratio))*constant

            return intensity

    def __intensity_error(self):

        if not self.__is_peak_integrated():

            return np.array([])

        else:

            data = self.__get_peak_data_arrays()
            norm = self.__get_peak_norm_arrays()

            bkg_data = self.__get_background_data_arrays()
            bkg_norm = self.__get_background_norm_arrays()

            scale_data = self.__data_scale[:,np.newaxis]
            scale_norm = self.__norm_scale[:,np.newaxis]
            
            volume_ratio = self.__pk_bkg_ratio()

            constant = self.get_peak_constant()*np.prod(self.get_bin_size())

            data_norm = np.multiply(data, scale_data)/np.multiply(norm, scale_norm)**2*(1+np.multiply(data, scale_data)/np.multiply(norm, scale_norm))
            bkg_data_norm = np.multiply(bkg_data, scale_data)/np.multiply(bkg_norm, scale_norm)**2*(1+np.multiply(bkg_data, scale_data)/np.multiply(bkg_norm, scale_norm))

            data_norm[np.isinf(data_norm)] = 0
            bkg_data_norm[np.isinf(bkg_data_norm)] = 0

            intens = np.nansum(data_norm, axis=1)
            bkg_intens = np.nansum(bkg_data_norm, axis=1)

            intensity = np.sqrt(intens+np.multiply(bkg_intens,volume_ratio**2))*constant

            return intensity
            
    def __is_peak_integrated(self):
        
        return not (self.__peak_num == 0 or self.__pk_norm is None)
        
    def __good_intensities(self, min_vol_fract=0.9):

        pk_vol_fract = np.array(self.__pk_vol_fract())
        
        indices = np.arange(len(pk_vol_fract)).tolist()
        
        return [ind for ind in indices if pk_vol_fract[ind] > min_vol_fract]
        
    def get_merged_data_scale(self):
        
        indices = self.__good_intensities()
        
        scale_data = self.get_data_scale()
        
        return np.array([scale_data[ind] for ind in indices])
        
    def get_merged_norm_scale(self):
        
        indices = self.__good_intensities()
        
        scale_norm = self.get_norm_scale()
        
        return np.array([scale_norm[ind] for ind in indices])
        
    def __get_merged_peak_data_arrays(self):
        
        indices = self.__good_intensities()
        
        pk_data = self.__get_peak_data_arrays()
        
        return [pk_data[ind] for ind in indices]
        
    def __get_merged_peak_norm_arrays(self):
        
        indices = self.__good_intensities()
        
        pk_norm = self.__get_peak_norm_arrays()
        
        return [pk_norm[ind] for ind in indices]
                
    def __get_merged_background_data_arrays(self):
        
        indices = self.__good_intensities()
        
        bkg_data = self.__get_background_data_arrays()
        
        return [bkg_data[ind] for ind in indices]
        
    def __get_merged_background_norm_arrays(self):
        
        indices = self.__good_intensities()
        
        bkg_norm = self.__get_background_norm_arrays()
        
        return [bkg_norm[ind] for ind in indices]
                    
    def __get_peak_data_arrays(self):
        
        return self.__pk_data
        
    def __get_peak_norm_arrays(self):
        
        return self.__pk_norm
        
    def __get_background_data_arrays(self):
        
        return self.__bkg_data
        
    def __get_background_norm_arrays(self):
        
        return self.__bkg_norm

class PeakDictionary:

    def __init__(self, a, b, c, alpha, beta, gamma):

        self.peak_dict = { }

        self.scale_constant = 1e+8
        
        self.mass = 0

        CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace='pws')
        CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace='iws')
        CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace='cws')

        CreateSingleValuedWorkspace(OutputWorkspace='nws')

        self.pws = mtd['pws']
        self.iws = mtd['iws']
        self.cws = mtd['cws']
        self.nws = mtd['nws']
        
        self.set_constants(a, b, c, alpha, beta, gamma)

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

                peak.set_peak_constant(constant)

    def set_bank_constant(self, bank_scale):

        for key in self.peak_dict.keys():

            peaks = self.peak_dict.get(key)

            for peak in peaks:

                peak.set_norm_scale(bank_scale)

    def set_constants(self, a, b, c, alpha, beta, gamma):

        SetUB(Workspace=self.pws, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        SetUB(Workspace=self.iws, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        SetUB(Workspace=self.cws, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

        self.__reset_peaks()

    def set_satellite_info(self, mod_vector_1, mod_vector_2, mod_vector_3, max_order):

        self.__set_satellite_info(self.pws, mod_vector_1, mod_vector_2, mod_vector_3, max_order)
        self.__set_satellite_info(self.iws, mod_vector_1, mod_vector_2, mod_vector_3, max_order)
        self.__set_satellite_info(self.cws, mod_vector_1, mod_vector_2, mod_vector_3, max_order)
        
    def set_material_info(self, chemical_formula, z_parameter, sample_mass):

        self.mass = sample_mass
        
        self.__set_material_info(self.pws, chemical_formula, z_parameter)
        self.__set_material_info(self.iws, chemical_formula, z_parameter)
        self.__set_material_info(self.cws, chemical_formula, z_parameter)

    def set_norm_info(self, chemical_formula, z_parameter, sample_mass, a=3.02, b=3.02, c=3.02, alpha=90, beta=90, gamma=90):

        SetUB(Workspace=self.nws, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

        self.norm_mass = sample_mass

        self.__set_material_info(self.nws, chemical_formula, z_parameter)

    def __set_material_info(self, pws, chemical_formula, z_parameter):

        sample_mass = self.mass

        volume = pws.sample().getOrientedLattice().volume()

        if chemical_formula is not None:
            SetSampleMaterial(InputWorkspace=pws,
                              ChemicalFormula=chemical_formula,
                              ZParameter=z_parameter,
                              UnitCellVolume=volume,
                              SampleMass=sample_mass)

    def __set_satellite_info(self, pws, mod_vector_1, mod_vector_2, mod_vector_3, max_order):

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
                pk_vol_fract = peak.get_merged_peak_volume_fraction()

                R = peak.get_goniometers()[0]

                pk = self.pws.createPeakHKL(V3D(h,k,l))
                pk.setIntMNP(V3D(m,n,p))
                pk.setIntensity(intens)
                pk.setSigmaIntensity(sig_intens)
                pk.setPeakNumber(peak_num)
                pk.setGoniometerMatrix(R)
                pk.setBinCount(pk_vol_fract)
                self.pws.addPeak(pk)

    def add_peaks(self, ws):

        if not mtd.doesExist('pws'):
            print('pws does not exist')

        if mtd.doesExist(ws):

            pws = mtd[ws]

            for pn in range(pws.getNumberPeaks()):

                bank = pws.row(pn)['BankName']
                row = int(pws.row(pn)['Row'])
                col = int(pws.row(pn)['Col'])
                    
                peak = pws.getPeak(pn)

                intens = peak.getIntensity()
                sig_intens = peak.getSigmaIntensity()

                if bank != 'None' and bank != '' and intens > 0 and sig_intens > 0 and intens/sig_intens > 1:

                    h, k, l = peak.getIntHKL()
                    m, n, p = peak.getIntMNP()

                    h, k, l, m, n, p = int(h), int(k), int(l), int(m), int(n), int(p)

                    key = (h,k,l,m,n,p)

                    run = peak.getRunNumber()
                    if run == 0 and ws.split('_')[-2].isnumeric():
                        _, exp, run, _ = ws.split('_')
                    
                    bank = 1 if bank == 'panel' else int(bank.strip('bank'))
                    ind = peak.getPeakNumber()
                    
                    Q = peak.getQSampleFrame()

                    wl = peak.getWavelength()
                    two_theta = peak.getScattering()
                    az_phi = peak.getAzimuthal()

                    R = peak.getGoniometerMatrix()

                    g.setR(R)
                    omega, chi, phi = g.getEulerAngles('YZY')

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

                    self.peak_dict[key][0].set_Q(Q)

        else:
            print('{} does not exist'.format(ws))

    def __dbscan_1d(self, array, eps):

        array = np.mod(array, 180)

        clusters = []

        index = np.argsort(array)

        i = index[0]
        curr_cluster = [i]
        for j in index[1:]:
            diff = array[j]-array[i]
            if min([diff,180-diff]) <= eps:
                curr_cluster.append(j)
            else:
                clusters.append(curr_cluster)
                curr_cluster = [j]
            i = j
        clusters.append(curr_cluster)

        if len(clusters) > 1:
            i, j = index[0], index[-1]
            diff = array[j]-array[i]
            if min([diff,180-diff]) <= eps:
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
                Q = peak.get_Q()

                varphi = peak.get_rotation_angle()

                clusters = self.__dbscan_1d(varphi, eps)

                if len(clusters) > 1:

                    for i, cluster in enumerate(clusters):

                        cluster = np.array(cluster)

                        new_peak = PeakInformation(self.scale_constant)

                        if i > 0:

                            peak_num = self.pws.getNumberPeaks()+1

                            pk = self.pws.createPeakHKL(V3D(h,k,l))
                            pk.setIntMNP(V3D(m,n,p))
                            pk.setPeakNumber(peak_num)
                            pk.setGoniometerMatrix(R)
                            pk.setQSampleFrame(Q)

                            self.pws.addPeak(pk)

                        new_peak.set_peak_number(peak_num)
                        new_peak.set_Q(Q)

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
        pk_vol_fract = peak.get_merged_peak_volume_fraction()

        run = peak.get_run_numbers().tolist()[0]
        R = peak.get_goniometers()[0]

        self.iws.run().getGoniometer().setR(R)
        self.cws.run().getGoniometer().setR(R)

        pk = self.iws.createPeakHKL(V3D(h,k,l))
        pk.setGoniometerMatrix(R)
        pk.setIntMNP(V3D(m,n,p))
        pk.setPeakNumber(peak_num)
        pk.setIntensity(intens)
        pk.setSigmaIntensity(sig_intens)
        pk.setBinCount(pk_vol_fract)
        self.iws.addPeak(pk)

        peak_num = self.cws.getNumberPeaks()+1

        pk = self.cws.createPeakQSample(V3D(Qx,Qy,Qz))
        pk.setGoniometerMatrix(R)
        pk.setHKL(h,k,l)
        pk.setIntHKL(V3D(h,k,l))
        pk.setIntMNP(V3D(m,n,p))
        pk.setPeakNumber(peak_num)
        pk.setIntensity(intens)
        pk.setSigmaIntensity(sig_intens)
        pk.setBinCount(pk_vol_fract)
        pk.setRunNumber(run)
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

        runs = peak.get_run_numbers().tolist()
        R = peak.get_goniometers()[runs.index(run_num)]

        self.cws.run().getGoniometer().setR(R)
        
        pk = self.cws.createPeakQSample(V3D(Qx,Qy,Qz))
        pk.setGoniometerMatrix(R)
        pk.setHKL(h,k,l)
        pk.setIntHKL(V3D(h,k,l))
        pk.setIntMNP(V3D(m,n,p))
        pk.setPeakNumber(peak_num)
        pk.setRunNumber(run_num)
        self.cws.addPeak(pk)
        
    def save_hkl(self, filename, min_signal_noise_ratio=3, min_pk_vol_fract=0.7, adaptive_scale=False, cross_terms=False):

        SortPeaksWorkspace(InputWorkspace=self.iws,
                           ColumnNameToSortBy='Intens',
                           SortAscending=False,
                           OutputWorkspace=self.iws)

        scale = 1
        if adaptive_scale:
            if self.iws.getNumberPeaks() > 0:
                I = self.iws.getPeak(0).getIntensity()
                scale = 9999.99/I

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
                intens, sig_intens, pk_vol_fract = pk.getIntensity()*scale, pk.getSigmaIntensity()*scale, pk.getBinCount()

                if (intens > 0 and sig_intens > 0 and intens/sig_intens > min_signal_noise_ratio and pk_vol_fract > min_pk_vol_fract):

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

    def save_calibration(self, filename, min_sig=5, min_sig_noise_ratio=3, min_vol_fract=0.9):
        
        CloneWorkspace(self.cws, OutputWorkspace='cal')

        FilterPeaks(InputWorkspace='cal',
                    FilterVariable='Signal/Noise',
                    FilterValue=min_sig_noise_ratio,
                    Operator='>',
                    OutputWorkspace='cal')

        FilterPeaks(InputWorkspace='cal',
                    FilterVariable='Intensity',
                    FilterValue=min_sig,
                    Operator='>',
                    OutputWorkspace='cal')

        cal = mtd['cal']
                
        n = cal.getNumberPeaks()
        for pn in range(n-1,-1,-1):
            pk = cal.getPeak(pn)
            vol_fract = pk.getBinCount()
            if vol_fract < min_vol_fract:
                cal.removePeak(pn)

        SaveNexus(InputWorkspace='cal', Filename=filename)

        DeleteWorkspace('cal')

    def save(self, filename):

        with open(filename, 'wb') as f:

            pickle.dump(self.peak_dict, f)

    def load(self, filename):

        self.peak_dict = self.load_dictionary(filename)

        self.__reset_peaks()

        self.__repopulate_workspaces()

    def load_dictionary(self, filename):
    
        ol = self.pws.sample().getOrientedLattice()

        a, b, c, alpha, beta, gamma = ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma()

        self.set_constants(a, b, c, alpha, beta, gamma)

        with open(filename, 'rb') as f:

            #self.peak_dict = pickle.load(f)
            self.peak_dict = CustomUnpickler(f).load()

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
                
        return peak_dict

    def __repopulate_workspaces(self):

        for key in self.peak_dict.keys():

            peaks = self.peak_dict.get(key)

            for peak in peaks:
                
                peak.set_peak_constant(self.scale_constant)

                peak_num = peak.get_peak_number()
                intens = peak.get_merged_intensity()
                sig_intens = peak.get_merged_intensity_error()
                pk_vol_fract = peak.get_merged_peak_volume_fraction()

                run = peak.get_run_numbers().tolist()[0]
                R = peak.get_goniometers()[0]

                h, k, l, m, n, p = key
                Qx, Qy, Qz = peak.get_Q()

                if intens > 0 and sig_intens > 0:

                    peak_num = self.iws.getNumberPeaks()+1

                    pk = self.iws.createPeakHKL(V3D(h,k,l))
                    pk.setIntMNP(V3D(m,n,p))
                    pk.setPeakNumber(peak_num)
                    pk.setIntensity(intens)
                    pk.setSigmaIntensity(sig_intens)
                    pk.setGoniometerMatrix(R)
                    pk.setBinCount(pk_vol_fract)
                    self.iws.addPeak(pk)

                    pk = self.cws.createPeakQSample(V3D(Qx,Qy,Qz))
                    pk.setGoniometerMatrix(R)
                    pk.setHKL(h,k,l)
                    pk.setIntHKL(V3D(h,k,l))
                    pk.setIntMNP(V3D(m,n,p))
                    pk.setPeakNumber(peak_num)
                    pk.setIntensity(intens)
                    pk.setSigmaIntensity(sig_intens)
                    pk.setBinCount(pk_vol_fract)
                    pk.setRunNumber(run)
                    self.cws.addPeak(pk)

        SortPeaksWorkspace(self.iws, ColumnNameToSortBy='DSpacing', SortAscending=False, OutputWorkspace=self.iws)
        SortPeaksWorkspace(self.cws, ColumnNameToSortBy='DSpacing', SortAscending=False, OutputWorkspace=self.cws)

    def __spherical_aborption(self):

        filename = os.path.join(os.path.dirname(__file__), 'absorption_sphere.csv')

        data = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=np.arange(1,92))

        muR = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=(0))
        theta = np.loadtxt(filename, delimiter=',', max_rows=1)

        return scipy.interpolate.interp2d(muR, 2*theta, data.T, kind='linear')

    def apply_spherical_correction(self):

        f = self.__spherical_aborption()

        mat = mtd['iws'].sample().getMaterial()

        sigma_a = mat.absorbXSection()
        sigma_s = mat.totalScatterXSection()

        m = self.mass # g
        M = mat.relativeMolecularMass()
        n = mat.numberDensityEffective # A^-3
        N = mat.totalAtoms 

        rho = (n/N)/0.6022*M
        V = m/rho

        R = (0.75/np.pi*V)**(1/3)

        chemical_formula = '-'.join([atm.symbol+str(no) for atm, no in zip(*mat.chemicalFormula())])

        print(chemical_formula)
        print('absoption cross section: {} barn'.format(sigma_a))
        print('scattering cross section: {} barn'.format(sigma_s))

        print('linear scattering coefficient: {} 1/cm'.format(n*sigma_s))
        print('linear absorption coefficient: {} 1/cm'.format(n*sigma_a))

        print('mass: {} g'.format(m))
        print('density: {} g/cm^3'.format(rho))

        print('volume: {} cm^3'.format(V))
        print('radius: {} cm'.format(R))

        print('total atoms: {}'.format(N))
        print('molar mass: {} g/mol'.format(M))
        print('number density: {} 1/A^3'.format(n))

        for key in self.peak_dict.keys():

            peaks = self.peak_dict.get(key)

            for peak in peaks:

                wls = peak.get_wavelengths()
                two_thetas = peak.get_scattering_angles()

                corrections = []

                for wl, two_theta in zip(wls, two_thetas):

                    mu = n*(sigma_s+(sigma_a/1.8)*wl)
                    muR = mu*R

                    #print('wavelength: {} ang'.format(wl))
                    #print('2theta: {} deg'.format(np.rad2deg(two_theta)))
                    #print('linear absorption coefficient: {} 1/cm'.format(mu))

                    correction = f(muR,np.rad2deg(two_theta))[0]

                    corrections.append(correction)

                peak.set_data_scale(corrections)

        # normalization

        mat = mtd['nws'].sample().getMaterial()

        sigma_a = mat.absorbXSection()
        sigma_s = mat.totalScatterXSection()

        m = self.norm_mass # g
        M = mat.relativeMolecularMass()
        n = mat.numberDensityEffective # A^-3
        N = mat.totalAtoms 

        rho = (n/N)/0.6022*M
        V = m/rho

        R = (0.75/np.pi*V)**(1/3)

        chemical_formula = '-'.join([atm.symbol+str(no) for atm, no in zip(*mat.chemicalFormula())])

        print(chemical_formula)
        print('absoption cross section: {} barn'.format(sigma_a))
        print('scattering cross section: {} barn'.format(sigma_s))

        print('linear scattering coefficient: {} 1/cm'.format(n*sigma_s))
        print('linear absorption coefficient: {} 1/cm'.format(n*sigma_a))

        print('mass: {} g'.format(m))
        print('density: {} g/cm^3'.format(rho))

        print('volume: {} cm^3'.format(V))
        print('radius: {} cm'.format(R))

        print('total atoms: {}'.format(N))
        print('molar mass: {} g/mol'.format(M))
        print('number density: {} 1/A^3'.format(n))

        for key in self.peak_dict.keys():

            peaks = self.peak_dict.get(key)

            for peak in peaks:

                intens = peak.get_merged_intensity()

                if intens > 0:

                    wls = peak.get_wavelengths()
                    two_thetas = peak.get_scattering_angles()

                    corrections = []

                    for wl, two_theta in zip(wls, two_thetas):

                        mu = n*(sigma_s+(sigma_a/1.8)*wl)
                        muR = mu*R

                        #print('wavelength: {} ang'.format(wl))
                        #print('2theta: {} deg'.format(np.rad2deg(two_theta)))
                        #print('linear absorption coefficient: {} 1/cm'.format(mu))

                        correction = f(muR,np.rad2deg(two_theta))[0]

                        corrections.append(correction)

                    peak.set_norm_scale(corrections)

        for pws in [self.iws, self.cws]:
            for pn in range(pws.getNumberPeaks()-1,-1,-1):
                pws.removePeak(pn)

        self.__repopulate_workspaces()

    def apply_extinction_correction(self, k):

        for key in self.peak_dict.keys():

            peaks = self.peak_dict.get(key)

            for peak in peaks:

                intenss = peak.get_intensity()
                wls = peak.get_wavelengths()
                two_thetas = peak.get_scattering_angles()
                scales = peak.get_data_scale()

                corrections = []

                for intens, wl, two_theta, scale in zip(intenss, wls, two_thetas, scales):

                    lorentz = wl**4/np.sin(two_theta/2)**2

                    correction = scale*0.5*(k*lorentz*intens+np.sqrt(4+(k*lorentz*intens)**2))

                    corrections.append(correction)

                peak.set_data_scale(corrections)

        for pws in [self.iws, self.cws]:
            for pn in range(pws.getNumberPeaks()-1,-1,-1):
                pws.removePeak(pn)

        self.__repopulate_workspaces()

class GaussianFit3D:

    def __init__(self, x, y, e, Q, sig):

        self.params = Parameters()

        self.params.add('A', value=y.max()-np.median(y), min=0, max=y.max())
        self.params.add('B', value=np.median(y), min=0, max=y.mean())

        self.params.add('mu0', value=Q[0], min=Q[0]-0.1, max=Q[0]+0.1)
        self.params.add('mu1', value=Q[1], min=Q[1]-0.1, max=Q[1]+0.1)
        self.params.add('mu2', value=Q[2], min=Q[2]-0.1, max=Q[2]+0.1)

        self.params.add('sig0', value=sig[0], min=0.25*sig[0], max=4*sig[0])
        self.params.add('sig1', value=sig[1], min=0.25*sig[0], max=4*sig[0])
        self.params.add('sig2', value=sig[2], min=0.25*sig[0], max=4*sig[0])

        self.params.add('rho12', value=0, min=-1, max=1)
        self.params.add('rho02', value=0, min=-1, max=1)
        self.params.add('rho01', value=0, min=-1, max=1)

        self.x = x
        self.y = y
        self.e = np.sqrt(e)

    def gaussian_3d(self, Q0, Q1, Q2, A, B, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01):

        U = np.array([[1, rho01, rho02],
                      [0,     1, rho12],
                      [0,     0,     1]])
        L = U.T

        sigma = np.array([[sig0, 0, 0],
                          [0, sig1, 0],
                          [0, 0, sig2]])

        S = np.dot(np.dot(L,sigma),U)

        inv_S = np.linalg.inv(S)

        x0, x1, x2 = Q0-mu0, Q1-mu1, Q2-mu2

        return A*np.exp(-0.5*(inv_S[0,0]*x0**2+inv_S[1,1]*x1**2+inv_S[2,2]*x2**2\
                          +2*(inv_S[1,2]*x1*x2+inv_S[0,2]*x0*x2+inv_S[0,1]*x0*x1)))+B

    def residual(self, params, x, y, e):
        
        Q0, Q1, Q2 = x

        A = params['A']
        B = params['B']

        mu0 = params['mu0']
        mu1 = params['mu1']
        mu2 = params['mu2']

        sig0 = params['sig0']
        sig1 = params['sig1']
        sig2 = params['sig2']

        rho12 = params['rho12']
        rho02 = params['rho02']
        rho01 = params['rho01']

        args = Q0, Q1, Q2, A, B, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01

        yfit = self.gaussian_3d(*args)

        return (y-yfit)/e

    def fit(self):

        out = Minimizer(self.residual, self.params, fcn_args=(self.x, self.y, self.e))
        result = out.minimize(method='leastsq')

        report_fit(result)

        A = result.params['A'].value
        B = result.params['B'].value

        mu0 = result.params['mu0'].value
        mu1 = result.params['mu1'].value
        mu2 = result.params['mu2'].value

        sig0 = result.params['sig0'].value
        sig1 = result.params['sig1'].value
        sig2 = result.params['sig2'].value

        rho12 = result.params['rho12'].value
        rho02 = result.params['rho02'].value
        rho01 = result.params['rho01'].value

        return A, B, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01

def draw_ellispoid(center2d, covariance2d, lscale=5.99):

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

    return xe, ye, x0, y0, x1, y1