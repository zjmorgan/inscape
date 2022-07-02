from mantid.simpleapi import CreateSingleValuedWorkspace, CreatePeaksWorkspace
from mantid.simpleapi import CloneWorkspace, DeleteWorkspace
from mantid.simpleapi import SortPeaksWorkspace, FilterPeaks
from mantid.simpleapi import SetUB, SaveIsawUB, FindUBUsingIndexedPeaks
from mantid.simpleapi import SetSampleMaterial, SaveNexus
from mantid.simpleapi import mtd

from mantid.kernel import V3D
from mantid import config

import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 8

import numpy as np
import scipy.interpolate
import scipy.optimize

from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth

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

class PeakEnvelope:

    def __init__(self, pdf_file):

        self.pp = PdfPages(pdf_file)

        #plt.close('peak-envelope')

        self.fig = plt.figure(num='peak-envelope', figsize=(18,6), dpi=144)
        gs = gridspec.GridSpec(1, 3, figure=self.fig, wspace=0.333, width_ratios=[0.2,0.2,0.6])

        gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.25)
        gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.25)
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[2], hspace=0.25, wspace=0.5)

        self.ax_Q = self.fig.add_subplot(gs0[0,0])
        self.ax_Q2 = self.fig.add_subplot(gs0[1,0])

        self.ax_Q.minorticks_on()
        self.ax_Q2.minorticks_on()

        self.ax_p_proj = self.fig.add_subplot(gs1[0,0])
        self.ax_s_proj = self.fig.add_subplot(gs1[1,0])

        self.ax_p_proj.minorticks_on()
        self.ax_s_proj.minorticks_on()

        self.ax_Qu = self.fig.add_subplot(gs2[0,0])
        self.ax_Qv = self.fig.add_subplot(gs2[0,1])
        self.ax_uv = self.fig.add_subplot(gs2[0,2])

        self.ax_Qu.minorticks_on()
        self.ax_Qv.minorticks_on()
        self.ax_uv.minorticks_on()

        self.ax_Qu_fit = self.fig.add_subplot(gs2[1,0])
        self.ax_Qv_fit = self.fig.add_subplot(gs2[1,1])
        self.ax_uv_fit = self.fig.add_subplot(gs2[1,2])

        self.ax_Qu_fit.minorticks_on()
        self.ax_Qv_fit.minorticks_on()
        self.ax_uv_fit.minorticks_on()

        #self.ax_p_proj.set_xlabel('Q\u2081 [\u212B\u207B\u00B9]') # [$\AA^{-1}$]
        self.ax_s_proj.set_xlabel('Q\u2081 [\u212B\u207B\u00B9]') # [$\AA^{-1}$]

        self.ax_p_proj.set_ylabel('Q\u2082 [\u212B\u207B\u00B9]') # [$\AA^{-1}$]
        self.ax_s_proj.set_ylabel('Q\u2082 [\u212B\u207B\u00B9]') # [$\AA^{-1}$]

        # self.ax_p_proj.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        # self.ax_s_proj.ticklabel_format(axis='both', style='sci', scilimits=(0,0))

        self.ax_Q.set_rasterization_zorder(100)
        self.ax_Q.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        self.line_Q_p, self.caps_Q_p, self.bars_Q_p = self.ax_Q.errorbar([0], [0], yerr=[0], fmt='.-', rasterized=False, zorder=2)
        self.norm_Q_p = self.ax_Q.plot([0], [0], '--', rasterized=False, zorder=1)
        self.line_Q_s, self.caps_Q_s, self.bars_Q_s = self.ax_Q.errorbar([0], [0], yerr=[0], fmt='.-', rasterized=False, zorder=2)
        self.norm_Q_s = self.ax_Q.plot([0], [0], '--', rasterized=False, zorder=1)

        self.im_p_proj = self.ax_p_proj.imshow([[0,1],[0,1]], interpolation='nearest',
                                               origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.im_s_proj = self.ax_s_proj.imshow([[0,1],[0,1]], interpolation='nearest',
                                               origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        divider_p = make_axes_locatable(self.ax_p_proj)
        divider_s = make_axes_locatable(self.ax_s_proj)

        width_p = axes_size.AxesY(self.ax_p_proj, aspect=0.05)
        width_s = axes_size.AxesY(self.ax_s_proj, aspect=0.05)

        pad_p = axes_size.Fraction(0.5, width_p)
        pad_s = axes_size.Fraction(0.5, width_p)

        cax_p = divider_p.append_axes('right', size=width_p, pad=pad_p)
        cax_s = divider_s.append_axes('right', size=width_s, pad=pad_s)

        self.cb_p = self.fig.colorbar(self.im_p_proj, cax=cax_p)
        self.cb_s = self.fig.colorbar(self.im_s_proj, cax=cax_s)

        self.cb_p.ax.minorticks_on()
        self.cb_s.ax.minorticks_on()

        self.elli_p = Ellipse((0,0), width=1, height=1, edgecolor='C3', facecolor='none', rasterized=False, zorder=1000)
        self.elli_s = Ellipse((0,0), width=1, height=1, edgecolor='C3', facecolor='none', rasterized=False, zorder=1000)

        self.trans_elli_p = transforms.Affine2D()
        self.trans_elli_s = transforms.Affine2D()

        self.ax_p_proj.add_patch(self.elli_p)
        self.ax_s_proj.add_patch(self.elli_s)

        self.ax_Q2.set_rasterization_zorder(100)
        self.ax_Q2.set_xlabel('Q\u209A [\u212B\u207B\u00B9]') # [$\AA^{-1}$]
        self.ax_Q2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        self.line_Q2_p, self.caps_Q2_p, self.bars_Q2_p = self.ax_Q2.errorbar([0], [0], yerr=[0], fmt='.-', rasterized=False, zorder=2)
        self.norm_Q2_p = self.ax_Q2.plot([0], [0], '--', rasterized=False, zorder=1)
        self.line_Q2_s, self.caps_Q2_s, self.bars_Q2_s = self.ax_Q2.errorbar([0], [0], yerr=[0], fmt='.-', rasterized=False, zorder=2)
        self.norm_Q2_s = self.ax_Q2.plot([0], [0], '--', rasterized=False, zorder=1)

        self.im_Qu = self.ax_Qu.imshow([[0,1],[0,1]], interpolation='nearest',
                                       origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.im_Qv = self.ax_Qv.imshow([[0,1],[0,1]], interpolation='nearest',
                                       origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.im_uv = self.ax_uv.imshow([[0,1],[0,1]], interpolation='nearest',
                                       origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.ax_Qu.set_rasterization_zorder(100)
        self.ax_Qv.set_rasterization_zorder(100)
        self.ax_uv.set_rasterization_zorder(100)

        self.im_Qu_fit = self.ax_Qu_fit.imshow([[0,1],[0,1]], interpolation='nearest',
                                               origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.im_Qv_fit = self.ax_Qv_fit.imshow([[0,1],[0,1]], interpolation='nearest',
                                               origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.im_uv_fit = self.ax_uv_fit.imshow([[0,1],[0,1]], interpolation='nearest',
                                               origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.ax_Qu_fit.set_rasterization_zorder(100)
        self.ax_Qv_fit.set_rasterization_zorder(100)
        self.ax_uv_fit.set_rasterization_zorder(100)

        self.peak_pu = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.inner_pu = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.outer_pu = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.peak_pv = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.inner_pv = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.outer_pv = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.peak_uv = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.inner_uv = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.outer_uv = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.peak_pu_fit = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.inner_pu_fit = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.outer_pu_fit = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.peak_pv_fit = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.inner_pv_fit = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.outer_pv_fit = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.peak_uv_fit = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.inner_uv_fit = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.outer_uv_fit = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.trans_peak_pu = transforms.Affine2D()
        self.trans_inner_pu = transforms.Affine2D()
        self.trans_outer_pu = transforms.Affine2D()

        self.trans_peak_pv = transforms.Affine2D()
        self.trans_inner_pv = transforms.Affine2D()
        self.trans_outer_pv = transforms.Affine2D()

        self.trans_peak_uv = transforms.Affine2D()
        self.trans_inner_uv = transforms.Affine2D()
        self.trans_outer_uv = transforms.Affine2D()

        self.trans_peak_pu_fit = transforms.Affine2D()
        self.trans_inner_pu_fit = transforms.Affine2D()
        self.trans_outer_pu_fit = transforms.Affine2D()

        self.trans_peak_pv_fit = transforms.Affine2D()
        self.trans_inner_pv_fit = transforms.Affine2D()
        self.trans_outer_pv_fit = transforms.Affine2D()

        self.trans_peak_uv_fit = transforms.Affine2D()
        self.trans_inner_uv_fit = transforms.Affine2D()
        self.trans_outer_uv_fit = transforms.Affine2D()

        self.ax_Qu.add_patch(self.peak_pu)
        self.ax_Qu.add_patch(self.inner_pu)
        self.ax_Qu.add_patch(self.outer_pu)

        self.ax_Qv.add_patch(self.peak_pv)
        self.ax_Qv.add_patch(self.inner_pv)
        self.ax_Qv.add_patch(self.outer_pv)

        self.ax_uv.add_patch(self.peak_uv)
        self.ax_uv.add_patch(self.inner_uv)
        self.ax_uv.add_patch(self.outer_uv)

        self.ax_Qu_fit.add_patch(self.peak_pu_fit)
        self.ax_Qu_fit.add_patch(self.inner_pu_fit)
        self.ax_Qu_fit.add_patch(self.outer_pu_fit)

        self.ax_Qv_fit.add_patch(self.peak_pv_fit)
        self.ax_Qv_fit.add_patch(self.inner_pv_fit)
        self.ax_Qv_fit.add_patch(self.outer_pv_fit)

        self.ax_uv_fit.add_patch(self.peak_uv_fit)
        self.ax_uv_fit.add_patch(self.inner_uv_fit)
        self.ax_uv_fit.add_patch(self.outer_uv_fit)

        # self.ax_Qv.set_xlabel('Q\u209A [\u212B\u207B\u00B9]') # [$\AA^{-1}$]
        # self.ax_Qu.set_xlabel('Q\u209A [\u212B\u207B\u00B9]') # [$\AA^{-1}$]
        # self.ax_uv.set_xlabel('Q\u2081\u2032 [\u212B\u207B\u00B9]') # [$\AA^{-1}$]

        self.ax_Qu.set_ylabel('Q\u2081\u2032 [\u212B\u207B\u00B9]') # [$\AA^{-1}$]
        self.ax_Qv.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [$\AA^{-1}$]
        self.ax_uv.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [$\AA^{-1}$]

        self.ax_Qv_fit.set_xlabel('Q\u209A [\u212B\u207B\u00B9]') # [$\AA^{-1}$]
        self.ax_Qu_fit.set_xlabel('Q\u209A [\u212B\u207B\u00B9]') # [$\AA^{-1}$]
        self.ax_uv_fit.set_xlabel('Q\u2081\u2032 [\u212B\u207B\u00B9]') # [$\AA^{-1}$]

        self.ax_Qu_fit.set_ylabel('Q\u2081\u2032 [\u212B\u207B\u00B9]') # [$\AA^{-1}$]
        self.ax_Qv_fit.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [$\AA^{-1}$]
        self.ax_uv_fit.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [$\AA^{-1}$]

        self.__show_plots = False

    def clear_plots(self, key, d, lamda, two_theta, n_runs):

        h, k, l, m, n, p = key

        if m**2+n**2+p**2 > 0:
            self.ax_Q.set_title('({} {} {}) ({} {} {})'.format(h,k,l,m,n,p))
        else:
            self.ax_Q.set_title('({} {} {})'.format(h,k,l))

        self.ax_Q2.set_title('')
        self.ax_p_proj.set_title('')

        barsy_p, = self.bars_Q_p
        barsy_s, = self.bars_Q_s

        self.line_Q_p.set_data([0],[0])
        self.line_Q_s.set_data([0],[0])

        barsy_p.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip([0], [0], [0])])
        barsy_s.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip([0], [0], [0])])

        self.norm_Q_p[0].set_data([0],[0])
        self.norm_Q_s[0].set_data([0],[0])

        self.ax_Q.relim()
        self.ax_Q.autoscale()

        # ---

        barsy_p, = self.bars_Q2_p
        barsy_s, = self.bars_Q2_s

        self.line_Q2_p.set_data([0],[0])
        self.line_Q2_s.set_data([0],[0])

        barsy_p.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip([0], [0], [0])])
        barsy_s.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip([0], [0], [0])])

        self.norm_Q2_p[0].set_data([0],[0])
        self.norm_Q2_s[0].set_data([0],[0])

        self.ax_Q2.relim()
        self.ax_Q2.autoscale()

        # ---

        self.ax_p_proj.set_title('d = {:.4f} \u212B'.format(d))
        self.ax_s_proj.set_title('')

        self.ax_p_proj.set_aspect(1)
        self.ax_s_proj.set_aspect(1)

        self.im_p_proj.set_array(np.c_[[0,1],[1,0]])
        self.im_s_proj.set_array(np.c_[[0,1],[1,0]])

        self.im_p_proj.autoscale()
        self.im_s_proj.autoscale()

        self.im_p_proj.set_extent([0,1,0,1])
        self.im_s_proj.set_extent([0,1,0,1])

        self.elli_p.width = 1
        self.elli_s.width = 1

        self.elli_p.height = 1
        self.elli_s.height = 1

        self.trans_elli_p.clear()
        self.trans_elli_s.clear()

        self.elli_p.set_transform(self.trans_elli_p+self.ax_p_proj.transData)
        self.elli_s.set_transform(self.trans_elli_s+self.ax_s_proj.transData)

        # ---

        if n_runs > 1:
            self.ax_Qu.set_title('\u03BB = {:.3f}-{:.3f} \u212B'.format(*lamda))
            self.ax_Qu_fit.set_title('2\u03B8 = {:.1f}-{:.1f}\u00B0'.format(*two_theta))

            self.ax_Qv.set_title('{} orientations'.format(n_runs))
        else:
            self.ax_Qu.set_title('\u03BB = {:.3f} \u212B'.format(lamda[0]))
            self.ax_Qu_fit.set_title('2\u03B8 = {:.1f}\u00B0'.format(two_theta[0]))

            self.ax_Qv.set_title('1 orientation')

        self.ax_Qv_fit.set_title('')
            
        self.ax_uv.set_title('')
        self.ax_uv_fit.set_title('')

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

        self.im_Qu_fit.set_array(np.c_[[0,1],[1,0]])
        self.im_Qv_fit.set_array(np.c_[[0,1],[1,0]])
        self.im_uv_fit.set_array(np.c_[[0,1],[1,0]])

        self.im_Qu_fit.autoscale()
        self.im_Qv_fit.autoscale()
        self.im_uv_fit.autoscale()

        self.im_Qu_fit.set_extent([0,1,0,1])
        self.im_Qv_fit.set_extent([0,1,0,1])
        self.im_uv_fit.set_extent([0,1,0,1])

        self.ax_Qu_fit.set_aspect(1)
        self.ax_Qv_fit.set_aspect(1)
        self.ax_uv_fit.set_aspect(1)

        # ---

        self.peak_pu.width = 1
        self.inner_pu.width = 1
        self.outer_pu.width = 1

        self.peak_pu.height = 1
        self.inner_pu.height = 1
        self.outer_pu.height = 1

        self.trans_peak_pu.clear()
        self.trans_inner_pu.clear()
        self.trans_outer_pu.clear()

        self.peak_pu.set_transform(self.trans_peak_pu+self.ax_Qu.transData)
        self.inner_pu.set_transform(self.trans_inner_pu+self.ax_Qu.transData)
        self.outer_pu.set_transform(self.trans_outer_pu+self.ax_Qu.transData)

        # ---

        self.peak_pv.width = 1
        self.inner_pv.width = 1
        self.outer_pv.width = 1

        self.peak_pv.height = 1
        self.inner_pv.height = 1
        self.outer_pv.height = 1

        self.trans_peak_pv.clear()
        self.trans_inner_pv.clear()
        self.trans_outer_pv.clear()

        self.peak_pv.set_transform(self.trans_peak_pv+self.ax_Qv.transData)
        self.inner_pv.set_transform(self.trans_inner_pv+self.ax_Qv.transData)
        self.outer_pv.set_transform(self.trans_outer_pv+self.ax_Qv.transData)

        # ---

        self.peak_uv.width = 1
        self.inner_uv.width = 1
        self.outer_uv.width = 1

        self.peak_uv.height = 1
        self.inner_uv.height = 1
        self.outer_uv.height = 1

        self.trans_peak_uv.clear()
        self.trans_inner_uv.clear()
        self.trans_outer_uv.clear()

        self.peak_uv.set_transform(self.trans_peak_uv+self.ax_uv.transData)
        self.inner_uv.set_transform(self.trans_inner_uv+self.ax_uv.transData)
        self.outer_uv.set_transform(self.trans_outer_uv+self.ax_uv.transData)

        # ---

        self.peak_pu_fit.width = 1
        self.inner_pu_fit.width = 1
        self.outer_pu_fit.width = 1

        self.peak_pu_fit.height = 1
        self.inner_pu_fit.height = 1
        self.outer_pu_fit.height = 1

        self.trans_peak_pu_fit.clear()
        self.trans_inner_pu_fit.clear()
        self.trans_outer_pu_fit.clear()

        self.peak_pu_fit.set_transform(self.trans_peak_pu_fit+self.ax_Qu_fit.transData)
        self.inner_pu_fit.set_transform(self.trans_inner_pu_fit+self.ax_Qu_fit.transData)
        self.outer_pu_fit.set_transform(self.trans_outer_pu_fit+self.ax_Qu_fit.transData)

        # ---

        self.peak_pv_fit.width = 1
        self.inner_pv_fit.width = 1
        self.outer_pv_fit.width = 1

        self.peak_pv_fit.height = 1
        self.inner_pv_fit.height = 1
        self.outer_pv_fit.height = 1

        self.trans_peak_pv_fit.clear()
        self.trans_inner_pv_fit.clear()
        self.trans_outer_pv_fit.clear()

        self.peak_pv_fit.set_transform(self.trans_peak_pv_fit+self.ax_Qv_fit.transData)
        self.inner_pv_fit.set_transform(self.trans_inner_pv_fit+self.ax_Qv_fit.transData)
        self.outer_pv_fit.set_transform(self.trans_outer_pv_fit+self.ax_Qv_fit.transData)

        # ---

        self.peak_uv_fit.width = 1
        self.inner_uv_fit.width = 1
        self.outer_uv_fit.width = 1

        self.peak_uv_fit.height = 1
        self.inner_uv_fit.height = 1
        self.outer_uv_fit.height = 1

        self.trans_peak_uv_fit.clear()
        self.trans_inner_uv_fit.clear()
        self.trans_outer_uv_fit.clear()

        self.peak_uv_fit.set_transform(self.trans_peak_uv_fit+self.ax_uv_fit.transData)
        self.inner_uv_fit.set_transform(self.trans_inner_uv_fit+self.ax_uv_fit.transData)
        self.outer_uv_fit.set_transform(self.trans_outer_uv_fit+self.ax_uv_fit.transData)

    def show_plots(self, show):

        self.__show_plots = show

    def create_pdf(self):

        self.pp.close()

    def plot_Q(self, x, y, y0, yerr, y_fit, y_bkg):

        if np.any(y > 0):

            barsy_p, = self.bars_Q_p
            barsy_s, = self.bars_Q_s

            self.line_Q_p.set_data(x,y0)
            self.line_Q_s.set_data(x,y)

            barsy_p.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(x, y0+yerr, y0-yerr)])
            barsy_s.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(x, y+yerr, y-yerr)])

            self.norm_Q_p[0].set_data(x,y_bkg)
            self.norm_Q_s[0].set_data(x,y_fit)

            self.ax_Q.relim()
            self.ax_Q.autoscale()

            if self.__show_plots: self.fig.show()

    def plot_extracted_Q(self, x, y, y0, yerr, y_fit, y_bkg, chi_sq):

        if np.any(y > 0):

            barsy_p, = self.bars_Q2_p
            barsy_s, = self.bars_Q2_s

            self.line_Q2_p.set_data(x,y0)
            self.line_Q2_s.set_data(x,y)

            barsy_p.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(x, y0+yerr, y0-yerr)])
            barsy_s.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(x, y+yerr, y-yerr)])

            self.norm_Q2_p[0].set_data(x,y_bkg)
            self.norm_Q2_s[0].set_data(x,y_fit)

            self.ax_Q2.relim()
            self.ax_Q2.autoscale()

            self.ax_Q2.set_title('\u03A7\u00B2 = {:.4f}'.format(chi_sq))

            if self.__show_plots: self.fig.show()

    def plot_projection(self, z, z0, x_extents, y_extents, mu, sigma, rho, chi_sq):

        if np.any(z > 0) and np.any(z0 > 0) and np.diff(x_extents) > 0 and np.diff(y_extents) > 0 and np.all(sigma):

            self.im_p_proj.set_array(z0.T)
            self.im_s_proj.set_array(z.T)

            self.im_p_proj.autoscale()
            self.im_s_proj.autoscale()

            extents = [*x_extents, *y_extents]

            self.im_p_proj.set_extent(extents)
            self.im_s_proj.set_extent(extents)

            self.ax_s_proj.set_title('\u03A7\u00B2 = {:.4f}'.format(chi_sq))

            mu_x, mu_y = mu
            sigma_x, sigma_y = sigma

            rx = np.sqrt(1+rho)
            ry = np.sqrt(1-rho)

            scale_x = 3*sigma_x
            scale_y = 3*sigma_y

            self.elli_p.width = 2*rx
            self.elli_s.width = 2*rx

            self.elli_p.height = 2*ry
            self.elli_s.height = 2*ry

            self.trans_elli_p.rotate_deg(45).scale(scale_x,scale_y).translate(mu_x,mu_y)
            self.trans_elli_s.rotate_deg(45).scale(scale_x,scale_y).translate(mu_x,mu_y)

            self.elli_p.set_transform(self.trans_elli_p+self.ax_p_proj.transData)
            self.elli_s.set_transform(self.trans_elli_s+self.ax_s_proj.transData)

            if self.__show_plots: self.fig.show()

    def plot_integration(self, signal, u_extents, v_extents, Q_extents, centers, radii, scales):

        if np.any(signal > 0) and np.diff(u_extents[0::2]) > 0 and np.diff(v_extents[0::2]) > 0 and np.diff(Q_extents[0::2]) > 0:

            Qu = np.nansum(signal, axis=1)
            Qv = np.nansum(signal, axis=0)
            uv = np.nansum(signal, axis=2).T

            self.im_Qu.set_array(Qu)
            self.im_Qv.set_array(Qv)
            self.im_uv.set_array(uv)

            self.im_Qu.autoscale()
            self.im_Qv.autoscale()
            self.im_uv.autoscale()

            clim_Qu = self.im_Qu.get_clim()
            clim_Qv = self.im_Qv.get_clim()
            clim_uv = self.im_uv.get_clim()

            self.im_Qu_fit.set_clim(*clim_Qu)
            self.im_Qv_fit.set_clim(*clim_Qv)
            self.im_uv_fit.set_clim(*clim_uv)

            Qu_extents = [Q_extents[0],Q_extents[2],u_extents[0],u_extents[2]]
            Qv_extents = [Q_extents[0],Q_extents[2],v_extents[0],v_extents[2]]
            uv_extents = [u_extents[0],u_extents[2],v_extents[0],v_extents[2]]

            self.im_Qu.set_extent(Qu_extents)
            self.im_Qv.set_extent(Qv_extents)
            self.im_uv.set_extent(uv_extents)

            self.im_Qu_fit.set_extent(Qu_extents)
            self.im_Qv_fit.set_extent(Qv_extents)
            self.im_uv_fit.set_extent(uv_extents)

            Qu_aspect = Q_extents[1]/u_extents[1]
            Qv_aspect = Q_extents[1]/v_extents[1]
            uv_aspect = u_extents[1]/v_extents[1]

            self.ax_Qu.set_aspect(Qu_aspect)
            self.ax_Qv.set_aspect(Qv_aspect)
            self.ax_uv.set_aspect(uv_aspect)

            self.ax_Qu_fit.set_aspect(Qu_aspect)
            self.ax_Qv_fit.set_aspect(Qv_aspect)
            self.ax_uv_fit.set_aspect(uv_aspect)

            u_center, v_center, Q_center = centers

            u_size = 2*radii[0]
            v_size = 2*radii[1]
            Q_size = 2*radii[2]

            pk_scale, in_scale, out_scale = scales

            # --

            self.peak_pu.width = Q_size*pk_scale
            self.inner_pu.width = Q_size*in_scale
            self.outer_pu.width = Q_size*out_scale

            self.peak_pu.height = u_size*pk_scale
            self.inner_pu.height = u_size*in_scale
            self.outer_pu.height = u_size*out_scale

            self.trans_peak_pu.rotate_deg(0).scale(1,1).translate(Q_center,u_center)
            self.trans_inner_pu.rotate_deg(0).scale(1,1).translate(Q_center,u_center)
            self.trans_outer_pu.rotate_deg(0).scale(1,1).translate(Q_center,u_center)

            self.peak_pu.set_transform(self.trans_peak_pu+self.ax_Qu.transData)
            self.inner_pu.set_transform(self.trans_inner_pu+self.ax_Qu.transData)
            self.outer_pu.set_transform(self.trans_outer_pu+self.ax_Qu.transData)

            # ---

            self.peak_pv.width = Q_size*pk_scale
            self.inner_pv.width = Q_size*in_scale
            self.outer_pv.width = Q_size*out_scale

            self.peak_pv.height = v_size*pk_scale
            self.inner_pv.height = v_size*in_scale
            self.outer_pv.height = v_size*out_scale

            self.trans_peak_pv.rotate_deg(0).scale(1,1).translate(Q_center,v_center)
            self.trans_inner_pv.rotate_deg(0).scale(1,1).translate(Q_center,v_center)
            self.trans_outer_pv.rotate_deg(0).scale(1,1).translate(Q_center,v_center)

            self.peak_pv.set_transform(self.trans_peak_pv+self.ax_Qv.transData)
            self.inner_pv.set_transform(self.trans_inner_pv+self.ax_Qv.transData)
            self.outer_pv.set_transform(self.trans_outer_pv+self.ax_Qv.transData)

            # ---

            self.peak_uv.width = u_size*pk_scale
            self.inner_uv.width = u_size*in_scale
            self.outer_uv.width = u_size*out_scale

            self.peak_uv.height = v_size*pk_scale
            self.inner_uv.height = v_size*in_scale
            self.outer_uv.height = v_size*out_scale

            self.trans_peak_uv.rotate_deg(0).scale(1,1).translate(u_center,v_center)
            self.trans_inner_uv.rotate_deg(0).scale(1,1).translate(u_center,v_center)
            self.trans_outer_uv.rotate_deg(0).scale(1,1).translate(u_center,v_center)

            self.peak_uv.set_transform(self.trans_peak_uv+self.ax_uv.transData)
            self.inner_uv.set_transform(self.trans_inner_uv+self.ax_uv.transData)
            self.outer_uv.set_transform(self.trans_outer_uv+self.ax_uv.transData)

            # --

            self.peak_pu_fit.width = Q_size*pk_scale
            self.inner_pu_fit.width = Q_size*in_scale
            self.outer_pu_fit.width = Q_size*out_scale

            self.peak_pu_fit.height = u_size*pk_scale
            self.inner_pu_fit.height = u_size*in_scale
            self.outer_pu_fit.height = u_size*out_scale

            self.trans_peak_pu_fit.rotate_deg(0).scale(1,1).translate(Q_center,u_center)
            self.trans_inner_pu_fit.rotate_deg(0).scale(1,1).translate(Q_center,u_center)
            self.trans_outer_pu_fit.rotate_deg(0).scale(1,1).translate(Q_center,u_center)

            self.peak_pu_fit.set_transform(self.trans_peak_pu_fit+self.ax_Qu_fit.transData)
            self.inner_pu_fit.set_transform(self.trans_inner_pu_fit+self.ax_Qu_fit.transData)
            self.outer_pu_fit.set_transform(self.trans_outer_pu_fit+self.ax_Qu_fit.transData)

            # ---

            self.peak_pv_fit.width = Q_size*pk_scale
            self.inner_pv_fit.width = Q_size*in_scale
            self.outer_pv_fit.width = Q_size*out_scale

            self.peak_pv_fit.height = v_size*pk_scale
            self.inner_pv_fit.height = v_size*in_scale
            self.outer_pv_fit.height = v_size*out_scale

            self.trans_peak_pv_fit.rotate_deg(0).scale(1,1).translate(Q_center,v_center)
            self.trans_inner_pv_fit.rotate_deg(0).scale(1,1).translate(Q_center,v_center)
            self.trans_outer_pv_fit.rotate_deg(0).scale(1,1).translate(Q_center,v_center)

            self.peak_pv_fit.set_transform(self.trans_peak_pv_fit+self.ax_Qv_fit.transData)
            self.inner_pv_fit.set_transform(self.trans_inner_pv_fit+self.ax_Qv_fit.transData)
            self.outer_pv_fit.set_transform(self.trans_outer_pv_fit+self.ax_Qv_fit.transData)

            # ---

            self.peak_uv_fit.width = u_size*pk_scale
            self.inner_uv_fit.width = u_size*in_scale
            self.outer_uv_fit.width = u_size*out_scale

            self.peak_uv_fit.height = v_size*pk_scale
            self.inner_uv_fit.height = v_size*in_scale
            self.outer_uv_fit.height = v_size*out_scale

            self.trans_peak_uv_fit.rotate_deg(0).scale(1,1).translate(u_center,v_center)
            self.trans_inner_uv_fit.rotate_deg(0).scale(1,1).translate(u_center,v_center)
            self.trans_outer_uv_fit.rotate_deg(0).scale(1,1).translate(u_center,v_center)

            self.peak_uv_fit.set_transform(self.trans_peak_uv_fit+self.ax_uv_fit.transData)
            self.inner_uv_fit.set_transform(self.trans_inner_uv_fit+self.ax_uv_fit.transData)
            self.outer_uv_fit.set_transform(self.trans_outer_uv_fit+self.ax_uv_fit.transData)

            if self.__show_plots: self.fig.show()

    def plot_fitting(self, signal, I, sig, chi_sq):

        if np.any(signal > 0):

            Qu = np.nansum(signal, axis=1)
            Qv = np.nansum(signal, axis=0)
            uv = np.nansum(signal, axis=2).T

            self.im_Qu_fit.set_array(Qu)
            self.im_Qv_fit.set_array(Qv)
            self.im_uv_fit.set_array(uv)

            self.ax_uv.set_title('I = {:.2e} \u00B1 {:.2e} [arb. unit]'.format(I[0], sig[0]))
            self.ax_uv_fit.set_title('I = {:.2e} \u00B1 {:.2e} [arb. unit]'.format(I[1], sig[1]))

            # op = ' < ' if Dmax < Dn_crit else ' >= '
            # self.ax_Qv_fit.set_title('D\u2099 = {:.3}'.format(Dmax)+op+'{:.3}'.format(Dn_crit))
            
            self.ax_Qv_fit.set_title('\u03A7\u00B2 = {:.4f}'.format(chi_sq))

            if self.__show_plots: self.fig.show()

    def write_figure(self):

        self.pp.savefig(self.fig)

class PeakInformation:

    def __init__(self, scale_constant):

        self.__peak_num = 0
        self.__run_num = []
        self.__bank_num = []
        self.__peak_ind = []
        self.__row = []
        self.__col = []

        self.__D = np.eye(3)
        self.__W = np.eye(3)

        self.__bin_size = np.zeros(3)
        self.__Q = np.zeros(3)

        self.__pk_data = None
        self.__pk_norm = None

        self.__bkg_data = None
        self.__bkg_norm = None

        self.__Q0 = None
        self.__Q1 = None
        self.__Q2 = None

        self.data = None
        self.norm = None

        self.__peak_fit = 0.0
        self.__peak_bkg_ratio = 0.0
        self.__peak_score = 0.0

        self.__peak_fit2d = 0.0
        self.__peak_bkg_ratio2d = 0.0
        self.__peak_score2d = 0.0

        self.__data_scale = np.array([])
        self.__norm_scale = np.array([])

        self.__ext_constants = None
        self.__tbar = None

        self.__wl = []
        self.__two_theta = []
        self.__az_phi = []

        self.__phi = []
        self.__chi = []
        self.__omega = []

        self.__est_int = []
        self.__est_int_err = []

        self.__scale_constant = scale_constant

        self.__intens_fit = 0
        self.__bkg_fit = 0

        self.__sig_fit = 0

        self.__mu_1d = None
        self.__sigma_1d = None

        self.__mu_x_2d = None
        self.__mu_y_2d = None
        self.__sigma_x_2d = None
        self.__sigma_y_2d = None
        self.__rho_2d = None

    def get_Q(self):

        return self.__Q

    def set_Q(self, Q):

        self.__Q = Q

    def get_A(self):
        
        W = self.get_W()
        D = self.get_D()

        return np.dot(np.dot(W,D),W.T)

    def get_D(self):

        return self.__D

    def set_D(self, D):

        self.__D = D
        
    def get_W(self):

        return self.__W

    def set_W(self, W):

        self.__W = W

    def set_peak_constant(self, scale_constant):

        self.__scale_constant = scale_constant
        
    def get_peak_constant(self):

        return self.__scale_constant

    def get_bin_size(self):

        return self.__bin_size

    def set_ext_constant(self, c):

        self.__ext_constant = c

    def get_ext_constant(self):

        if not hasattr(self, '_PeakInformation__ext_constant'):
           self.__ext_constant = 0

        return self.__ext_constant

    def set_data_scale(self, corr_scale):

        self.__data_scale = np.array(corr_scale)

    def set_transmission_coefficient(self, T):

        self.__t = np.array(T)

    def get_transmission_coefficient(self):

        if not hasattr(self, '_PeakInformation__t'):
            self.__t = np.ones_like(self.__run_num)

        return self.__t

    def set_weighted_mean_path_length(self, Tbar):

        self.__tbar = np.array(Tbar)

    def get_weighted_mean_path_length(self):

        if not hasattr(self, '_PeakInformation__tbar'):
            self.__tbar = np.ones_like(self.__run_num)

        return self.__tbar

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

    def get_partial_merged_peak_volume_fraction(self, indices):

        return self.__partial_merge_pk_vol_fract(indices)

    def get_partial_merged_intensity(self, indices):

        return self.__partial_merge_intensity(indices)

    def get_partial_merged_intensity_error(self, indices):

        return self.__partial_merge_intensity_error(indices)

    def get_peak_volume_fraction(self):

        return self.__pk_vol_fract()

    def get_intensity(self, normalize=True):

        return self.__intensity(normalize=normalize)

    def get_intensity_error(self, normalize=True):

        return self.__intensity_error(normalize=normalize)

    def get_goniometers(self):

        R = []

        for phi, chi, omega in zip(self.__phi,self.__chi,self.__omega):
            R.append(np.dot(self.__R_y(omega), np.dot(self.__R_z(chi), self.__R_y(phi))))

        return R

    def get_rotation_angle(self):

        varphi = []

        for phi, chi, omega in zip(self.__phi,self.__chi,self.__omega):
            R = np.dot(self.__R_y(omega), np.dot(self.__R_z(chi), self.__R_y(phi)))
            varphi.append(np.arccos((np.trace(R)-1)/2))

        return varphi

    def get_rotation_axis(self):

        u = []

        for phi, chi, omega in zip(self.__phi,self.__chi,self.__omega):
            R = np.dot(self.__R_y(omega), np.dot(self.__R_z(chi), self.__R_y(phi)))
            val, vec = np.linalg.eig(R)
            u.append(vec[:,np.argwhere(np.isclose(val, 1))[0][0]].real)

        return u

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

    def get_lorentz_factors(self, laue=True):

        two_theta = self.get_scattering_angles()
        az_phi = self.get_azimuthal_angles()
        lamda = self.get_wavelengths()

        if laue:
            factors = lamda**4/np.sin(two_theta/2)**2
        else:
            factors = lamda**3/np.abs(np.sin(two_theta)*np.cos(az_phi))

        return factors

    def __get_lorentz_factors(self, indices, laue=True):

        factors = self.get_lorentz_factors(laue=laue)

        return np.array([factors[ind] for ind in indices])

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
              'Ellispoid': self.__round(self.get_A()[np.triu_indices(3)],2),
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
              'ExctinctionScale': self.get_ext_constant(),
              'PeakScaleConstant': self.__round(self.get_peak_constant(),2),
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

    def add_integration(self, Q, D, W, statistics, data_norm, pk_bkg):

        peak_fit, peak_bkg_ratio, sig_noise_ratio, peak_fit2d, peak_bkg_ratio2d, sig_noise_ratio2d = statistics

        self.__Q = Q

        self.__D = D
        self.__W = W

        pk_data, pk_norm, bkg_data, bkg_norm, bin_size = pk_bkg
        Q0, Q1, Q2, data, norm = data_norm

        self.__bin_size = bin_size

        self.__peak_fit = peak_fit
        self.__peak_bkg_ratio = peak_bkg_ratio
        self.__peak_score = sig_noise_ratio

        self.__peak_fit2d = peak_fit2d
        self.__peak_bkg_ratio2d = peak_bkg_ratio2d
        self.__peak_score2d = sig_noise_ratio2d

        self.__pk_data = pk_data
        self.__pk_norm = pk_norm

        self.__bkg_data = bkg_data
        self.__bkg_norm = bkg_norm

        self.__data = data
        self.__norm = norm

        self.__Q0 = Q0
        self.__Q1 = Q1
        self.__Q2 = Q2

        self.__norm_scale = np.ones(len(pk_data))
        self.__data_scale = np.ones(len(pk_data))

        self.__abs_scale = np.ones(len(pk_data))
        self.__ext_scale = np.ones(len(pk_data))

        self.__tbar = np.ones(len(pk_data))
        self.__t = np.ones(len(pk_data))

    def add_partial_integration(self, Q, A, peak_fit, peak_bkg_ratio, peak_score):

        self.__Q = Q
        self.__A = A

        self.__peak_fit = peak_fit
        self.__peak_bkg_ratio = peak_bkg_ratio
        self.__peak_score = peak_score

    def add_fit(self, fit_1d, fit_2d, fit_prod):

        intens_fit, bkg_fit, sig_sq = fit_prod

        self.__intens_fit = intens_fit
        self.__bkg_fit = bkg_fit

        self.__sig_fit = sig_sq

        mu_1d, sigma_1d = fit_1d # a_1d, b_1d, c_1d

        self.__mu_1d = mu_1d
        self.__sigma_1d = sigma_1d

        mu_x_2d, mu_y_2d, sigma_x_2d, sigma_y_2d, rho_2d = fit_2d # a_2d, b_2d, cx_2d, cy_2d, cxy_2d

        self.__mu_x_2d = mu_x_2d
        self.__mu_y_2d = mu_y_2d
        self.__sigma_x_2d = sigma_x_2d
        self.__sigma_y_2d = sigma_y_2d
        self.__rho_2d = rho_2d

    # ---

    def __merge_pk_vol_fract(self):

        return self.__partial_merge_pk_vol_fract(self.__good_intensities())

    def __merge_pk_bkg_ratio(self):

        return self.__partial_merge_pk_bkg_ratio(self.__good_intensities())

    def __merge_intensity(self):

        return self.__partial_merge_intensity(self.__good_intensities())

    def __merge_intensity_error(self):

        return self.__partial_merge_intensity_error(self.__good_intensities())

    # ---

    def __partial_merge_pk_vol_fract(self, indices):

        if not self.__is_peak_integrated() or len(indices) == 0:

            return 0.0

        else:

            data = self.__get_partial_merged_peak_data_arrays(indices)
            norm = self.__get_partial_merged_peak_norm_arrays(indices)

            pk_data = np.sum(data, axis=0)
            pk_norm = np.sum(norm, axis=0)

            pk_vol_fract = np.sum(~(np.isnan(pk_data/pk_norm)))/len(data[0])

            return pk_vol_fract

    def __partial_merge_pk_bkg_ratio(self, indices):

        if not self.__is_peak_integrated() or len(indices) == 0:

            return 0.0

        else:

            data = self.__get_partial_merged_peak_data_arrays(indices)
            norm = self.__get_partial_merged_peak_norm_arrays(indices)

            bkg_data = self.__get_partial_merged_background_data_arrays(indices)
            bkg_norm = self.__get_partial_merged_background_norm_arrays(indices)

            pk_vol = np.sum(~np.isnan([data,norm]).any(axis=0))
            bkg_vol = np.sum(~np.isnan([bkg_data,bkg_norm]).any(axis=0))

            return pk_vol/bkg_vol

    def __partial_merge_intensity(self, indices):

        if not self.__is_peak_integrated() or len(indices) == 0:

            return 0.0

        else:

            data = self.__get_partial_merged_peak_data_arrays(indices)
            norm = self.__get_partial_merged_peak_norm_arrays(indices)

            bkg_data = self.__get_partial_merged_background_data_arrays(indices)
            bkg_norm = self.__get_partial_merged_background_norm_arrays(indices)

            scale_data = self.get_partial_merged_data_scale(indices)[:,np.newaxis]
            scale_norm = self.get_partial_merged_norm_scale(indices)[:,np.newaxis]

            volume_ratio = self.__partial_merge_pk_bkg_ratio(indices)

            constant = self.get_peak_constant()*np.prod(self.get_bin_size())

            data_scale, norm_scale = np.multiply(data, scale_data), np.multiply(norm, scale_norm)
            bkg_data_scale, bkg_norm_scale = np.multiply(bkg_data, scale_data), np.multiply(bkg_norm, scale_norm)

            data_norm = np.nansum(data_scale, axis=0)/np.nansum(norm_scale, axis=0)
            bkg_data_norm = np.nansum(bkg_data_scale, axis=0)/np.nansum(bkg_norm_scale, axis=0)

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

    def __partial_merge_intensity_error(self, indices):

        if not self.__is_peak_integrated() or len(indices) == 0:

            return 0.0

        else:

            sig_fit = self.__sig_fit

            data = self.__get_partial_merged_peak_data_arrays(indices)
            norm = self.__get_partial_merged_peak_norm_arrays(indices)

            bkg_data = self.__get_partial_merged_background_data_arrays(indices)
            bkg_norm = self.__get_partial_merged_background_norm_arrays(indices)

            scale_data = self.get_partial_merged_data_scale(indices)[:,np.newaxis]
            scale_norm = self.get_partial_merged_norm_scale(indices)[:,np.newaxis]

            volume_ratio = self.__partial_merge_pk_bkg_ratio(indices)

            constant = self.get_peak_constant()*np.prod(self.get_bin_size())

            data_scale, norm_scale = np.multiply(data, scale_data), np.multiply(norm, scale_norm)
            bkg_data_scale, bkg_norm_scale = np.multiply(bkg_data, scale_data), np.multiply(bkg_norm, scale_norm)

            data_norm = np.nansum(data_scale, axis=0)/np.nansum(norm_scale, axis=0)**2
            bkg_data_norm = np.nansum(bkg_data_scale, axis=0)/np.nansum(bkg_norm_scale, axis=0)**2

            data_norm[np.isinf(data_norm)] = np.nan
            bkg_data_norm[np.isinf(bkg_data_norm)] = np.nan

            intens = np.nansum(data_norm)
            bkg_intens = np.nansum(bkg_data_norm)

            intensity = np.sqrt(intens+bkg_intens*volume_ratio**2+sig_fit**2)*constant

            return intensity

    # ---

    def __pk_vol_fract(self):

        if not self.__is_peak_integrated():

            return np.array([])

        else:

            data = self.__get_peak_data_arrays()
            norm = self.__get_peak_norm_arrays()

            pk_vol_fract = np.sum(~(np.isnan(np.array(data)/np.array(norm))), axis=1)/len(data[0])

            return pk_vol_fract

    def __pk_bkg_ratio(self, normalize=True):

        if not self.__is_peak_integrated():

            return np.array([])

        else:

            data = self.__get_peak_data_arrays()
            norm = self.__get_peak_norm_arrays()

            bkg_data = self.__get_background_data_arrays()
            bkg_norm = self.__get_background_norm_arrays()

            if normalize:
                pk_vol = np.sum(~np.isnan([data,norm]).any(axis=0),axis=1)
                bkg_vol = np.sum(~np.isnan([bkg_data,bkg_norm]).any(axis=0),axis=1)
            else:
                pk_vol = np.sum(~np.isnan([data]).any(axis=0),axis=1)
                bkg_vol = np.sum(~np.isnan([bkg_data]).any(axis=0),axis=1)

            return pk_vol/bkg_vol

    def __intensity(self, normalize=True):

        if not self.__is_peak_integrated():

            return np.array([])

        else:

            data = self.__get_peak_data_arrays()
            norm = self.__get_peak_norm_arrays()

            bkg_data = self.__get_background_data_arrays()
            bkg_norm = self.__get_background_norm_arrays()

            scale_data = self.__data_scale[:,np.newaxis]
            scale_norm = self.__norm_scale[:,np.newaxis]

            volume_ratio = self.__pk_bkg_ratio(normalize=normalize)

            if normalize:

                constant = self.get_peak_constant()*np.prod(self.get_bin_size())

                data_scale, norm_scale = np.multiply(data, scale_data), np.multiply(norm, scale_norm)
                bkg_data_scale, bkg_norm_scale = np.multiply(bkg_data, scale_data), np.multiply(bkg_norm, scale_norm)

                intens = np.nansum(data_scale/norm_scale, axis=1)
                bkg_intens = np.nansum(bkg_data_scale/bkg_norm_scale, axis=1)

            else:

                constant = self.get_peak_constant()

                intens = np.nansum(data, axis=1)
                bkg_intens = np.nansum(bkg_data, axis=1)

            intensity = (intens-np.multiply(bkg_intens,volume_ratio))*constant

            return intensity

    def __intensity_error(self, normalize=True):

        if not self.__is_peak_integrated():

            return np.array([])

        else:

            data = self.__get_peak_data_arrays()
            norm = self.__get_peak_norm_arrays()

            bkg_data = self.__get_background_data_arrays()
            bkg_norm = self.__get_background_norm_arrays()

            scale_data = self.get_data_scale()[:,np.newaxis]
            scale_norm = self.get_norm_scale()[:,np.newaxis]

            volume_ratio = self.__pk_bkg_ratio(normalize=normalize)

            if normalize:

                constant = self.get_peak_constant()*np.prod(self.get_bin_size())

                data_scale, norm_scale = np.multiply(data, scale_data), np.multiply(norm, scale_norm)
                bkg_data_scale, bkg_norm_scale = np.multiply(bkg_data, scale_data), np.multiply(bkg_norm, scale_norm)

                intens = np.nansum(data_scale/norm_scale**2, axis=1)
                bkg_intens = np.nansum(bkg_data_scale/bkg_norm_scale**2, axis=1)

            else:

                constant = self.get_peak_constant()

                intens = np.nansum(data, axis=1)
                bkg_intens = np.nansum(bkg_data, axis=1)

            intensity = np.sqrt(intens+np.multiply(bkg_intens,volume_ratio**2))*constant

            return intensity

#     def __get_extinction_scale(self, indices, laue=True):
# 
#         L = self.get_lorentz_factors(laue=laue)
#         Tbar = 1#self.get_weighted_mean_path_length()
# 
#         c = self.get_ext_constant()
#         constant = self.get_peak_constant()
# 
#         clusters = self.get_peak_clusters()
# 
#         X = L*Tbar
# 
#         intensity = np.zeros(len(X))
#         factors = np.zeros(len(X))
# 
#         for cluster in clusters:
# 
#             intensity[cluster] = self.__partial_merge_intensity(cluster, ext_corr=False)/constant
#             factors[cluster] = np.mean(X[cluster])
# 
#         scale = 0.5*(c*factors*intensity+np.sqrt(4+(c*factors*intensity)**2))
# 
#         return np.array([scale[ind] for ind in indices])
# 
#     def get_peak_clusters(self, laue=True, quantile=0.25):
# 
#         L = self.get_lorentz_factors(laue=laue)
#         Tbar = 1#self.get_weighted_mean_path_length()
# 
#         X = L*Tbar
# 
#         n_orient = len(X)
# 
#         data = np.column_stack((X,np.zeros(n_orient)))
# 
#         bandwidth = estimate_bandwidth(data, quantile=quantile)
# 
#         if bandwidth > 0:
# 
#             clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(data)
# 
#             labels = clustering.labels_
#             n_cluster = len(set(labels))
# 
#             clusters = [np.argwhere(label == labels).flatten().tolist() for label in range(n_cluster)]
# 
#         else:
# 
#             clusters = [np.arange(n_orient)]
# 
#         return clusters

    def is_peak_integrated(self):

        return self.__is_peak_integrated() and self.__has_good_fit()

    def __is_peak_integrated(self):

        return not (self.__peak_num == 0 or self.__pk_norm is None)

    def __good_intensities(self, min_vol_fract=0.5):

        pk_vol_fract = np.array(self.__pk_vol_fract())

        indices = np.arange(len(pk_vol_fract)).tolist()

        return [ind for ind in indices if pk_vol_fract[ind] > min_vol_fract]

    def __has_good_fit(self):

        statistics = np.array([self.peak_fit, self.peak_bkg_ratio, self.peak_score, self.peak_fit2d, self.peak_bkg_ratio2d, self.peak_score2d])

        if statistics.all() is not None:

            good = True

            if self.peak_fit < 0.02 or self.peak_fit2d < 0.02 or self.peak_fit > 200 or self.peak_fit2d > 200:

                good = False

            if self.peak_bkg_ratio < 0.5:

                good = False

            if self.peak_score < 3 or self.peak_score2d < 3:

                good = False

            # powder line in profile
            if self.peak_bkg_ratio > 1 and self.peak_bkg_ratio2d < 1 and self.peak_bkg_ratio/self.peak_bkg_ratio2d > 10:

                good = False

            # powder line in projection
            if self.peak_bkg_ratio2d > 1 and self.peak_bkg_ratio < 1 and self.peak_bkg_ratio2d/self.peak_bkg_ratio > 10:

                good = False

        else:

            good = False

        return good

    # ---

    def get_merged_data_scale(self):

        indices = self.__good_intensities()

        return self.get_partial_merged_data_scale(indices)

    def get_merged_norm_scale(self):

        indices = self.__good_intensities()

        return self.get_partial_merged_norm_scale(indices)

    def __get_merged_peak_data_arrays(self):

        indices = self.__good_intensities()

        return self.__get_partial_merged_peak_data_arrays(indices)

    def __get_merged_peak_norm_arrays(self):

        indices = self.__good_intensities()

        return self.__get_partial_merged_peak_norm_arrays(indices)

    def __get_merged_background_data_arrays(self):

        indices = self.__good_intensities()

        return self.__get_partial_merged_background_data_arrays(indices)

    def __get_merged_background_norm_arrays(self):

        indices = self.__good_intensities()

        return self.__get_partial_merged_background_norm_arrays(indices)

    # ---

    def get_partial_merged_data_scale(self, indices):

        scale_data = self.get_data_scale()

        return np.array([scale_data[ind] for ind in indices])

    def get_partial_merged_norm_scale(self, indices):

        scale_norm = self.get_norm_scale()

        return np.array([scale_norm[ind] for ind in indices])

    def __get_partial_merged_peak_data_arrays(self, indices):

        pk_data = self.__get_peak_data_arrays()

        return [pk_data[ind] for ind in indices]

    def __get_partial_merged_peak_norm_arrays(self, indices):

        pk_norm = self.__get_peak_norm_arrays()

        return [pk_norm[ind] for ind in indices]

    def __get_partial_merged_background_data_arrays(self, indices):

        bkg_data = self.__get_background_data_arrays()

        return [bkg_data[ind] for ind in indices]

    def __get_partial_merged_background_norm_arrays(self, indices):

        bkg_norm = self.__get_background_norm_arrays()

        return [bkg_norm[ind] for ind in indices]

    # ---

    def __get_peak_data_arrays(self):

        return self.__pk_data

    def __get_peak_norm_arrays(self):

        return self.__pk_norm

    def __get_background_data_arrays(self):

        return self.__bkg_data

    def __get_background_norm_arrays(self):

        return self.__bkg_norm

    def __get_peak_bin_centers(self):

        return self.__pk_Q0, self.__pk_Q1, self.__pk_Q2

    def __get_background_bin_centers(self):

        return self.__bkg_Q0, self.__bkg_Q1, self.__bkg_Q2

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
        self.set_satellite_info([0,0,0], [0,0,0], [0,0,0], 0)
        
        chemical_formula = 'V'
        unit_cell_volume = 27.642
        z_parameter = 2

        SetSampleMaterial(InputWorkspace=self.nws,
                          ChemicalFormula=chemical_formula,
                          ZParameter=z_parameter,
                          UnitCellVolume=unit_cell_volume,
                          SampleMass=0)

    def __call_peak(self, h, k, l, m=0, n=0, p=0):

        key = (h,k,l,m,n,p)

        d_spacing = self.get_d(h,k,l,m,n,p)

        peak_key = (h,k,l,m,n,p) if m**2+n**2+p**2 > 0 else (h,k,l)

        #print('{} {:2.4f} (\u212B)'.format(peak_key, d_spacing))

        peaks = self.peak_dict.get(key)

        if peaks is not None:

            for peak in peaks:
                pprint.pprint(peak.dictionary())

    __call__ = __call_peak

    def get_d(self, h, k, l, m=0, n=0, p=0):

        ol = self.iws.sample().getOrientedLattice()

        H, K, L = self.get_hkl(h, k, l, m, n, p)

        d_spacing = ol.d(V3D(H,K,L))

        return d_spacing

    def get_hkl(self, h, k, l, m=0, n=0, p=0):

        ol = self.iws.sample().getOrientedLattice()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

        dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)

        return h+dh, k+dk, l+dl
        
    def set_UB(self, UB):

        self.UB = UB

    def get_UB(self):

        return self.UB

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

    def __set_material_info(self, pws, chemical_formula, z_parameter):

        sample_mass = self.mass

        volume = pws.sample().getOrientedLattice().volume()

        if chemical_formula is not None and z_parameter > 0 and sample_mass > 0:

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

        ol = self.pws.sample().getOrientedLattice()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

        for key in self.peak_dict.keys():

            peaks = self.peak_dict.get(key)

            h, k, l, m, n, p = key

            for peak in peaks:

                peak_num = peak.get_peak_number()
                intens = peak.get_merged_intensity()
                sig_intens = peak.get_merged_intensity_error()
                pk_vol_fract = peak.get_merged_peak_volume_fraction()

                R = peak.get_goniometers()[0]

                self.iws.run().getGoniometer().setR(R)

                dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)

                pk = self.iws.createPeakHKL(V3D(h+dh,k+dk,l+dl))
                pk.setGoniometerMatrix(R)
                pk.setIntHKL(V3D(h,k,l))
                pk.setIntMNP(V3D(m,n,p))
                pk.setIntensity(intens)
                pk.setSigmaIntensity(sig_intens)
                pk.setPeakNumber(peak_num)
                pk.setBinCount(pk_vol_fract)
                self.pws.addPeak(pk)

    def add_peaks(self, ws):

        # if not mtd.doesExist('pws'):
        #     print('pws does not exist')

        if mtd.doesExist(ws):

            pws = mtd[ws]

            ol = pws.sample().getOrientedLattice()

            UB = ol.getUB()

            mod_vec_1 = ol.getModVec(0)
            mod_vec_2 = ol.getModVec(1)
            mod_vec_3 = ol.getModVec(2)

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

                    bank = int(round(peak.getBinCount())) if bank == 'panel' else int(bank.strip('bank'))
                    ind = peak.getPeakNumber()

                    dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)            

                    Q = 2*np.pi*np.dot(UB, np.array([h+dh,k+dk,l+dl]))
                    #Q = peak.getQSampleFrame()

                    wl = peak.getWavelength()
                    R = peak.getGoniometerMatrix()

                    self.pws.run().getGoniometer().setR(R)
                    omega, chi, phi = self.pws.run().getGoniometer().getEulerAngles('YZY')

                    if self.peak_dict.get(key) is None:

                        peak_num = self.pws.getNumberPeaks()+1

                        new_peak = PeakInformation(self.scale_constant)
                        new_peak.set_peak_number(peak_num)

                        self.peak_dict[key] = [new_peak]

                        pk = self.pws.createPeakHKL(V3D(h+dh,k+dk,l+dl))
                        pk.setIntHKL(V3D(h,k,l))
                        pk.setIntMNP(V3D(m,n,p))
                        pk.setPeakNumber(peak_num)
                        pk.setGoniometerMatrix(R)

                        self.pws.addPeak(pk)

                    self.peak_dict[key][0].set_Q(Q)

                    Ql = np.dot(R, Q)

                    sign = -1 if config.get('Q.convention') == 'Inelastic' else 1

                    two_theta = 2*np.abs(np.arcsin(Ql[2]/np.linalg.norm(Ql)))
                    az_phi = np.arctan2(sign*Ql[1],sign*Ql[0])

                    self.peak_dict[key][0].add_information(run, bank, ind, row, col, wl, two_theta, az_phi,
                                                           phi, chi, omega, intens, sig_intens)

        # else:
        #     print('{} does not exist'.format(ws))

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

    def __dbscan_orientation(self, varphi, u, eps):

        n = len(varphi)

        if eps > 0:

            d = np.zeros((n,n))

            for i in range(n):
                t0, u0 = varphi[i], u[i]
                a0, b0, c0, d0 = np.cos(t0/2), u0[0]*np.sin(t0/2), u0[1]*np.sin(t0/2), u0[2]*np.sin(t0/2)
                for j in range(i+1,n):
                    t1, u1 = varphi[j], u[j]
                    a1, b1, c1, d1 = np.cos(t1/2), u1[0]*np.sin(t1/2), u1[1]*np.sin(t1/2), u1[2]*np.sin(t1/2)
                    v = 2*(a0*a1+b0*b1+c0*c1+d0*d1)**2-1
                    v = 1 if v > 1 else v
                    v = -1 if v < -1 else v
                    d[i,j] = d[j,i] = np.abs(np.rad2deg(np.arccos(v)))

            clustering = DBSCAN(eps=eps, min_samples=1, metric='precomputed').fit(d)

            labels = clustering.labels_
            n_clusters = len(set(labels))

            clusters = [np.argwhere(k == labels).flatten().tolist() for k in range(n_clusters)]

        else:

            clusters = [[i] for i in range(n)]

        return clusters

    def split_peaks(self, eps=5):

        ol = self.pws.sample().getOrientedLattice()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

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
                u = peak.get_rotation_axis()

                if len(u) > 1:

                    #clusters = self.__dbscan_orientation(varphi, u, eps)
                    clusters = self.__dbscan_1d(omega, eps)

                    if len(clusters) > 1:

                        for i, cluster in enumerate(clusters):

                            cluster = np.array(cluster)

                            new_peak = PeakInformation(self.scale_constant)

                            if i > 0:

                                dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)

                                peak_num = self.pws.getNumberPeaks()+1

                                pk = self.iws.createPeakHKL(V3D(h+dh,k+dk,l+dl))
                                pk.setGoniometerMatrix(R)
                                pk.setIntHKL(V3D(h,k,l))
                                pk.setIntMNP(V3D(m,n,p))
                                pk.setPeakNumber(peak_num)
                                pk.setGoniometerMatrix(R)
                                pk.setQSampleFrame(V3D(*Q))

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

            peak_dict[key] = peaks

        return peak_dict

    def integrated_result(self, key, Q, D, W, statistics, data_norm, pkg_bk, index=0):

        peaks = self.peak_dict[key]

        peak = peaks[index]
        peak.add_integration(Q, D, W, statistics, data_norm, pkg_bk)

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

        ol = self.pws.sample().getOrientedLattice()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

        dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)

        pk = self.iws.createPeakHKL(V3D(h+dh,k+dk,l+dl))
        pk.setGoniometerMatrix(R)
        pk.setIntHKL(V3D(h,k,l))
        pk.setIntMNP(V3D(m,n,p))
        pk.setPeakNumber(peak_num)
        pk.setIntensity(intens)
        pk.setSigmaIntensity(sig_intens)
        pk.setBinCount(pk_vol_fract)
        self.iws.addPeak(pk)

        peak_num = self.cws.getNumberPeaks()+1

        pk = self.cws.createPeakQSample(V3D(Qx,Qy,Qz))
        pk.setGoniometerMatrix(R)
        pk.setHKL(h+dh,k+dk,l+dl)
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

    def fitted_result(self, key, fit_1d, fit_2d, fit_prod, index=0):

        peaks = self.peak_dict[key]

        peak = peaks[index]
        peak.add_fit(fit_1d, fit_2d, fit_prod)

    def calibrated_result(self, key, run_num, Q, index=0):

        peaks = self.peak_dict[key]

        peak = peaks[index]

        h, k, l, m, n, p = key
        Qx, Qy, Qz = Q

        peak_num = self.cws.getNumberPeaks()+1

        runs = peak.get_run_numbers().tolist()
        R = peak.get_goniometers()[runs.index(run_num)]

        self.cws.run().getGoniometer().setR(R)

        ol = self.cws.sample().getOrientedLattice()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

        dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)     

        pk = self.cws.createPeakQSample(V3D(Qx,Qy,Qz))
        pk.setGoniometerMatrix(R)
        pk.setHKL(h+dh,k+dk,l+dl)
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

    def save_reflections(self, filename, min_sig_noise_ratio=3, min_vol_fract=0.5, adaptive_scale=True, normalize=True):

        # SortPeaksWorkspace(InputWorkspace=self.iws,
        #                    ColumnNameToSortBy='Intens',
        #                    SortAscending=False,
        #                    OutputWorkspace=self.iws)

        U = mtd['cws'].sample().getOrientedLattice().getU()

        ol = self.iws.sample().getOrientedLattice()

        max_order = ol.getMaxOrder()

        n_ind = 3 if max_order == 0 else 6

        hkl_fmt = n_ind*'{:4d}'+2*'{:8.2f}'+'{:4d}'+2*'{:8.5f}'+6*'{:9.5f}'+\
                  '{:6d}{:7d}{:7.4f}{:4d}{:9.5f}{:8.4f}'+2*'{:7.2f}'+'\n'

        with open(filename, 'w') as f:

            hkl_intensity = []

            pk_info_1 = []
            pk_info_2 = []

            run_bank_dict = {}

            j = 0

            key_set = set()

            for pn in range(self.iws.getNumberPeaks()):

                pk = self.iws.getPeak(pn)

                h, k, l = pk.getIntHKL()
                m, n, p = pk.getIntMNP()

                h, k, l, m, n, p = int(h), int(k), int(l), int(m), int(n), int(p)

                key = (h, k, l, m, n, p)

                key_set.add(key)

            keys = set(key_set)

            I_max = 1

            for key in keys:

                peaks = self.peak_dict.get(key)

                h, k, l, m, n, p = key

                for peak in peaks:

                    intens = peak.get_intensity(normalize=normalize)
                    sig_intens = peak.get_intensity_error(normalize=normalize)

                    pk_vol_fract = peak.get_peak_volume_fraction()

                    rows = peak.get_rows()
                    cols = peak.get_cols()

                    runs = peak.get_run_numbers()
                    banks = peak.get_bank_numbers()

                    Q = peak.get_Q()

                    lamda = peak.get_wavelengths()
                    two_theta = peak.get_scattering_angles()
                    az_phi = peak.get_azimuthal_angles()

                    T = peak.get_transmission_coefficient()
                    Tbar = peak.get_weighted_mean_path_length()

                    R = peak.get_goniometers()

                    for i in range(len(intens)):

                        if (intens[i] > 0 and sig_intens[i] > 0 and intens[i]/sig_intens[i] > min_sig_noise_ratio and pk_vol_fract[i] > min_vol_fract):

                            if intens[i] > I_max: 
                                I_max = intens[i].copy()

                            ki_norm = np.array([0, 0, -1])
                            kf_norm = np.array([np.cos(az_phi[i])*np.sin(two_theta[i]),
                                                np.sin(az_phi[i])*np.sin(two_theta[i]),
                                                np.cos(two_theta[i])])

                            RU = np.dot(R[i],U)

                            incident = np.dot(RU.T, ki_norm)
                            reflected = np.dot(RU.T, kf_norm)

                            d = 2*np.pi/np.linalg.norm(Q)

                            if max_order == 0:
                                hkl_intensity.append([h, k, l, intens[i], sig_intens[i]])
                            else:
                                hkl_intensity.append([h, k, l, m, n, p, intens[i], sig_intens[i]])

                            run = runs[i]
                            bank = banks[i]

                            pk_info_1.append([lamda[i], Tbar[i], incident[0], reflected[0], incident[1], reflected[1], incident[2], reflected[2], run])
                            pk_info_2.append([T[i], bank, two_theta[i], d, cols[i], rows[i]])

                            key = (run)

                            if run_bank_dict.get(key) is None:
                                run_bank_dict[key] = [j]
                            else:
                                ind = run_bank_dict[key]
                                ind.append(j)
                                run_bank_dict[key] = ind

                            j += 1

            if adaptive_scale:
                scale = 9999.99/I_max
            else:
                scale = 1

            pk_num = 0

            for j, (run) in enumerate(sorted(run_bank_dict.keys())):

                seq_num = j+1

                for i in run_bank_dict[(run)]:

                    if max_order == 0:
                        hkl_intensity[i][3] *= scale
                        hkl_intensity[i][4] *= scale
                    else:
                        hkl_intensity[i][6] *= scale
                        hkl_intensity[i][7] *= scale

                    f.write(hkl_fmt.format(*[*hkl_intensity[i], seq_num, *pk_info_1[i], pk_num, *pk_info_2[i]]))

                    pk_num += 1

            if max_order > 0:
                f.write(hkl_fmt.format(*[0]*25))
            else:
                f.write(hkl_fmt.format(*[0]*22))

    def save_calibration(self, filename, min_sig_noise_ratio=3, min_vol_fract=0.9):

        CloneWorkspace(self.cws, OutputWorkspace='cal')

        FilterPeaks(InputWorkspace='cal',
                    FilterVariable='Signal/Noise',
                    FilterValue=min_sig_noise_ratio,
                    Operator='>',
                    OutputWorkspace='cal')

        cal = mtd['cal']

        n = cal.getNumberPeaks()
        for pn in range(n-1,-1,-1):
            pk = cal.getPeak(pn)
            sig_noise = pk.getIntensityOverSigma()
            if sig_noise < 10:
                cal.removePeak(pn)

        n = cal.getNumberPeaks()

        if n > 20:

            ol = self.iws.sample().getOrientedLattice()

            mod_vec_1 = ol.getModVec(0)
            mod_vec_2 = ol.getModVec(1)
            mod_vec_3 = ol.getModVec(2)

            max_order = ol.getMaxOrder()

            Q, hkl = [], []

            for pn in range(n):
                pk = cal.getPeak(pn)
                hkl.append(pk.getHKL())
                Q.append(pk.getQSampleFrame())

            Q, hkl = np.array(Q), np.array(hkl)

            a, b, c, alpha, beta, gamma = ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma()

            if (np.allclose([a, b], c) and np.allclose([alpha, beta, gamma], 90)):
                fun = self.__cub
                x0 = (a, )
            elif (np.allclose([a, b], c) and np.allclose([alpha, beta], gamma)):
                fun = self.__rhom
                x0 = (a, np.deg2rad(alpha))
            elif (np.isclose(a, b) and np.allclose([alpha, beta, gamma], 90)):
                fun = self.__tet
                x0 = (a, c)
            elif (np.isclose(a, b) and np.allclose([alpha, beta], 90) and np.isclose(gamma, 120)):
                fun = self.__hex
                x0 = (a, c)
            elif (np.allclose([alpha, beta, gamma], 90)):
                fun = self.__ortho
                x0 = (a, b, c)
            elif np.allclose([alpha, beta], 90):
                fun = self.__mono1
                x0 = (a, b, c, np.deg2rad(gamma))
            elif np.allclose([alpha, gamma], 90):
                fun = self.__mono2
                x0 = (a, b, c, np.deg2rad(beta))
            else:
                fun = self.__tri
                x0 = (a, b, c, np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma))

            U = mtd['cws'].sample().getOrientedLattice().getU()

            omega = np.arccos((np.trace(U)-1)/2)

            val, vec = np.linalg.eig(U)

            ux, uy, uz = vec[:,np.argwhere(np.isclose(val, 1))[0][0]].real

            theta = np.arccos(uz)
            phi = np.arctan2(uy,ux)

            sol = scipy.optimize.least_squares(self.__res, x0=x0+(phi,theta,omega), args=(hkl,Q,fun))

            a, b, c, alpha, beta, gamma, phi, theta, omega = fun(sol.x)

            B = self.__B_matrix(a, b, c, alpha, beta, gamma)
            U = self.__U_matrix(phi, theta, omega)

            UB = np.dot(U,B)

            SetUB(Workspace='cal', UB=UB)

            self.__set_satellite_info(cal, mod_vec_1, mod_vec_2, mod_vec_3, max_order)
            SaveIsawUB(InputWorkspace='cal', Filename=filename.replace('nxs','mat'))

        SaveNexus(InputWorkspace='cal', Filename=filename)

        DeleteWorkspace('cal')

    def __U_matrix(self, phi, theta, omega):

        ux = np.cos(phi)*np.sin(theta)
        uy = np.sin(phi)*np.sin(theta)
        uz = np.cos(theta)

        U = np.array([[np.cos(omega)+ux**2*(1-np.cos(omega)), ux*uy*(1-np.cos(omega))-uz*np.sin(omega), ux*uz*(1-np.cos(omega))+uy*np.sin(omega)],
                      [uy*ux*(1-np.cos(omega))+uz*np.sin(omega), np.cos(omega)+uy**2*(1-np.cos(omega)), uy*uz*(1-np.cos(omega))-ux*np.sin(omega)],
                      [uz*ux*(1-np.cos(omega))-uy*np.sin(omega), uz*uy*(1-np.cos(omega))+ux*np.sin(omega), np.cos(omega)+uz**2*(1-np.cos(omega))]])

        return U

    def __B_matrix(self, a, b, c, alpha, beta, gamma):

        G = np.array([[a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
                      [b*a*np.cos(gamma), b**2, b*c*np.cos(alpha)],
                      [c*a*np.cos(beta), c*b*np.cos(alpha), c**2]])

        B = scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

        return B

    def __cub(self, x):

        a, phi, theta, omega = x

        return a, a, a, np.pi/2, np.pi/2, np.pi/2, phi, theta, omega

    def __rhom(self, x):

        a, alpha, phi, theta, omega = x

        return a, a, a, alpha, alpha, alpha, phi, theta, omega

    def __tet(self, x):

        a, c, phi, theta, omega = x

        return a, a, c, np.pi/2, np.pi/2, np.pi/2, phi, theta, omega

    def __hex(self, x):

        a, c, phi, theta, omega = x

        return a, a, c, np.pi/2, np.pi/2, 2*np.pi/3, phi, theta, omega

    def __ortho(self, x):

        a, b, c, phi, theta, omega = x

        return a, b, c, np.pi/2, np.pi/2, np.pi/2, phi, theta, omega

    def __mono1(self, x):

        a, b, c, gamma, phi, theta, omega = x

        return a, b, c, np.pi/2, np.pi/2, gamma, phi, theta, omega

    def __mono2(self, x):

        a, b, c, beta, phi, theta, omega = x

        return a, b, c, np.pi/2, beta, np.pi/2, phi, theta, omega

    def __tri(self, x):

        a, b, c, alpha, beta, gamma, phi, theta, omega = x

        return a, b, c, alpha, beta, gamma, phi, theta, omega

    def __res(self, x, hkl, Q, fun):

        a, b, c, alpha, beta, gamma, phi, theta, omega = fun(x)

        B = self.__B_matrix(a, b, c, alpha, beta, gamma)
        U = self.__U_matrix(phi, theta, omega)

        UB = np.dot(U,B)

        return (np.einsum('ij,lj->li', UB, hkl)*2*np.pi-Q).flatten()

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

        max_order = ol.getMaxOrder()

        mod_vector_1 = ol.getModVec(0)
        mod_vector_2 = ol.getModVec(1)
        mod_vector_3 = ol.getModVec(2)

        self.set_constants(a, b, c, alpha, beta, gamma)
        self.set_satellite_info(mod_vector_1, mod_vector_2, mod_vector_3, max_order)

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

        ol = self.pws.sample().getOrientedLattice()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

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

                    self.cws.run().getGoniometer().setR(R)
                    self.iws.run().getGoniometer().setR(R)

                    dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)            

                    pk = self.iws.createPeakHKL(V3D(h+dh,k+dk,l+dl))
                    pk.setGoniometerMatrix(R)
                    pk.setIntHKL(V3D(h,k,l))
                    pk.setIntMNP(V3D(m,n,p))
                    pk.setPeakNumber(peak_num)
                    pk.setIntensity(intens)
                    pk.setSigmaIntensity(sig_intens)
                    pk.setBinCount(pk_vol_fract)
                    self.iws.addPeak(pk)

                    pk = self.cws.createPeakQSample(V3D(Qx,Qy,Qz))
                    pk.setGoniometerMatrix(R)
                    pk.setHKL(h+dh,k+dk,l+dl)
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
        theta = np.loadtxt(filename, delimiter=',', max_rows=1, usecols=np.arange(1,92))

        f = scipy.interpolate.interp2d(muR, 2*theta, data.T, kind='cubic')

        return f

    def apply_spherical_correction(self, vanadium_mass=0):

        f = self.__spherical_aborption()

        mat = mtd['iws'].sample().getMaterial()

        sigma_a = mat.absorbXSection()
        sigma_s = mat.totalScatterXSection()

        m = self.mass # g
        M = mat.relativeMolecularMass()
        n = mat.numberDensityEffective # A^-3
        N = mat.totalAtoms 

        chemical_formula = '-'.join([atm.symbol+str(no) for atm, no in zip(*mat.chemicalFormula())])

        if chemical_formula == '':
            rho, V, R = 0, 0, 0

        else:
            rho = (n/N)/0.6022*M
            V = m/rho

            R = (0.75/np.pi*V)**(1/3)

        # print(chemical_formula)
        # print('absoption cross section: {} barn'.format(sigma_a))
        # print('scattering cross section: {} barn'.format(sigma_s))

        # print('linear scattering coefficient: {} 1/cm'.format(n*sigma_s))
        # print('linear absorption coefficient: {} 1/cm'.format(n*sigma_a))

        # print('mass: {} g'.format(m))
        # print('density: {} g/cm^3'.format(rho))

        # print('volume: {} cm^3'.format(V))
        # print('radius: {} cm'.format(R))

        # print('total atoms: {}'.format(N))
        # print('molar mass: {} g/mol'.format(M))
        # print('number density: {} 1/A^3'.format(n))
        
        van = mtd['nws'].sample().getMaterial()
        
        van_sigma_a = van.absorbXSection()
        van_sigma_s = van.totalScatterXSection()
        
        van_M = van.relativeMolecularMass()
        van_n = van.numberDensityEffective # A^-3
        van_N = van.totalAtoms 

        van_rho = (van_n/van_N)/0.6022*van_M
        van_V = vanadium_mass/van_rho

        van_R = (0.75/np.pi*van_V)**(1/3)

        van_mu_s = van_n*van_sigma_s
        van_mu_a = van_n*van_sigma_a

        # print('V')
        # print('absoption cross section: {} barn'.format(van_sigma_a))
        # print('scattering cross section: {} barn'.format(van_sigma_s))

        # print('linear scattering coefficient: {} 1/cm'.format(van_n*van_sigma_s))
        # print('linear absorption coefficient: {} 1/cm'.format(van_n*van_sigma_a))

        # print('mass: {} g'.format(vanadium_mass))
        # print('density: {} g/cm^3'.format(van_rho))

        # print('volume: {} cm^3'.format(van_V))
        # print('radius: {} cm'.format(van_R))

        # print('total atoms: {}'.format(van_N))
        # print('molar mass: {} g/mol'.format(van_M))
        # print('number density: {} 1/A^3'.format(van_n))

        for key in self.peak_dict.keys():

            peaks = self.peak_dict.get(key)

            for peak in peaks:

                wls = peak.get_wavelengths()
                two_thetas = peak.get_scattering_angles()

                Astar, Astar_van, T, Tbar = [], [], [], []

                for wl, two_theta in zip(wls, two_thetas):

                    mu = n*(sigma_s+(sigma_a/1.8)*wl)
                    muR = mu*R

                    # print('wavelength: {} ang'.format(wl))
                    # print('2theta: {} deg'.format(np.rad2deg(two_theta)))
                    # print('linear absorption coefficient: {} 1/cm'.format(mu))

                    correction = f(muR,np.rad2deg(two_theta))[0]
                    Astar.append(correction)

                    transmission = 1/correction
                    T.append(transmission)

                    length = R*f(muR, np.rad2deg(two_theta), dx=1)[0]/f(muR, np.rad2deg(two_theta))[0]
                    Tbar.append(length)

                    # --- 
                    
                    van_mu = van_n*(van_sigma_s+(van_sigma_a/1.8)*wl)
                    van_muR = van_mu*van_R

                    # print('linear absorption coefficient: {} 1/cm'.format(mu))

                    correction = f(van_muR,np.rad2deg(two_theta))[0]
                    Astar_van.append(correction)

                peak.set_data_scale(Astar)
                peak.set_norm_scale(Astar_van)
                
                peak.set_transmission_coefficient(T)
                peak.set_weighted_mean_path_length(Tbar)

        for pws in [self.iws, self.cws]:
            for pn in range(pws.getNumberPeaks()-1,-1,-1):
                pws.removePeak(pn)

        self.__repopulate_workspaces()

#     def apply_extinction_correction(self, constants):
# 
#         for key in self.peak_dict.keys():
# 
#             peaks = self.peak_dict.get(key)
# 
#             h, k, l, m, n, p = key
# 
#             for peak in peaks:
# 
#                 if type(constants) is list:
#                     c = constants[0]*abs(h*h)+constants[1]*abs(k*k)+constants[2]*abs(l*l)+\
#                         constants[3]*abs(h*k)+constants[4]*abs(k*l)+constants[5]*abs(l*h)
#                 else:
#                     c = constants
# 
#                 peak.set_ext_constant(c)
# 
#         for pws in [self.iws, self.cws]:
#             for pn in range(pws.getNumberPeaks()-1,-1,-1):
#                 pws.removePeak(pn)
# 
#         self.__repopulate_workspaces()

class GaussianFit3D:

    def __init__(self, x, y, e, mu, var):

        self.params = Parameters()

        self.params.add('A', value=y.max()-np.min(y), min=y.min()/1000, max=y.max())
        self.params.add('B', value=np.min(y), min=0, max=y.mean())

        self.params.add('mu0', value=mu[0], min=mu[0]-0.1, max=mu[0]+0.1)
        self.params.add('mu1', value=mu[1], min=mu[1]-0.1, max=mu[1]+0.1)
        self.params.add('mu2', value=mu[2], min=mu[2]-0.1, max=mu[2]+0.1)

        self.params.add('var0', value=var[0], min=0.25*var[0], max=4*var[0])
        self.params.add('var1', value=var[1], min=0.25*var[1], max=4*var[1])
        self.params.add('var2', value=var[2], min=0.25*var[2], max=4*var[2])

        self.params.add('phi', value=0, min=-np.pi, max=np.pi)
        self.params.add('theta', value=np.pi/2, min=0, max=np.pi)
        self.params.add('omega', value=0, min=-np.pi, max=np.pi)

        self.x = x
        self.y = y
        self.e = e

    def gaussian_3d(self, Q0, Q1, Q2, A, B, mu0, mu1, mu2, var0, var1, var2, phi=0, theta=0, omega=0):

        S = self.S_matrix(var0, var1, var2, phi, theta, omega)

        inv_S = np.linalg.inv(S)

        x0, x1, x2 = Q0-mu0, Q1-mu1, Q2-mu2

        return A*np.exp(-0.5*(inv_S[0,0]*x0**2+inv_S[1,1]*x1**2+inv_S[2,2]*x2**2\
                          +2*(inv_S[1,2]*x1*x2+inv_S[0,2]*x0*x2+inv_S[0,1]*x0*x1)))+B

    def S_matrix(self, var0, var1, var2, phi=0, theta=0, omega=0):

        V = self.V_matrix(var0, var1, var2)
        U = self.U_matrix(phi, theta, omega)

        S = np.dot(np.dot(U,V),U.T)

        return S

    def V_matrix(self, var0, var1, var2):

        V = np.diag([var0, var1, var2])

        return V

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

        mu0 = params['mu0']
        mu1 = params['mu1']
        mu2 = params['mu2']

        var0 = params['var0']
        var1 = params['var1']
        var2 = params['var2']

        phi = params['phi']
        theta = params['theta']
        omega = params['omega']

        args = Q0, Q1, Q2, A, B, mu0, mu1, mu2, var0, var1, var2, phi, theta, omega

        yfit = self.gaussian_3d(*args)

        yfit[np.isnan(yfit)] = 1e+15
        yfit[np.isinf(yfit)] = 1e+15

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

        var0 = result.params['var0'].value
        var1 = result.params['var1'].value
        var2 = result.params['var2'].value

        phi = result.params['phi'].value
        theta = result.params['theta'].value
        omega = result.params['omega'].value

        S = self.S_matrix(var0, var1, var2, phi, theta, omega)

        var = np.diag(S)
        sig = np.sqrt(var)

        sig_inv = np.diag(1/sig)

        rho = np.dot(np.dot(sig_inv, S), sig_inv)

        sig0, sig1, sig2 = sig[0], sig[1], sig[2]
        rho12, rho02, rho01 = rho[1,2], rho[0,2], rho[0,1]

        return A, B, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01

    def covariance_matrix(self, sig0, sig1, sig2, rho12, rho02, rho01):

        sig = np.diag([sig0, sig1, sig2])

        rho = np.array([[1, rho01, rho02],
                        [rho01, 1, rho12],
                        [rho02, rho12, 1]])

        S = np.dot(np.dot(sig, rho), sig)

        return S

    def integrated(self, A, sig0, sig1, sig2, rho12, rho02, rho01):

        S = self.covariance_matrix(sig0, sig1, sig2, rho12, rho02, rho01)

        return A*np.sqrt((2*np.pi)**3*np.linalg.det(S))

    def model(self, x, A, B, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01):

        S = self.covariance_matrix(sig0, sig1, sig2, rho12, rho02, rho01)

        inv_S = np.linalg.inv(S)

        x0, x1, x2 = x[0]-mu0, x[1]-mu1, x[2]-mu2

        return A*np.exp(-0.5*(inv_S[0,0]*x0**2+inv_S[1,1]*x1**2+inv_S[2,2]*x2**2\
                          +2*(inv_S[1,2]*x1*x2+inv_S[0,2]*x0*x2+inv_S[0,1]*x0*x1)))+B