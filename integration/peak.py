from mantid.simpleapi import CreateSingleValuedWorkspace, CreatePeaksWorkspace
from mantid.simpleapi import CloneWorkspace, DeleteWorkspace
from mantid.simpleapi import SortPeaksWorkspace, FilterPeaks
from mantid.simpleapi import SetUB, SaveIsawUB, CalculatePeaksHKL
from mantid.simpleapi import SetSampleMaterial, SaveNexus
from mantid.simpleapi import CreateSampleWorkspace, LoadCIF
from mantid.simpleapi import mtd

from mantid.kernel import V3D
from mantid.geometry import PointGroupFactory, SpaceGroupFactory
from mantid.geometry import CrystalStructure, ReflectionGenerator
from mantid import config

import matplotlib as mpl
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
import scipy.spatial

#np.seterr(divide='ignore', invalid='ignore')
#np.seterr(**settings)

np.seterr(divide='ignore', invalid='ignore')

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

    def __init__(self):

        plt.close('peak-envelope')

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

        #self.ax_p_proj.set_xlabel('Q\u2081 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_s_proj.set_xlabel('Q\u2081 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        self.ax_p_proj.set_ylabel('Q\u2082 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_s_proj.set_ylabel('Q\u2082 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

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
        pad_s = axes_size.Fraction(0.5, width_s)

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
        self.ax_Q2.set_xlabel('Q\u209A [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
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

        # self.ax_Qv.set_xlabel('Q\u209A [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        # self.ax_Qu.set_xlabel('Q\u209A [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        # self.ax_uv.set_xlabel('Q\u2081\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        self.ax_Qu.set_ylabel('Q\u2081\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_Qv.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_uv.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        self.ax_Qv_fit.set_xlabel('Q\u209A [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_Qu_fit.set_xlabel('Q\u209A [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_uv_fit.set_xlabel('Q\u2081\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        self.ax_Qu_fit.set_ylabel('Q\u2081\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_Qv_fit.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_uv_fit.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

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
            if np.isclose(lamda[0],lamda[1]):
                self.ax_Qu.set_title('\u03BB = {:.3f} \u212B'.format(lamda[0]))
            else:
                self.ax_Qu.set_title('\u03BB = {:.3f}-{:.3f} \u212B'.format(*lamda))

            if np.isclose(two_theta[0],two_theta[1]):
                self.ax_Qu_fit.set_title('2\u03B8 = {:.1f}\u00B0'.format(two_theta[0]))
            else:
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
        
        z[z <= 0] = np.nan
        z0[z0 <= 0] = np.nan

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
            
    def update_ellipse(self, mu, sigma, rho):

        mu_x, mu_y = mu
        sigma_x, sigma_y = sigma

        rx = np.sqrt(1+rho)
        ry = np.sqrt(1-rho)

        scale_x = 3*sigma_x
        scale_y = 3*sigma_y
        
        self.trans_elli_s.clear()

        self.elli_s.width = 2*rx

        self.elli_s.height = 2*ry

        self.trans_elli_s.rotate_deg(45).scale(scale_x,scale_y).translate(mu_x,mu_y)

        self.elli_s.set_transform(self.trans_elli_s+self.ax_s_proj.transData)

    def plot_integration(self, signal, u_extents, v_extents, Q_extents, centers, radii, scales):

        Qu = np.nansum(signal, axis=1)
        Qv = np.nansum(signal, axis=0)
        uv = np.nansum(signal, axis=2).T

        Qu[Qu <= 0] = np.nan
        Qv[Qv <= 0] = np.nan
        uv[uv <= 0] = np.nan

        if np.any(Qu > 0) and np.any(Qv > 0) and np.any(uv > 0) and np.diff(u_extents[0::2]) > 0 and np.diff(v_extents[0::2]) > 0 and np.diff(Q_extents[0::2]) > 0:

            if np.nanmax(Qu)-np.nanmin(Qu) > 0 and np.nanmax(Qv)-np.nanmin(Qv) > 0 and np.nanmax(uv)-np.nanmin(uv) > 0:

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

        Qu = np.nansum(signal, axis=1)
        Qv = np.nansum(signal, axis=0)
        uv = np.nansum(signal, axis=2).T

        Qu[Qu <= 0] = np.nan
        Qv[Qv <= 0] = np.nan
        uv[uv <= 0] = np.nan

        if np.any(Qu > 0) and np.any(Qv > 0) and np.any(uv > 0) and (np.array(I) > 0).all():

            if np.nanmax(Qu)-np.nanmin(Qu) > 0 and np.nanmax(Qv)-np.nanmin(Qv) > 0 and np.nanmax(uv)-np.nanmin(uv) > 0:

                self.im_Qu_fit.set_array(Qu)
                self.im_Qv_fit.set_array(Qv)
                self.im_uv_fit.set_array(uv)

            self.ax_uv.set_title('I = {:.2e} \u00B1 {:.2e} [arb. unit]'.format(I[0], sig[0]))
            self.ax_uv_fit.set_title('I = {:.2e} \u00B1 {:.2e} [arb. unit]'.format(I[1], sig[1]))

            # op = ' < ' if Dmax < Dn_crit else ' >= '
            # self.ax_Qv_fit.set_title('D\u2099 = {:.3}'.format(Dmax)+op+'{:.3}'.format(Dn_crit))

            self.ax_Qv_fit.set_title('\u03A7\u00B2 = {:.4f}'.format(chi_sq))

            if self.__show_plots: self.fig.show()

    def write_figure(self, figname):

        self.fig.savefig(figname, facecolor='white', transparent=False)

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

        self._chi_sq = 0

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

    def set_ext_scale(self, corr_scale):

        self.__ext_constants = np.array(corr_scale)

    def get_ext_scale(self):

        if self.__ext_constants is not None:
            return self.__ext_constants
        else:
            return np.ones_like(self.__norm_scale)

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

    def get_merged_background_volume_fraction(self):

        return self.__merge_bkg_vol_fract()

    def get_merged_intensity(self):

        return self.__merge_intensity()

    def get_merged_intensity_error(self):

        return self.__merge_intensity_error()

    def get_partial_merged_peak_volume_fraction(self, indices):

        return self.__partial_merge_pk_vol_fract(indices)

    def get_partial_merged_background_volume_fraction(self, indices):

        return self.__partial_merge_bkg_vol_fract(indices)

    def get_partial_merged_intensity(self, indices):

        return self.__partial_merge_intensity(indices)

    def get_partial_merged_intensity_error(self, indices):

        return self.__partial_merge_intensity_error(indices)

    def get_peak_volume_fraction(self):

        return self.__pk_vol_fract()

    def get_background_volume_fraction(self):

        return self.__bkg_vol_fract()

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
              'PeakVolumeFraction': self.__round(self.__pk_vol_fract(),2),
              'BackgroundVolumeFraction': self.__round(self.__bkg_vol_fract(),2),
              'ExtinctionConstants': self.__round(self.__ext_constants,3),
              'DataScale': self.__round(self.__data_scale,3),
              'NormalizationScale': self.__round(self.__norm_scale,2),
              'PeakScaleConstant': self.__round(self.get_peak_constant(),2),
              'Wavelength': self.__round(self.__wl,2),
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

    def add_fit(self, fit_1d, fit_2d, fit_prod, chi_sq):

        intens_fit, bkg_fit, sig = fit_prod

        self.__intens_fit = intens_fit
        self.__bkg_fit = bkg_fit

        self.__sig_fit = sig
        self.__chi_sq = chi_sq

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

    def __merge_bkg_vol_fract(self):

        return self.__partial_merge_bkg_vol_fract(self.__good_intensities())

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

            pk_data = np.nansum(data, axis=0)
            pk_norm = np.nansum(norm, axis=0)

            pk_vol_fract = np.sum(np.isfinite(pk_data/pk_norm))/len(data[0])

            return pk_vol_fract

    def __partial_merge_bkg_vol_fract(self, indices):

        if not self.__is_peak_integrated() or len(indices) == 0:

            return 0.0

        else:

            data = self.__get_partial_merged_background_data_arrays(indices)
            norm = self.__get_partial_merged_background_norm_arrays(indices)

            bkg_data = np.nansum(data, axis=0)
            bkg_norm = np.nansum(norm, axis=0)

            bkg_vol_fract = np.sum(np.isfinite(bkg_data/bkg_norm))/len(data[0])

            return bkg_vol_fract

    def __partial_merge_pk_bkg_ratio(self, indices):

        if not self.__is_peak_integrated() or len(indices) == 0:

            return 0.0

        else:

            data = self.__get_partial_merged_peak_data_arrays(indices)
            norm = self.__get_partial_merged_peak_norm_arrays(indices)

            bkg_data = self.__get_partial_merged_background_data_arrays(indices)
            bkg_norm = self.__get_partial_merged_background_norm_arrays(indices)

            pk_vol = np.sum(np.isfinite(np.nansum(data, axis=0)/np.nansum(norm, axis=0)))
            bkg_vol = np.sum(np.isfinite(np.nansum(bkg_data, axis=0)/np.nansum(bkg_norm, axis=0)))

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

            data_norm[~np.isfinite(data_norm)] = np.nan
            bkg_data_norm[~np.isfinite(bkg_data_norm)] = np.nan

            # Q1, Q2, Q3 = np.nanpercentile(bkg_data_norm, [25,50,75])
            # IQR = Q3-Q1
            # mask = (bkg_data_norm > Q3+1.5*IQR) | (bkg_data_norm < Q1-1.5*IQR)

            # bkg_data_norm[mask] = Q2

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

            data_norm[~np.isfinite(data_norm)] = np.nan
            bkg_data_norm[~np.isfinite(bkg_data_norm)] = np.nan

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

            pk_vol_fract = np.sum(np.isfinite(np.array(data)/np.array(norm)), axis=1)/len(data[0])

            return pk_vol_fract

    def __bkg_vol_fract(self):

        if not self.__is_peak_integrated():

            return np.array([])

        else:

            data = self.__get_background_data_arrays()
            norm = self.__get_background_norm_arrays()

            bkg_vol_fract = np.sum(np.isfinite(np.array(data)/np.array(norm)), axis=1)/len(data[0])

            return bkg_vol_fract

    def __pk_bkg_ratio(self):

        if not self.__is_peak_integrated():

            return np.array([])

        else:

            data = self.__get_peak_data_arrays()
            norm = self.__get_peak_norm_arrays()

            bkg_data = self.__get_background_data_arrays()
            bkg_norm = self.__get_background_norm_arrays()

            pk_vol = np.sum(np.isfinite([data,norm]).any(axis=0),axis=1)
            bkg_vol = np.sum(np.isfinite([bkg_data,bkg_norm]).any(axis=0),axis=1)

            return pk_vol/bkg_vol

    def __intensity(self):

        if not self.__is_peak_integrated():

            return np.array([])

        else:

            data = self.__get_peak_data_arrays()
            norm = self.__get_peak_norm_arrays()

            bkg_data = self.__get_background_data_arrays()
            bkg_norm = self.__get_background_norm_arrays()

            scale_data = self.get_data_scale()[:,np.newaxis]
            scale_norm = self.get_norm_scale()[:,np.newaxis]

            volume_ratio = self.__pk_bkg_ratio()

            constant = self.get_peak_constant()*np.prod(self.get_bin_size())

            data_scale, norm_scale = np.multiply(data, scale_data), np.multiply(norm, scale_norm)
            bkg_data_scale, bkg_norm_scale = np.multiply(bkg_data, scale_data), np.multiply(bkg_norm, scale_norm)

            intens = np.nansum(data_scale/norm_scale, axis=1)
            bkg_intens = np.nansum(bkg_data_scale/bkg_norm_scale, axis=1)

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

            scale_data = self.get_data_scale()[:,np.newaxis]
            scale_norm = self.get_norm_scale()[:,np.newaxis]

            volume_ratio = self.__pk_bkg_ratio()

            constant = self.get_peak_constant()*np.prod(self.get_bin_size())

            data_scale, norm_scale = np.multiply(data, scale_data), np.multiply(norm, scale_norm)
            bkg_data_scale, bkg_norm_scale = np.multiply(bkg_data, scale_data), np.multiply(bkg_norm, scale_norm)

            intens = np.nansum(data_scale/norm_scale**2, axis=1)
            bkg_intens = np.nansum(bkg_data_scale/bkg_norm_scale**2, axis=1)

            intensity = np.sqrt(intens+np.multiply(bkg_intens,volume_ratio**2))*constant

            return intensity

    def get_peak_clusters(self, quantile=0.0):

        # L = self.get_lorentz_factors(laue=laue)
        # Tbar = self.get_weighted_mean_path_length()

        # X = L*Tbar

        X = self.get_wavelengths()

        n_orient = len(X)

        data = np.column_stack((X,np.zeros(n_orient)))

        bandwidth = estimate_bandwidth(data, quantile=quantile)

        if bandwidth > 0:

            clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(data)

            labels = clustering.labels_
            n_cluster = len(set(labels))

            clusters = [np.argwhere(label == labels).flatten().tolist() for label in range(n_cluster)]

        else:

            clusters = [np.arange(n_orient)]

        return clusters

    def is_peak_integrated(self):

        return self.__is_peak_integrated() and self.__has_good_fit()

    def __is_peak_integrated(self):

        return not (self.__peak_num == 0 or self.__pk_norm is None)

    def __good_intensities(self, min_vol_fract=0.25):

        pk_vol_fract = np.array(self.__pk_vol_fract())

        indices = np.arange(len(pk_vol_fract)).tolist()

        return [ind for ind in indices if pk_vol_fract[ind] > min_vol_fract]

    def __has_good_fit(self):

        #                      chi-sq             std(pk)/med(bk)          I/sig
        statistics = np.array([self.__peak_fit,   self.__peak_bkg_ratio,   self.__peak_score,
                               self.__peak_fit2d, self.__peak_bkg_ratio2d, self.__peak_score2d])

        if statistics.all() is not None:

            good = True

            if self.__peak_bkg_ratio < 100 or self.__peak_bkg_ratio2d < 100:

                if self.__chi_sq < 0.02 or self.__chi_sq > 200:

                    good = False 

                if self.__peak_fit > 200 or self.__peak_fit2d > 200:

                    good = False

                if self.__peak_fit < 0.02 or self.__peak_fit2d < 0.02:

                    good = False

                if self.__peak_bkg_ratio < 0.5:

                    good = False

                if self.__peak_score < 3 or self.__peak_score2d < 3:

                    good = False

                # powder line in profile
                if self.__peak_score/self.__peak_score2d > 2:

                    good = False

                # powder line in projection
                if self.__peak_score/self.__peak_score2d > 2:

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

        return np.array([pk_data[ind] for ind in indices])

    def __get_partial_merged_peak_norm_arrays(self, indices):

        pk_norm = self.__get_peak_norm_arrays()

        return np.array([pk_norm[ind] for ind in indices])

    def __get_partial_merged_background_data_arrays(self, indices):

        bkg_data = self.__get_background_data_arrays()

        return np.array([bkg_data[ind] for ind in indices])

    def __get_partial_merged_background_norm_arrays(self, indices):

        bkg_norm = self.__get_background_norm_arrays()

        return np.array([bkg_norm[ind] for ind in indices])

    # ---

    def __get_peak_data_arrays(self):

        return np.array(self.__pk_data)

    def __get_peak_norm_arrays(self):

        return np.array(self.__pk_norm)

    def __get_background_data_arrays(self):

        return np.array(self.__bkg_data)

    def __get_background_norm_arrays(self):

        return np.array(self.__bkg_norm)

    def __get_peak_bin_centers(self):

        return self.__pk_Q0, self.__pk_Q1, self.__pk_Q2

    def __get_background_bin_centers(self):

        return self.__bkg_Q0, self.__bkg_Q1, self.__bkg_Q2

class PeakDictionary:

    def __init__(self, a=5, b=5, c=5, alpha=90, beta=90, gamma=90):

        self.peak_dict = { }

        self.scale_constant = 1e+4

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

    def load_cif(self, filename):

        sws = CreateSampleWorkspace()

        LoadCIF(sws, filename)

        cs = sws.sample().getCrystalStructure()

        self.hm = cs.getSpaceGroup().getHMSymbol()
        atoms = '; '.join(list(cs.getScatterers()))

        uc = cs.getUnitCell()
        a, b, c, alpha, beta, gamma = uc.a(), uc.b(), uc.c(), uc.alpha(), uc.beta(), uc.gamma()

        constants = '{} {} {} {} {} {}'.format(a,b,c,alpha,beta,gamma)

        self.cs = CrystalStructure(constants, self.hm, atoms)

        self.set_constants(a, b, c, alpha, beta, gamma)

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

        for pn in range(self.pws.getNumberPeaks()-1,-1,-1):
            self.pws.removePeak(pn)

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
                bkg_vol_fract = peak.get_merged_background_volume_fraction()

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
                pk.setAbsorptionWeightedPathLength(bkg_vol_fract)
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

                if len(u) > 1 and eps < 360:

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

            if peak_dict.get(key) is None:

                peaks = self.peak_dict.get(key)

                peak_dict[key] = peaks

        return peak_dict

    def construct_tree(self):

        keys = self.peak_dict.keys()

        Q_points = []

        for key in keys:

            peaks = self.peak_dict.get(key)

            Q_point = []

            for peak in peaks:

                Q0 = peak.get_Q()
                Q_point.append(Q0)

            if len(Q_point) > 0:

                Q_points.append(np.mean(Q_point, axis=0))

        Q_points = np.stack(Q_points)

        self.peak_tree = scipy.spatial.KDTree(Q_points)

    def query_planes(self, Q0, radius):

        indices = self.peak_tree.query_ball_point(Q0, radius)

        indices = [ind for ind in indices if not np.allclose(self.peak_tree.data[ind], Q0)]

        midpoints = [(Q0+self.peak_tree.data[ind])/2 for ind in indices]
        normals = [(self.peak_tree.data[ind]-Q0) for ind in indices]

        return midpoints, normals

    def integrated_result(self, key, Q, D, W, statistics, data_norm, pkg_bk, index=0):

        peaks = self.peak_dict[key]

        peak = peaks[index]
        peak.add_integration(Q, D, W, statistics, data_norm, pkg_bk)

    def partial_result(self, key, Q, A, peak_fit, peak_bkg_ratio, peak_score, index=0):

        peaks = self.peak_dict[key]

        peak = peaks[index]
        peak.add_partial_integration(Q, A, peak_fit, peak_bkg_ratio, peak_score)

    def fitted_result(self, key, fit_1d, fit_2d, fit_prod, chi_sq, index=0):

        peaks = self.peak_dict[key]

        peak = peaks[index]
        peak.add_fit(fit_1d, fit_2d, fit_prod, chi_sq)

        if peak.is_peak_integrated():

            h, k, l, m, n, p = key
            Qx, Qy, Qz = peak.get_Q()

            intens = peak.get_merged_intensity()
            sig_intens = peak.get_merged_intensity_error()
            pk_vol_fract = peak.get_merged_peak_volume_fraction()
            bkg_vol_fract = peak.get_merged_background_volume_fraction()

            run = peak.get_run_numbers().tolist()[0]
            R = peak.get_goniometers()[0]

            self.iws.run().getGoniometer().setR(R)
            self.cws.run().getGoniometer().setR(R)

            ol = self.pws.sample().getOrientedLattice()

            mod_vec_1 = ol.getModVec(0)
            mod_vec_2 = ol.getModVec(1)
            mod_vec_3 = ol.getModVec(2)

            dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)

            peak_num = self.iws.getNumberPeaks()+1

            pk = self.iws.createPeakHKL(V3D(h+dh,k+dk,l+dl))
            pk.setGoniometerMatrix(R)
            pk.setIntHKL(V3D(h,k,l))
            pk.setIntMNP(V3D(m,n,p))
            pk.setPeakNumber(peak_num)
            pk.setIntensity(intens)
            pk.setSigmaIntensity(sig_intens)
            pk.setBinCount(pk_vol_fract)
            pk.setAbsorptionWeightedPathLength(bkg_vol_fract)
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
            pk.setAbsorptionWeightedPathLength(bkg_vol_fract)
            pk.setRunNumber(run)
            self.cws.addPeak(pk)

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

    def save_hkl(self, filename, min_signal_noise_ratio=3,
                       min_pk_vol_fract=0.7, min_bkg_vol_fract=0.35,
                       adaptive_scale=True, scale=1, cross_terms=False):

        SortPeaksWorkspace(InputWorkspace=self.iws,
                           ColumnNameToSortBy='Intens',
                           SortAscending=False,
                           OutputWorkspace=self.iws)

        if adaptive_scale:
            if self.iws.getNumberPeaks() > 0:
                I = self.iws.getPeak(0).getIntensity()
                scale = 1000/I

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
                intens, sig_intens = pk.getIntensity()*scale, pk.getSigmaIntensity()*scale, 
                pk_vol_fract, bkg_vol_fract = pk.getBinCount(), pk.getAbsorptionWeightedPathLength()

                if (intens > 0 and sig_intens > 0 and intens/sig_intens > min_signal_noise_ratio and pk_vol_fract > min_pk_vol_fract and bkg_vol_fract > min_bkg_vol_fract):

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

        return scale

    def save_reflections(self, filename, min_sig_noise_ratio=3, min_pk_vol_fract=0.7, min_bkg_vol_fract=0.35, adaptive_scale=True, scale=1):

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

        hkl_intensity = []

        pk_info_1 = []
        pk_info_2 = []

        run_bank_dict = {}
        bank_run_dict = {}

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

                if peak.is_peak_integrated():

                    intens = peak.get_intensity()
                    sig_intens = peak.get_intensity_error()

                    pk_vol_fract = peak.get_peak_volume_fraction()
                    bkg_vol_fract = peak.get_background_volume_fraction()

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

                        if (intens[i] > 0 and sig_intens[i] > 0 and intens[i]/sig_intens[i] > min_sig_noise_ratio and pk_vol_fract[i] > min_pk_vol_fract and bkg_vol_fract[i] > min_bkg_vol_fract):

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

                            key = (bank)

                            if bank_run_dict.get(key) is None:
                                bank_run_dict[key] = [j]
                            else:
                                ind = bank_run_dict[key]
                                ind.append(j)
                                bank_run_dict[key] = ind

                            j += 1

        if adaptive_scale:
            scale = 1000/I_max

        filename, ext = os.path.splitext(filename)

        with open(filename+'_sn'+ext, 'w') as f:

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

        with open(filename+'_dn'+ext, 'w') as f:

            pk_num = 0

            for j, (bank) in enumerate(sorted(bank_run_dict.keys())):

                seq_num = j+1

                for i in bank_run_dict[(bank)]:

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

        return scale

    def save_calibration(self, filename, min_sig_noise_ratio=3):

        CloneWorkspace(self.cws, OutputWorkspace='cal')

        FilterPeaks(InputWorkspace='cal',
                    FilterVariable='Signal/Noise',
                    FilterValue=min_sig_noise_ratio,
                    Operator='>',
                    OutputWorkspace='cal')

        cal = mtd['cal']

        # n = cal.getNumberPeaks()
        # for pn in range(n-1,-1,-1):
        #     pk = cal.getPeak(pn)
        #     sig_noise = pk.getIntensityOverSigma()
        #     if sig_noise < min_sig_noise_ratio:
        #         cal.removePeak(pn)

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

        if n <= 20:

            if mtd.doesExist('cal'):
                DeleteWorkspace('cal')

    def recalculate_hkl(self, tol=0.15, fname=None):

        if mtd.doesExist('cal'):

            CloneWorkspace(InputWorkspace=self.cws, OutputWorkspace='out')
            CloneWorkspace(InputWorkspace=self.cws, OutputWorkspace='in')

            SetUB(Workspace='out', UB=mtd['cal'].sample().getOrientedLattice().getUB())

            CalculatePeaksHKL(PeaksWorkspace='out', OverWrite=True)

            for ws in [self.iws, self.cws, 'out', 'in']:

                SortPeaksWorkspace(InputWorkspace=ws,
                                   ColumnNameToSortBy='PeakNumber',
                                   SortAscending=True,
                                   OutputWorkspace=ws)

            ol = mtd['out'].sample().getOrientedLattice()

            if fname is not None:

                hdr_hkl = ['#      h', '       k', '       l', '    d-sp',
                           '       h', '       k', '       l', '    d-sp',
                           '      dh', '      dk', '      dl', '   d-sp%', 
                           'pk-vol%', ' bkg-vol%']
                fmt_khl = 12*'{:8}'+2*'{:9}'+'\n'

                hkl_file = open(fname, 'w')
                hkl_file.write(fmt_khl.format(*hdr_hkl))

                fmt_khl = 3*'{:8.3f}'+'{:8.4f}'+3*'{:8.3f}'+'{:8.4f}'+3*'{:8.3f}'+'{:8.4f}'+2*'{:9.2f}'+'\n'

            lines_hkl = []
            for pn in range(mtd['out'].getNumberPeaks()-1,-1,-1):
                ipk, opk = mtd['in'].getPeak(pn), mtd['out'].getPeak(pn)
                dHKL = np.abs(ipk.getHKL()-opk.getHKL())

                HKL = ipk.getHKL()
                D = ol.d(V3D(*HKL))

                hkl = opk.getHKL()
                Q = opk.getQSampleFrame()
                d = opk.getDSpacing()
                pk_vol_perc = np.round(100*opk.getBinCount(),2)
                bkg_vol_perc = np.round(100*opk.getAbsorptionWeightedPathLength(),2)
                
                dDp = np.abs(d-D)/D*100

                line_hkl = [*HKL, D, *hkl, d, *dHKL, dDp, pk_vol_perc, bkg_vol_perc]
                lines_hkl.append(line_hkl)

                if np.any(dHKL > tol):
                    self.iws.removePeak(pn)
                    self.cws.removePeak(pn)

            if fname is not None:

                sort = np.argsort([line_hkl[3] for line_hkl in lines_hkl])[::-1]

                for i in sort:
                    hkl_file.write(fmt_khl.format(*lines_hkl[i]))

                hkl_file.close()

            #SaveNexus(InputWorkspace='out', Filename='/tmp/out.nxs')
            #SaveNexus(InputWorkspace='in', Filename='/tmp/in.nxs')
            #SaveNexus(InputWorkspace='cws', Filename='/tmp/cws.nxs')
            #SaveNexus(InputWorkspace='iws', Filename='/tmp/iws.nxs')

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

        a, *params = x

        return (a, a, a, np.pi/2, np.pi/2, np.pi/2, *params)

    def __rhom(self, x):

        a, alpha, *params = x

        return (a, a, a, alpha, alpha, alpha, *params)

    def __tet(self, x):

        a, c, *params = x

        return (a, a, c, np.pi/2, np.pi/2, np.pi/2, *params)

    def __hex(self, x):

        a, c, *params = x

        return (a, a, c, np.pi/2, np.pi/2, 2*np.pi/3, *params)

    def __ortho(self, x):

        a, b, c, *params = x

        return (a, b, c, np.pi/2, np.pi/2, np.pi/2, *params)

    def __mono1(self, x):

        a, b, c, gamma, *params = x

        return (a, b, c, np.pi/2, np.pi/2, gamma, *params)

    def __mono2(self, x):

        a, b, c, beta, *params = x

        return (a, b, c, np.pi/2, beta, np.pi/2, *params)

    def __tri(self, x):

        a, b, c, alpha, beta, gamma, *params = x

        return (a, b, c, alpha, beta, gamma, *params)

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

        self.clear_peaks()

        self.repopulate_workspaces()

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

    def repopulate_workspaces(self):

        self.__repopulate_workspaces()

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
                bkg_vol_fract = peak.get_merged_background_volume_fraction()

                run = peak.get_run_numbers().tolist()[0]
                R = peak.get_goniometers()[0]

                h, k, l, m, n, p = key
                Qx, Qy, Qz = peak.get_Q()

                if peak.is_peak_integrated():

                    peak_num = self.iws.getNumberPeaks()+1

                    self.cws.run().getGoniometer().setR(R)
                    self.iws.run().getGoniometer().setR(R)

                    dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)            

                    pk = self.iws.createPeakHKL(V3D(h+dh,k+dk,l+dl))
                    pk.setGoniometerMatrix(R)
                    pk.setIntHKL(V3D(h,k,l))
                    pk.setIntMNP(V3D(m,n,p))
                    # pk.setQSampleFrame(V3D(Qx,Qy,Qz))
                    pk.setPeakNumber(peak_num)
                    pk.setIntensity(intens)
                    pk.setSigmaIntensity(sig_intens)
                    pk.setBinCount(pk_vol_fract)
                    pk.setAbsorptionWeightedPathLength(bkg_vol_fract)
                    pk.setRunNumber(run)
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
                    pk.setAbsorptionWeightedPathLength(bkg_vol_fract)
                    pk.setRunNumber(run)
                    self.cws.addPeak(pk)

        SortPeaksWorkspace(self.iws, ColumnNameToSortBy='DSpacing', SortAscending=False, OutputWorkspace=self.iws)
        SortPeaksWorkspace(self.cws, ColumnNameToSortBy='DSpacing', SortAscending=False, OutputWorkspace=self.cws)

    def clear_peaks(self):

        for pws in [self.iws, self.cws]:
            for pn in range(pws.getNumberPeaks()-1,-1,-1):
                pws.removePeak(pn)
                
    def __equivalent_sphere(self):

        mat = mtd['iws'].sample().getMaterial()

        #V = mtd['iws'].sample().getOrientedLattice().volume()

        m = self.mass # g
        M = mat.relativeMolecularMass()
        n = mat.numberDensityEffective # A^-3
        N = mat.totalAtoms 

        if N > 0:
            rho = (n/N)/0.6022*M
            V = m/rho
            R = (0.75/np.pi*m/rho)**(1/3) # cm
        else:
            rho, V, R = 0, 0, 0

        return m, M, n, N, rho, V, R 

    def __spherical_absorption(self):

        filename = os.path.join(os.path.dirname(__file__), 'absorption_sphere.csv')

        data = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=np.arange(1,92))

        muR = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=(0))
        theta = np.loadtxt(filename, delimiter=',', max_rows=1, usecols=np.arange(1,92))

        f = scipy.interpolate.interp2d(muR, 2*theta, data.T, kind='cubic')

        return f

    def apply_spherical_correction(self, vanadium_mass=0, fname=None):

        if fname is not None:
            absorption_file = open(fname, 'w')

        f = self.__spherical_absorption()

        mat = mtd['iws'].sample().getMaterial()

        sigma_a = mat.absorbXSection()
        sigma_s = mat.totalScatterXSection()

        chemical_formula = '-'.join([atm.symbol+str(no) for atm, no in zip(*mat.chemicalFormula())])

        m, M, n, N, rho, V, R = self.__equivalent_sphere()

        if fname is not None:

            absorption_file.write('{}\n'.format(chemical_formula))
            absorption_file.write('absoption cross section: {:.4f} barn\n'.format(sigma_a))
            absorption_file.write('scattering cross section: {:.4f} barn\n'.format(sigma_s))

            absorption_file.write('linear scattering coefficient: {:.4f} 1/cm\n'.format(n*sigma_s))
            absorption_file.write('linear absorption coefficient: {:.4f} 1/cm\n'.format(n*sigma_a))

            absorption_file.write('mass: {:.4f} g\n'.format(m))
            absorption_file.write('density: {:.4f} g/cm^3\n'.format(rho))

            absorption_file.write('volume: {:.4f} cm^3\n'.format(V))
            absorption_file.write('radius: {:.4f} cm\n'.format(R))

            absorption_file.write('total atoms: {:.4f}\n'.format(N))
            absorption_file.write('molar mass: {:.4f} g/mol\n'.format(M))
            absorption_file.write('number density: {:.4f} 1/A^3\n'.format(n))
        
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

        if fname is not None:

            absorption_file.write('\nV\n')
            absorption_file.write('absoption cross section: {:.4f} barn\n'.format(van_sigma_a))
            absorption_file.write('scattering cross section: {:.4f} barn\n'.format(van_sigma_s))

            absorption_file.write('linear scattering coefficient: {:.4f} 1/cm\n'.format(van_n*van_sigma_s))
            absorption_file.write('linear absorption coefficient: {:.4f} 1/cm\n'.format(van_n*van_sigma_a))

            absorption_file.write('mass: {:.4f} g\n'.format(vanadium_mass))
            absorption_file.write('density: {:.4f} g/cm^3\n'.format(van_rho))

            absorption_file.write('volume: {:.4f} cm^3\n'.format(van_V))
            absorption_file.write('radius: {:.4f} cm\n'.format(van_R))

            absorption_file.write('total atoms: {:.4f}\n'.format(van_N))
            absorption_file.write('molar mass: {:.4f} g/mol\n'.format(van_M))
            absorption_file.write('number density: {:.4f} 1/A^3\n'.format(van_n))

            absorption_file.close()

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

        self.clear_peaks()
        self.repopulate_workspaces()

    def __spherical_extinction(self, model):

        if 'gaussian' in model:
            fname = 'secondary_extinction_gaussian_sphere.csv'
        elif 'lorentzian' in model:
            fname = 'secondary_extinction_lorentzian_sphere.csv'
        else:
            fname = 'primary_extinction_sphere.csv'

        filename = os.path.join(os.path.dirname(__file__), fname)

        data = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=np.arange(91))
        theta = np.loadtxt(filename, delimiter=',', max_rows=1)

        f1 = scipy.interpolate.interp1d(2*np.deg2rad(theta), data[0], kind='cubic')
        f2 = scipy.interpolate.interp1d(2*np.deg2rad(theta), data[1], kind='cubic')

        return f1, f2
        
    def apply_extinction_correction(self, r, g, s, model='secondary, gaussian'):

        generator = ReflectionGenerator(self.cs)

        f1, f2 = self.__spherical_extinction(model)

        m, M, n, N, rho, V, R = self.__equivalent_sphere()

        V = mtd['iws'].sample().getOrientedLattice().volume() # Ang^3
        R *= 1e+8 # Ang

        for key in self.peak_dict.keys():

            peaks = self.peak_dict.get(key)

            h, k, l, m, n, p = key

            for peak in peaks:

                if peak.get_merged_intensity() > 0:

                    scale = peak.get_data_scale()
                    intens = peak.get_intensity()
                    two_theta = peak.get_scattering_angles()
                    lamda = peak.get_wavelengths()
                    Tbar = peak.get_weighted_mean_path_length()

                    F2 = generator.getFsSquared([V3D(h,k,l)])[0]

                    c1, c2 = f1(two_theta), f2(two_theta)

                    x = self.__extinction_x(r, g, F2, two_theta, lamda, Tbar, R, V, model)
                    y = self.__extinction_correction(r, g, F2, c1, c2, two_theta, lamda, Tbar, R, V, model)

                    y[~np.isfinite(y)] = 1
                    y[x > 30] = np.inf

                    scale /= y

                    peak.set_data_scale(scale)

        self.clear_peaks()
        self.repopulate_workspaces()

    # ---

    def peak_families(self, top_fraction=0.2, min_pk_vol_fract=0.7, min_bkg_vol_fract=0.5):

        generator = ReflectionGenerator(self.cs)

        sg = SpaceGroupFactory.createSpaceGroup(self.hm)
        pg = PointGroupFactory.createPointGroupFromSpaceGroup(sg)

        I, E = [], []
        two_theta, lamda, Tbar = [], [], []
        hkl, F2, d_spacing = [], [], []
        band = []

        keys = list(self.peak_dict.keys())

        for key in keys:

            i, e = [], []
            tt, wl, wpl = [], [], []

            h, k, l, m, n, p = key

            equivalents = pg.getEquivalents(V3D(h,k,l))[::-1]

            for equivalent in equivalents:

                h, k, l = equivalent

                h, k, l = int(h), int(k), int(l)

                key = h, k, l, m, n, p

                if key in keys:

                    h, k, l, m, n, p = key

                    keys.remove(key)

                    peaks = self.peak_dict.get(key)

                    d = self.get_d(h,k,l,m,n,p)

                    for peak in peaks:

                        intensities = peak.get_intensity()
                        sig_intensities = peak.get_intensity_error()

                        two_thetas = peak.get_scattering_angles()
                        lamdas = peak.get_wavelengths()
                        wpls = peak.get_weighted_mean_path_length()

                        pk_vol_fracts = peak.get_peak_volume_fraction()
                        bkg_vol_fracts = peak.get_background_volume_fraction()

                        for j, (intens, sig_intens) in enumerate(zip(intensities, sig_intensities)):

                            if pk_vol_fracts[j] > min_pk_vol_fract and bkg_vol_fracts[j] > min_bkg_vol_fract:

                                if intens/sig_intens > 3:

                                    i.append(intens)
                                    e.append(sig_intens)

                                    tt.append(two_thetas[j])
                                    wl.append(lamdas[j])
                                    wpl.append(wpls[j])

            if len(wl) > 3:

                b = np.max(wl)-np.min(wl)

                if b > 0.5:

                    sf = generator.getFsSquared([V3D(h,k,l)])[0]
                    band.append(b*sf)

                    I.append(np.array(i))
                    E.append(np.array(e))

                    two_theta.append(np.array(tt))
                    lamda.append(np.array(wl))
                    Tbar.append(np.array(wpl))

                    hkl.append([h,k,l])
                    F2.append(sf)
                    d_spacing.append(d)

        no_fam = len(F2)

        min_no = int(no_fam*top_fraction)

        sort = np.argsort(band)[::-1][:min_no]

        I = [I[i] for i in sort]
        E = [E[i] for i in sort]

        two_theta = [two_theta[i] for i in sort]
        lamda = [lamda[i] for i in sort]
        Tbar = [Tbar[i] for i in sort]

        hkl = [hkl[i] for i in sort]
        F2 = [F2[i] for i in sort]
        d_spacing = [d_spacing[i] for i in sort]

        return I, E, two_theta, lamda, Tbar, hkl, F2, d_spacing

    def __extinction_factor(self, r, g, two_theta, lamda, Tbar, R, V, model):

        a = 1e-4 # Ang

        rho = r/lamda

        if model == 'primary':

            xi = 1.5*a**2/V**2*lamda**4*rho**2

        elif model == 'secondary, gaussian':

            xi = a**2/V**2*lamda**3*rho/np.sqrt(1+rho**2*np.sin(two_theta)**2/g**2)*Tbar*R

        elif model == 'secondary, lorentzian':

            xi = a**2/V**2*lamda**3*rho/(1+rho*np.sin(two_theta)/g)*Tbar*R

        elif 'type II' in model:

            xi = a**2/V**2*lamda**3*rho*Tbar*R

        elif 'type I' in model:

            xi = a**2/V**2*lamda**3*g/np.sin(two_theta)*Tbar*R

        return xi

    def __extinction_x(self, r, g, F2, two_theta, lamda, Tbar, R, V, model):

        xi = self.__extinction_factor(r, g, two_theta, lamda, Tbar, R, V, model)

        return xi*F2

    def __extinction_correction(self, r, g, F2, c1, c2, two_theta, lamda, Tbar, R, V, model):

        x = self.__extinction_x(r, g, F2, two_theta, lamda, Tbar, R, V, model)

        return 1/np.sqrt(1+c1*x+c2*x**2)

    def __extinction_model(self, r, g, s, F2, c1, c2, two_theta, lamda, Tbar, R, V, model):

        y = self.__extinction_correction(r, g, F2, c1, c2, two_theta, lamda, Tbar, R, V, model)

        return s*F2*y

    def __extinction_residual(self, params, I, E, HKL, two_theta, lamda, Tbar, R, V, f1, f2, model):

        uc = self.cs.getUnitCell()
        a, b, c, alpha, beta, gamma = uc.a(), uc.b(), uc.c(), uc.alpha(), uc.beta(), uc.gamma()

        constants = '{} {} {} {} {} {}'.format(a,b,c,alpha,beta,gamma)

        scatterers = self.cs.getScatterers()

        U = str(params['U'].value)

        atoms = []
        for j, scatterer in enumerate(scatterers):
            elm, x, y, z, occ, _ = scatterer.split(' ')
            atoms.append(' '.join([elm,x,y,z,occ,U]))

        atoms = '; '.join(atoms)

        cs = CrystalStructure(constants, self.hm, atoms)
        generator = ReflectionGenerator(cs)

        r, g, s = params['r'], params['g'], params['s']

        diff_I = np.array([])
        err_I = np.array([])

        for j, (i, e, hkl, tt, wl, wpl) in enumerate(zip(I, E, HKL, two_theta, lamda, Tbar)):

            c1, c2 = f1(tt), f2(tt)

            h, k, l = hkl

            sf = generator.getFsSquared([V3D(h,k,l)])[0]

            intens = self.__extinction_model(r, g, s, sf, c1, c2, tt, wl, wpl, R, V, model)
            intens[~np.isfinite(intens)] = 3*i[~np.isfinite(intens)]

            dI = (i-intens)

            diff_I = np.concatenate((diff_I,dI))
            err_I = np.concatenate((err_I,e))

        return diff_I/err_I

    def extinction_curves(self, r, g, s, model):

        f1, f2 = self.__spherical_extinction(model)

        m, M, n, N, rho, V, R = self.__equivalent_sphere()

        V = mtd['iws'].sample().getOrientedLattice().volume() # Ang^3
        R *= 1e+8 # Ang

        I, E, two_theta, lamda, Tbar, hkl, F2, d_spacing = self.peak_families()

        X, Y = [], []

        for j, (i, e, sf, tt, wl, wpl) in enumerate(zip(I, E, F2, two_theta, lamda, Tbar)):

            c1, c2 = f1(tt), f2(tt)

            x = self.__extinction_x(r, g, sf, tt, wl, wpl, R, V, model)
            y = self.__extinction_model(r, g, s, sf, c1, c2, tt, wl, wpl, R, V, model)

            mask = x > 30

            if np.any(mask):

                x = x[~mask]
                y = y[~mask]

                I[j] = i[~mask]
                E[j] = e[~mask]

                two_theta[j] = tt[~mask]
                lamda[j] = wl[~mask]
                Tbar[j] = wpl[~mask]

            X.append(x)
            Y.append(y)

        indices = np.argsort(d_spacing)[::-1]

        X = [X[i] for i in indices]
        Y = [Y[i] for i in indices]
        I = [I[i] for i in indices]
        E = [E[i] for i in indices]

        hkl = [hkl[i] for i in indices]
        d_spacing = [d_spacing[i] for i in indices]

        return X, Y, I, E, hkl, d_spacing

    def fit_extinction(self, model):

        f1, f2 = self.__spherical_extinction(model)

        m, M, n, N, rho, V, R = self.__equivalent_sphere()

        V = mtd['iws'].sample().getOrientedLattice().volume() # Ang^3
        R *= 1e+8 # Ang

        I, E, two_theta, lamda, Tbar, hkl, F2, d_spacing = self.peak_families()

        params = Parameters()

        params.add('s', value=1, min=0)

        if 'type II' in model:
            params.add('r', value=1000, min=0, max=1e7)
            params.add('g', value=0, vary=False)
        elif 'type I' in model:
            params.add('r', value=0, vary=False)
            params.add('g', value=500, min=0)
        elif 'secondary' in model:
            params.add('r', value=1000, min=0, max=1e7)
            params.add('g', value=500, min=0)
        else:
            params.add('r', value=1000, min=0, max=1e7)
            params.add('g', value=0, vary=False)

        params.add('U', min=0, max=1, value=0.001)

        out = Minimizer(self.__extinction_residual, params, fcn_args=(I, E, hkl, two_theta, lamda, Tbar, R, V, f1, f2, model))
        result = out.minimize(method='least_squares')

        r = result.params['r'].value
        g = result.params['g'].value
        s = result.params['s'].value
        U = result.params['U'].value

        for j, (i, e, sf, tt, wl, wpl) in enumerate(zip(I, E, F2, two_theta, lamda, Tbar)):

            c1, c2 = f1(tt), f2(tt)

            x = self.__extinction_x(r, g, sf, tt, wl, wpl, R, V, model)

            mask = x > 30

            if np.any(mask):

                I[j] = i[~mask]
                E[j] = e[~mask]

                two_theta[j] = tt[~mask]
                lamda[j] = wl[~mask]
                Tbar[j] = wpl[~mask]

        params['r'].set(value=r)
        params['g'].set(value=g)
        params['s'].set(value=s)
        params['U'].set(value=U)

        out = Minimizer(self.__extinction_residual, params, fcn_args=(I, E, hkl, two_theta, lamda, Tbar, R, V, f1, f2, model))
        result = out.minimize(method='least_squares')

        report_fit(result)

        return result.params['r'].value, result.params['g'].value, result.params['s'].value, result.params['U'].value, result.redchi

class GaussianFit3D:

    def __init__(self, x, y, e, mu, sigma):

        params = Parameters()

        y_min, y_max = np.min(y), np.max(y)

        y_range = y_max-y_min

        params.add('A', value=y_range, min=0.001*y_range, max=1000*y_range)
        params.add('B', value=y_min, min=y_min-10*y_range, max=y_max+10*y_range)

        params.add('mu0', value=mu[0], min=mu[0]-0.075, max=mu[0]+0.075)
        params.add('mu1', value=mu[1], min=mu[1]-0.075, max=mu[1]+0.075)
        params.add('mu2', value=mu[2], min=mu[2]-0.075, max=mu[2]+0.075)

        params.add('sigma0', value=sigma[0], min=0.5*sigma[0], max=2*sigma[0])
        params.add('sigma1', value=sigma[1], min=0.5*sigma[1], max=2*sigma[1])
        params.add('sigma2', value=sigma[2], min=0.5*sigma[2], max=2*sigma[2])

        params.add('phi', value=0, min=-np.pi, max=np.pi)
        params.add('theta', value=np.pi/2, min=0, max=np.pi)
        params.add('omega', value=0, min=-np.pi, max=np.pi)

        self.params = params

        self.x = x
        self.y = y
        self.e = e

    def gaussian_3d(self, Q0, Q1, Q2, A, mu0, mu1, mu2, sigma0, sigma1, sigma2, phi=0, theta=0, omega=0):

        S = self.S_matrix(sigma0, sigma1, sigma2, phi, theta, omega)

        inv_S = np.linalg.inv(S)

        x0, x1, x2 = Q0-mu0, Q1-mu1, Q2-mu2

        return A*np.exp(-0.5*(inv_S[0,0]*x0**2+inv_S[1,1]*x1**2+inv_S[2,2]*x2**2\
                          +2*(inv_S[1,2]*x1*x2+inv_S[0,2]*x0*x2+inv_S[0,1]*x0*x1)))

    def gaussian(self, Q0, Q1, Q2, A, mu0, mu1, mu2, sigma0, sigma1, sigma2, phi=0, theta=0, omega=0):

        x0, x1, x2 = Q0-mu0, Q1-mu1, Q2-mu2

        U = self.U_matrix(phi, theta, omega)

        a = 0.5*(U[0,0]**2/sigma0**2+U[0,1]**2/sigma1**2+U[0,2]**2/sigma2**2)
        b = 0.5*(U[1,0]**2/sigma0**2+U[1,1]**2/sigma1**2+U[1,2]**2/sigma2**2)
        c = 0.5*(U[2,0]**2/sigma0**2+U[2,1]**2/sigma1**2+U[2,2]**2/sigma2**2)

        d = U[1,0]*U[2,0]/sigma0**2+U[1,1]*U[2,1]/sigma1**2+U[1,2]*U[2,2]/sigma2**2
        e = U[2,0]*U[0,0]/sigma0**2+U[2,1]*U[0,1]/sigma1**2+U[2,2]*U[0,2]/sigma2**2
        f = U[0,0]*U[1,0]/sigma0**2+U[0,1]*U[1,1]/sigma1**2+U[0,2]*U[1,2]/sigma2**2

        return A*np.exp(-(a*x0**2+b*x1**2+c*x2**2+d*x1*x2+e*x0*x2+f*x0*x1))

    def S_matrix(self, sigma0, sigma1, sigma2, phi=0, theta=0, omega=0):

        V = self.V_matrix(sigma0, sigma1, sigma2)
        U = self.U_matrix(phi, theta, omega)

        S = np.dot(np.dot(U,V),U.T)

        return S

    def V_matrix(self, sigma0, sigma1, sigma2):

        V = np.diag([sigma0**2, sigma1**2, sigma2**2])

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

        sigma0 = params['sigma0']
        sigma1 = params['sigma1']
        sigma2 = params['sigma2']

        phi = params['phi']
        theta = params['theta']
        omega = params['omega']

        args = Q0, Q1, Q2, A, B, mu0, mu1, mu2, sigma0, sigma1, sigma2, phi, theta, omega

        yfit = self.func(*args)

        yfit[~np.isfinite(yfit)] = 1e+15
        yfit[~np.isfinite(yfit)] = 1e+15

        return (y-yfit)/e

    def func(self, Q0, Q1, Q2, A, B, mu0, mu1, mu2, sigma0, sigma1, sigma2, phi, theta, omega):

        args = Q0, Q1, Q2, A, mu0, mu1, mu2, sigma0, sigma1, sigma2, phi, theta, omega

        return self.gaussian(*args)+B

    def gradient(self, params, x, y, e):

        Q0, Q1, Q2 = x

        A = params['A']
        B = params['B']

        mu0 = params['mu0']
        mu1 = params['mu1']
        mu2 = params['mu2']

        sigma0 = params['sigma0']
        sigma1 = params['sigma1']
        sigma2 = params['sigma2']

        phi = params['phi']
        theta = params['theta']
        omega = params['omega']

        args = Q0, Q1, Q2, A, B, mu0, mu1, mu2, sigma0, sigma1, sigma2, phi, theta, omega

        return self.jac(*args)/e

    def jac(self, Q0, Q1, Q2, A, B, mu0, mu1, mu2, sigma0, sigma1, sigma2, phi, theta, omega):

        yfit = self.gaussian(Q0, Q1, Q2, A, mu0, mu1, mu2, sigma0, sigma1, sigma2, phi, theta, omega)

        x0, x1, x2 = Q0-mu0, Q1-mu1, Q2-mu2

        U = self.U_matrix(phi, theta, omega)

        a = 0.5*(U[0,0]**2/sigma0**2+U[0,1]**2/sigma1**2+U[0,2]**2/sigma2**2)
        b = 0.5*(U[1,0]**2/sigma0**2+U[1,1]**2/sigma1**2+U[1,2]**2/sigma2**2)
        c = 0.5*(U[2,0]**2/sigma0**2+U[2,1]**2/sigma1**2+U[2,2]**2/sigma2**2)

        d = U[1,0]*U[2,0]/sigma0**2+U[1,1]*U[2,1]/sigma1**2+U[1,2]*U[2,2]/sigma2**2
        e = U[2,0]*U[0,0]/sigma0**2+U[2,1]*U[0,1]/sigma1**2+U[2,2]*U[0,2]/sigma2**2
        f = U[0,0]*U[1,0]/sigma0**2+U[0,1]*U[1,1]/sigma1**2+U[0,2]*U[1,2]/sigma2**2

        yprime_A = np.exp(-(a*x0**2+b*x1**2+c*x2**2+d*x1*x2+e*x0*x2+f*x0*x1))
        yprime_B = np.ones_like(yprime_A)

        aprime_sigma0 = -U[0,0]**2/sigma0**3
        aprime_sigma1 = -U[0,1]**2/sigma1**3
        aprime_sigma2 = -U[0,2]**2/sigma2**3

        bprime_sigma0 = -U[1,0]**2/sigma0**3
        bprime_sigma1 = -U[1,1]**2/sigma1**3
        bprime_sigma2 = -U[1,2]**2/sigma2**3

        cprime_sigma0 = -U[2,0]**2/sigma0**3
        cprime_sigma1 = -U[2,1]**2/sigma1**3
        cprime_sigma2 = -U[2,2]**2/sigma2**3

        dprime_sigma0 = -2*U[1,0]*U[2,0]/sigma0**3
        dprime_sigma1 = -2*U[1,1]*U[2,1]/sigma1**3
        dprime_sigma2 = -2*U[1,2]*U[2,2]/sigma2**3

        eprime_sigma0 = -2*U[2,0]*U[0,0]/sigma0**3
        eprime_sigma1 = -2*U[2,1]*U[0,1]/sigma1**3
        eprime_sigma2 = -2*U[2,2]*U[0,2]/sigma2**3

        fprime_sigma0 = -2*U[0,0]*U[1,0]/sigma0**3
        fprime_sigma1 = -2*U[0,1]*U[1,1]/sigma1**3
        fprime_sigma2 = -2*U[0,2]*U[1,2]/sigma2**3

        ux = np.cos(phi)*np.sin(theta)
        uy = np.sin(phi)*np.sin(theta)
        uz = np.cos(theta)

        Uprime_omega = np.array([[(ux**2-1)*np.sin(omega), ux*uy*np.sin(omega)-uz*np.cos(omega), ux*uz*np.sin(omega)+uy*np.cos(omega)],
                                 [uy*ux*np.sin(omega)+uz*np.cos(omega), (uy**2-1)*np.sin(omega), uy*uz*np.sin(omega)-ux*np.cos(omega)],
                                 [uz*ux*np.sin(omega)-uy*np.cos(omega), uz*uy*np.sin(omega)+ux*np.cos(omega), (uz**2-1)*np.sin(omega)]])

        Uprime_phi = np.array([[-2*ux*uy*(1-np.cos(omega)), (ux**2-uy**2)*(1-np.cos(omega)), -uy*uz*(1-np.cos(omega))+ux*np.sin(omega)],
                               [(ux**2-uy**2)*(1-np.cos(omega)),  2*uy*ux*(1-np.cos(omega)),  ux*uz*(1-np.cos(omega))+uy*np.sin(omega)],
                               [-uz*uy*(1-np.cos(omega))-ux*np.sin(omega),  uz*ux*(1-np.cos(omega))-uy*np.sin(omega),                0]])

        Uprime_theta = np.array([[2*ux**2*(1-np.cos(omega)), 2*ux*uy*(1-np.cos(omega))+uz*np.sin(omega)*np.tan(theta)**2, ux*uz*(1-np.cos(omega))*(1-np.tan(theta)**2)+uy*np.sin(omega)],
                                 [2*uy*ux*(1-np.cos(omega))-uz*np.sin(omega)*np.tan(theta)**2, 2*uy**2*(1-np.cos(omega)), uy*uz*(1-np.cos(omega))*(1-np.tan(theta)**2)-ux*np.sin(omega)],
                                 [uz*uy*(1-np.cos(omega))*(1-np.tan(theta)**2)-uy*np.sin(omega), uz*uy*(1-np.cos(omega))*(1-np.tan(theta)**2)+ux*np.sin(omega), -2*uz**2*(1-np.cos(omega))*np.tan(theta)**2]])/np.tan(theta)

        yprime_mu0 = yfit*(2*a*x0+  f*x1+  e*x2)
        yprime_mu1 = yfit*(  f*x0+2*b*x1+  d*x2)
        yprime_mu2 = yfit*(  e*x0+  d*x1+2*c*x2)

        yprime_sigma0 = -yfit*(aprime_sigma0*x0**2+bprime_sigma0*x1**2+cprime_sigma0*x2**2+dprime_sigma0*x1*x2+eprime_sigma0*x0*x2+fprime_sigma0*x0*x1)
        yprime_sigma1 = -yfit*(aprime_sigma1*x0**2+bprime_sigma1*x1**2+cprime_sigma1*x2**2+dprime_sigma1*x1*x2+eprime_sigma1*x0*x2+fprime_sigma1*x0*x1)
        yprime_sigma2 = -yfit*(aprime_sigma2*x0**2+bprime_sigma2*x1**2+cprime_sigma2*x2**2+dprime_sigma2*x1*x2+eprime_sigma2*x0*x2+fprime_sigma2*x0*x1)

        aprime_phi   =   Uprime_phi[0,0]*U[0,0]/sigma0**2+  Uprime_phi[0,1]*U[0,1]/sigma1**2+  Uprime_phi[0,2]*U[0,2]/sigma2**2
        aprime_theta = Uprime_theta[0,0]*U[0,0]/sigma0**2+Uprime_theta[0,1]*U[0,1]/sigma1**2+Uprime_theta[0,2]*U[0,2]/sigma2**2
        aprime_omega = Uprime_omega[0,0]*U[0,0]/sigma0**2+Uprime_omega[0,1]*U[0,1]/sigma1**2+Uprime_omega[0,2]*U[0,2]/sigma2**2

        bprime_phi   =   Uprime_phi[1,0]*U[1,0]/sigma0**2+  Uprime_phi[1,1]*U[1,1]/sigma1**2+  Uprime_phi[1,2]*U[1,2]/sigma2**2
        bprime_theta = Uprime_theta[1,0]*U[1,0]/sigma0**2+Uprime_theta[1,1]*U[1,1]/sigma1**2+Uprime_theta[1,2]*U[1,2]/sigma2**2
        bprime_omega = Uprime_omega[1,0]*U[1,0]/sigma0**2+Uprime_omega[1,1]*U[1,1]/sigma1**2+Uprime_omega[1,2]*U[1,2]/sigma2**2

        cprime_phi   =   Uprime_phi[2,0]*U[2,0]/sigma0**2+  Uprime_phi[2,1]*U[2,1]/sigma1**2+  Uprime_phi[2,2]*U[2,2]/sigma2**2
        cprime_theta = Uprime_theta[2,0]*U[2,0]/sigma0**2+Uprime_theta[2,1]*U[2,1]/sigma1**2+Uprime_theta[2,2]*U[2,2]/sigma2**2
        cprime_omega = Uprime_omega[2,0]*U[2,0]/sigma0**2+Uprime_omega[2,1]*U[2,1]/sigma1**2+Uprime_omega[2,2]*U[2,2]/sigma2**2

        dprime_phi   = (  Uprime_phi[1,0]*U[2,0]+  Uprime_phi[2,0]*U[1,0])/sigma0**2+(  Uprime_phi[1,1]*U[2,1]+  Uprime_phi[2,1]*U[1,1])/sigma1**2+(  Uprime_phi[1,2]*U[2,2]+  Uprime_phi[2,2]*U[1,2])/sigma2**2
        dprime_theta = (Uprime_theta[1,0]*U[2,0]+Uprime_theta[2,0]*U[1,0])/sigma0**2+(Uprime_theta[1,1]*U[2,1]+Uprime_theta[2,1]*U[1,1])/sigma1**2+(Uprime_theta[1,2]*U[2,2]+Uprime_theta[2,2]*U[1,2])/sigma2**2
        dprime_omega = (Uprime_omega[1,0]*U[2,0]+Uprime_omega[2,0]*U[1,0])/sigma0**2+(Uprime_omega[1,1]*U[2,1]+Uprime_omega[2,1]*U[1,1])/sigma1**2+(Uprime_omega[1,2]*U[2,2]+Uprime_omega[2,2]*U[1,2])/sigma2**2

        eprime_phi   = (  Uprime_phi[2,0]*U[0,0]+  Uprime_phi[0,0]*U[2,0])/sigma0**2+(  Uprime_phi[2,1]*U[0,1]+  Uprime_phi[0,1]*U[2,1])/sigma1**2+(  Uprime_phi[2,2]*U[0,2]+  Uprime_phi[0,2]*U[2,2])/sigma2**2
        eprime_theta = (Uprime_theta[2,0]*U[0,0]+Uprime_theta[0,0]*U[2,0])/sigma0**2+(Uprime_theta[2,1]*U[0,1]+Uprime_theta[0,1]*U[2,1])/sigma1**2+(Uprime_theta[2,2]*U[0,2]+Uprime_theta[0,2]*U[2,2])/sigma2**2
        eprime_omega = (Uprime_omega[2,0]*U[0,0]+Uprime_omega[0,0]*U[2,0])/sigma0**2+(Uprime_omega[2,1]*U[0,1]+Uprime_omega[0,1]*U[2,1])/sigma1**2+(Uprime_omega[2,2]*U[0,2]+Uprime_omega[0,2]*U[2,2])/sigma2**2

        fprime_phi   = (  Uprime_phi[0,0]*U[1,0]+  Uprime_phi[1,0]*U[0,0])/sigma0**2+(  Uprime_phi[0,1]*U[1,1]+  Uprime_phi[1,1]*U[0,1])/sigma1**2+(  Uprime_phi[0,2]*U[1,2]+  Uprime_phi[1,2]*U[0,2])/sigma2**2
        fprime_theta = (Uprime_theta[0,0]*U[1,0]+Uprime_theta[1,0]*U[0,0])/sigma0**2+(Uprime_theta[0,1]*U[1,1]+Uprime_theta[1,1]*U[0,1])/sigma1**2+(Uprime_theta[0,2]*U[1,2]+Uprime_theta[1,2]*U[0,2])/sigma2**2
        fprime_omega = (Uprime_omega[0,0]*U[1,0]+Uprime_omega[1,0]*U[0,0])/sigma0**2+(Uprime_omega[0,1]*U[1,1]+Uprime_omega[1,1]*U[0,1])/sigma1**2+(Uprime_omega[0,2]*U[1,2]+Uprime_omega[1,2]*U[0,2])/sigma2**2

        yprime_phi   = -yfit*(aprime_phi  *x0**2+bprime_phi  *x1**2+cprime_phi  *x2**2+dprime_phi*  x1*x2+eprime_phi  *x0*x2+fprime_phi  *x0*x1)
        yprime_theta = -yfit*(aprime_theta*x0**2+bprime_theta*x1**2+cprime_theta*x2**2+dprime_theta*x1*x2+eprime_theta*x0*x2+fprime_theta*x0*x1)
        yprime_omega = -yfit*(aprime_omega*x0**2+bprime_omega*x1**2+cprime_omega*x2**2+dprime_omega*x1*x2+eprime_omega*x0*x2+fprime_omega*x0*x1)

        J = np.stack((yprime_A,yprime_B,yprime_mu0,yprime_mu1,yprime_mu2,yprime_sigma0,yprime_sigma1,yprime_sigma2,yprime_phi,yprime_theta,yprime_omega))

        J[~np.isfinite(J)] = 1e+15
        J[~np.isfinite(J)] = 1e+15

        return J

    def fit(self):

        out = Minimizer(self.residual, self.params, fcn_args=(self.x, self.y, self.e)) #, Dfun=self.gradient, col_deriv=True, nan_policy='raise'
        result = out.minimize(method='leastsq')

        #result = out.prepare_fit()

        self.params = result.params

        # report_fit(result)

        A = result.params['A'].value
        B = result.params['B'].value

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

        return A, B, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01, boundary

    def estimate(self):

        Q0, Q1, Q2 = self.x
        y, e = self.y, self.e

        params = self.params

        bkg = np.percentile(y, 25)

        z = y.copy()

        z -= bkg
        z[z < 0] = 0

        weights = z**2/e**2

        mask = (z > 0) & (z < np.inf) & (e > 0) & (e < np.inf)

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

        x = np.exp(-0.5*(inv_S[0,0]*x0**2+inv_S[1,1]*x1**2+inv_S[2,2]*x2**2\
                     +2*(inv_S[1,2]*x1*x2+inv_S[0,2]*x0*x2+inv_S[0,1]*x0*x1)))

        A = (np.array([x, np.ones_like(x)])/e).T
        b = y/e

        coeff, r, rank, s = np.linalg.lstsq(A, b, rcond=None)

        A, B = coeff

        boundary = self.check_outside(A, B, mu0, mu1, mu2, sig0, sig1, sig2, params)

        return A, B, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01, boundary

    def check_outside(self, A, B, mu0, mu1, mu2, sigma0, sigma1, sigma2, params):

        A_min, A_max = params['A'].min, params['A'].max
        B_min, B_max = params['B'].min, params['B'].max

        mu0_min, mu0_max = params['mu0'].min, params['mu0'].max
        mu1_min, mu1_max = params['mu1'].min, params['mu1'].max
        mu2_min, mu2_max = params['mu2'].min, params['mu2'].max

        sigma0_min, sigma0_max = params['sigma0'].min, params['sigma0'].max
        sigma1_min, sigma1_max = params['sigma1'].min, params['sigma1'].max
        sigma2_min, sigma2_max = params['sigma2'].min, params['sigma2'].max

        boundary =  (A <= A_min) or (A >= A_max)\
                 or (B <= B_min) or (B >= B_max)\
                 or (mu0 <= mu0_min) or (mu0 >= mu0_max)\
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

        boundary = np.isclose(A, A_min, rtol=1e-2) | np.isclose(A, A_max, rtol=1e-2) \
                 | np.isclose(A, A_min, rtol=1e-2) | np.isclose(A, A_max, rtol=1e-2) \
                 | np.isclose(mu0, mu0_min, rtol=1e-2) | np.isclose(mu0, mu0_max, rtol=1e-2) \
                 | np.isclose(mu1, mu1_min, rtol=1e-2) | np.isclose(mu1, mu1_max, rtol=1e-2) \
                 | np.isclose(mu2, mu2_min, rtol=1e-2) | np.isclose(mu2, mu2_max, rtol=1e-2) \
                 | np.isclose(sigma0, sigma0_min, rtol=1e-2) | np.isclose(sigma0, sigma0_max, rtol=1e-2) \
                 | np.isclose(sigma1, sigma1_min, rtol=1e-2) | np.isclose(sigma1, sigma1_max, rtol=1e-2) \
                 | np.isclose(sigma2, sigma2_min, rtol=1e-2) | np.isclose(sigma2, sigma2_max, rtol=1e-2)

        return boundary

    def covariance_matrix(self, sig0, sig1, sig2, rho12, rho02, rho01):

        sig = np.diag([sig0, sig1, sig2])

        rho = np.array([[1, rho01, rho02],
                        [rho01, 1, rho12],
                        [rho02, rho12, 1]])

        S = np.dot(np.dot(sig, rho), sig)

        return S

    def integrated(self, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01):

        S = self.covariance_matrix(sig0, sig1, sig2, rho12, rho02, rho01)

        inv_S = np.linalg.inv(S)

        Q0, Q1, Q2 = self.x

        x0, x1, x2 = Q0-mu0, Q1-mu1, Q2-mu2

        norm = np.sqrt(np.linalg.det(2*np.pi*S))

        x = np.exp(-0.5*(inv_S[0,0]*x0**2+inv_S[1,1]*x1**2+inv_S[2,2]*x2**2\
                     +2*(inv_S[1,2]*x1*x2+inv_S[0,2]*x0*x2+inv_S[0,1]*x0*x1)))/norm

        A = (np.array([x, np.ones_like(x)])/self.e).T
        b = self.y/self.e

        coeff, r, rank, s = np.linalg.lstsq(A, b, rcond=None)

        intens, bkg = coeff

        cov = np.dot(A.T, A)
        if np.linalg.det(cov) > 0:
            sig = np.sqrt(np.linalg.inv(cov)[0,0])
        else:
            sig = intens

        return intens, bkg, sig

    def model(self, x, intens, bkg, mu0, mu1, mu2, sig0, sig1, sig2, rho12, rho02, rho01):

        S = self.covariance_matrix(sig0, sig1, sig2, rho12, rho02, rho01)

        inv_S = np.linalg.inv(S)

        x0, x1, x2 = x[0]-mu0, x[1]-mu1, x[2]-mu2

        norm = np.sqrt(np.linalg.det(2*np.pi*S))

        return intens*np.exp(-0.5*(inv_S[0,0]*x0**2+inv_S[1,1]*x1**2+inv_S[2,2]*x2**2\
                               +2*(inv_S[1,2]*x1*x2+inv_S[0,2]*x0*x2+inv_S[0,1]*x0*x1)))/norm+bkg

class PeakStatistics:

    def __init__(self, filename, space_group):

        self.filename = filename
        self.data = np.genfromtxt(filename, delimiter=(4,4,4,8,8,8))

        self.sg = SpaceGroupFactory.createSpaceGroup(space_group)
        self.pg = PointGroupFactory.createPointGroupFromSpaceGroup(self.sg)

    def prune_outliers(self):

        filename = self.filename
        data = self.data

        sg = self.sg
        pg = self.pg

        miss, total = 0, 0

        lines = []

        fname, ext = os.path.splitext(filename)

        f = open(fname+'_prune.txt', 'w')

        f.write('space group #{} ({})\n'.format(sg.getNumber(),sg.getHMSymbol()))
        f.write('reflections:\n')

        dictionary = {}

        for line in data:

            h, k, l, I, sig, d = line

            hkl = V3D(h,k,l)

            if sg.isAllowedReflection(hkl):

                lines.append(line)

                equivalents = pg.getEquivalents(hkl)

                key = tuple(equivalents)

                if dictionary.get(key) is None:

                    dictionary[key] = [len(equivalents),d,[(h,k,l)],[I],[sig]]

                else:

                    item = dictionary[key]

                    redundancy, d_spacing, peak_list, intens_list, sig_intens_list = item

                    intens_list.append(I)
                    sig_intens_list.append(sig)
                    peak_list.append((h,k,l))

                    item = [redundancy, d_spacing, peak_list, intens_list, sig_intens_list]

                    dictionary[key] = item

            else:

                f.write('({},{},{}) forbidden d = {:2.4f} \u212B\n'.format(int(h),int(k),int(l),d))

                miss += 1

            total += 1

        f.write('{}/{} reflections not allowed in space group\n'.format(miss,total))

        miss, total = 0, 0

        data = []

        for key in dictionary.keys():

            item = dictionary[key]

            redundancy, d_spacing, peak_list, intens_list, sig_intens_list = item

            intens_list, sig_intens_list = np.array(intens_list), np.array(sig_intens_list)

            median = np.median(intens_list)
            Q1, Q3 = np.percentile(intens_list, [25,75])
            IQR = Q3-Q1

            high = np.argwhere(Q3+1.5*IQR < intens_list).flatten()
            low = np.argwhere(Q1-1.5*IQR > intens_list).flatten()

            if len(high) > 0:
                f.write('outlier, intensity too high : {}\n'.format(','.join(['({},{},{})'.format(*peak_list[ind]) for ind in high])))

            if len(low) > 0:
                f.write('outlier, intensity too low : {}\n'.format(','.join(['({},{},{})'.format(*peak_list[ind]) for ind in low])))

            mask = np.concatenate((low,high)).tolist()

            median = np.median(sig_intens_list)
            Q1, Q3 = np.percentile(sig_intens_list, [25,75])
            IQR = Q3-Q1

            high = np.argwhere(Q3+1.5*IQR < sig_intens_list).flatten().tolist()

            mask += high

            for i in range(len(intens_list)):

                if i not in mask:

                    h, k, l = peak_list[i]
                    I = intens_list[i]
                    sig = sig_intens_list[i]
                    d = d_spacing

                    line = h, k, l, I, sig, d

                    data.append(line)

                else:

                    miss += 1

                total += 1

        f.write('{}/{} reflections outliers\n'.format(miss,total))

        self.data = data

    def write_statisics(self):

        filename = self.filename
        data = self.data

        sg = self.sg
        pg = self.pg

        lines = []

        fname, ext = os.path.splitext(filename)

        f = open(fname+'_symm.txt', 'w')

        f.write('d-spacing \u212B   | Comp    | R(mrg)  | R(pim)\n')

        dictionary = {}

        for line in data:

            h, k, l, I, sig, d = line

            hkl = V3D(h,k,l)

            if sg.isAllowedReflection(hkl):

                lines.append(line)

                equivalents = pg.getEquivalents(hkl)

                key = tuple(equivalents)

                if dictionary.get(key) is None:

                    dictionary[key] = [len(equivalents),d,[(h,k,l)],[I],[sig]]

                else:

                    item = dictionary[key]

                    redundancy, d_spacing, peak_list, intens_list, sig_intens_list = item

                    intens_list.append(I)
                    sig_intens_list.append(sig)
                    peak_list.append((h,k,l))

                    item = [redundancy, d_spacing, peak_list, intens_list, sig_intens_list]

                    dictionary[key] = item

        r, n, d = [], [], []

        I_sum, I_mae = [], []

        for key in dictionary.keys():

            item = dictionary[key]

            redundancy, d_spacing, peak_list, intens_list, sig_intens_list = item

            I_mean = np.mean(intens_list)

            r.append(redundancy)
            n.append(np.unique(peak_list,axis=0).size)
            d.append(d_spacing)

            I_sum.append(np.sum(intens_list))
            I_mae.append(np.sum(np.abs(np.array(intens_list)-I_mean)))

        r, n, d = np.array(r), np.array(n), np.array(d)

        I_sum, I_mae = np.array(I_sum), np.array(I_mae)

        sort = np.argsort(d)[::-1]

        r, n, d = r[sort], n[sort], d[sort]
        I_sum, I_mae = I_sum[sort], I_mae[sort]

        n_pk = len(d)
        n_sp = np.min([n_pk,20])

        split = np.array_split(np.arange(len(d)), n_sp)

        for s in split:

            d_min, d_max = d[s].min(), d[s].max()

            comp = 100*n[s].sum()/r[s].sum()

            R_merge = 100*I_mae[s].sum()/I_sum[s].sum()
            R_pim = 100*(np.sqrt(1/(n[s]-1))*I_mae[s]).sum()/I_sum[s].sum()

            f.write('{:6.3f}-{:6.3f} | {:6.2f}% | {:6.2f}% | {:6.2f}%\n'.format(d_max,d_min,comp,R_merge,R_pim))

    def write_intensity(self):

        fname, ext = os.path.splitext(self.filename)

        hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:8.4f}\n'

        with open(fname+'_prune'+ext, 'w') as f:

            for line in self.data:

                f.write(hkl_format.format(*line))