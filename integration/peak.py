from mantid.simpleapi import CreateSingleValuedWorkspace, CreatePeaksWorkspace
from mantid.simpleapi import CloneWorkspace, DeleteWorkspace, DeleteTableRows
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
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 8

import numpy as np
import scipy.interpolate
import scipy.integrate
import scipy.special
import scipy.optimize
import scipy.spatial
import scipy.signal

#np.seterr(divide='ignore', invalid='ignore')
#np.seterr(**settings)

np.seterr(divide='ignore', invalid='ignore')

#from sklearn.cluster import DBSCAN
#from sklearn.cluster import MeanShift, estimate_bandwidth

import os
import pprint
import dill as pickle

from lmfit import Parameters, Minimizer, report_fit

import re

#import numba as nb
#@nb.jit(nopython=True)

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

        self.fig = plt.figure(num='peak-envelope', figsize=(20,12), dpi=144)
        gs = gridspec.GridSpec(2, 2, figure=self.fig, wspace=0.25, width_ratios=[0.5,0.5], height_ratios=[0.5,0.5])

        gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,0], hspace=0.25)
        gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0,1], hspace=0.25)
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1,0], hspace=0.25, wspace=0.5, width_ratios=[1,1,1,0.1])
        gs3 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1,1], hspace=0.25, wspace=0.5, width_ratios=[1,1,1,0.1])

        self.ax_Q = self.fig.add_subplot(gs0[0,0])
        self.ax_Q2 = self.fig.add_subplot(gs0[1,0])

        self.ax_Q.minorticks_on()
        self.ax_Q2.minorticks_on()

        self.ax_p_proj = self.fig.add_subplot(gs1[0,0])
        self.ax_p2_proj = self.fig.add_subplot(gs1[1,0])

        self.ax_s_proj = self.fig.add_subplot(gs1[0,1])
        self.ax_s2_proj = self.fig.add_subplot(gs1[1,1])

        self.ax_p_proj.minorticks_on()
        self.ax_p2_proj.minorticks_on()

        self.ax_s_proj.minorticks_on()
        self.ax_s2_proj.minorticks_on()

        self.ax_Qu = self.fig.add_subplot(gs2[0,0])
        self.ax_Qu2 = self.fig.add_subplot(gs3[0,0])

        self.ax_Qv = self.fig.add_subplot(gs2[0,1])
        self.ax_Qv2 = self.fig.add_subplot(gs3[0,1])

        self.ax_uv = self.fig.add_subplot(gs2[0,2])
        self.ax_uv2 = self.fig.add_subplot(gs3[0,2])

        self.ax_Qu.minorticks_on()
        self.ax_Qu2.minorticks_on()

        self.ax_Qv.minorticks_on()
        self.ax_Qv2.minorticks_on()

        self.ax_uv.minorticks_on()
        self.ax_uv2.minorticks_on()

        self.ax_Qu_fit = self.fig.add_subplot(gs2[1,0])
        self.ax_Qu2_fit = self.fig.add_subplot(gs3[1,0])

        self.ax_Qv_fit = self.fig.add_subplot(gs2[1,1])
        self.ax_Qv2_fit = self.fig.add_subplot(gs3[1,1])

        self.ax_uv_fit = self.fig.add_subplot(gs2[1,2])
        self.ax_uv2_fit = self.fig.add_subplot(gs3[1,2])

        self.ax_Qu_fit.minorticks_on()
        self.ax_Qu2_fit.minorticks_on()

        self.ax_Qv_fit.minorticks_on()
        self.ax_Qv2_fit.minorticks_on()

        self.ax_uv_fit.minorticks_on()
        self.ax_uv2_fit.minorticks_on()

        #self.ax_p_proj.set_xlabel('Q\u2081 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_p2_proj.set_xlabel('Q\u2081 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        #self.ax_s_proj.set_xlabel('Q\u2081 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_s2_proj.set_xlabel('Q\u2081 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        self.ax_p_proj.set_ylabel('Q\u2082 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_p2_proj.set_ylabel('Q\u2082 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        #self.ax_s_proj.set_ylabel('Q\u2082 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        #self.ax_s2_proj.set_ylabel('Q\u2082 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

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
        self.im_p2_proj = self.ax_p2_proj.imshow([[0,1],[0,1]], interpolation='nearest',
                                                 origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.im_s_proj = self.ax_s_proj.imshow([[0,1],[0,1]], interpolation='nearest',
                                               origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())
        self.im_s2_proj = self.ax_s2_proj.imshow([[0,1],[0,1]], interpolation='nearest',
                                                 origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        divider_p = make_axes_locatable(self.ax_p_proj)
        divider_p2 = make_axes_locatable(self.ax_p2_proj)
        
        divider_s = make_axes_locatable(self.ax_s_proj)
        divider_s2 = make_axes_locatable(self.ax_s2_proj)

        width_p = axes_size.AxesY(self.ax_p_proj, aspect=0.05)
        width_p2 = axes_size.AxesY(self.ax_p2_proj, aspect=0.05)

        width_s = axes_size.AxesY(self.ax_s_proj, aspect=0.05)
        width_s2 = axes_size.AxesY(self.ax_s2_proj, aspect=0.05)

        pad_p = axes_size.Fraction(0.5, width_p)
        pad_p2 = axes_size.Fraction(0.5, width_p2)
        
        pad_s = axes_size.Fraction(0.5, width_s)
        pad_s2 = axes_size.Fraction(0.5, width_s2)

        cax_p = divider_p.append_axes('right', size=width_p, pad=pad_p)
        cax_p2 = divider_p2.append_axes('right', size=width_p2, pad=pad_p2)
        
        cax_s = divider_s.append_axes('right', size=width_s, pad=pad_s)
        cax_s2 = divider_s2.append_axes('right', size=width_s2, pad=pad_s2)

        self.cb_p = self.fig.colorbar(self.im_p_proj, cax=cax_p)
        self.cb_p2 = self.fig.colorbar(self.im_p2_proj, cax=cax_p2)
        
        self.cb_s = self.fig.colorbar(self.im_s_proj, cax=cax_s)
        self.cb_s2 = self.fig.colorbar(self.im_s2_proj, cax=cax_s2)

        self.cb_p.ax.minorticks_on()
        self.cb_p2.ax.minorticks_on()

        self.cb_s.ax.minorticks_on()
        self.cb_s2.ax.minorticks_on()

        self.elli_p = Ellipse((0,0), width=1, height=1, edgecolor='C3', facecolor='none', rasterized=False, zorder=1000)
        self.elli_p2 = Ellipse((0,0), width=1, height=1, edgecolor='C3', facecolor='none', rasterized=False, zorder=1000)

        self.elli_s = Ellipse((0,0), width=1, height=1, edgecolor='C3', facecolor='none', rasterized=False, zorder=1000)
        self.elli_s2 = Ellipse((0,0), width=1, height=1, edgecolor='C3', facecolor='none', rasterized=False, zorder=1000)

        self.trans_elli_p = transforms.Affine2D()
        self.trans_elli_p2 = transforms.Affine2D()

        self.trans_elli_s = transforms.Affine2D()
        self.trans_elli_s2 = transforms.Affine2D()

        self.ax_p_proj.add_patch(self.elli_p)
        self.ax_p2_proj.add_patch(self.elli_p2)

        self.ax_s_proj.add_patch(self.elli_s)
        self.ax_s2_proj.add_patch(self.elli_s2)

        self.ax_Q2.set_rasterization_zorder(100)
        self.ax_Q2.set_xlabel('Q\u209A [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_Q2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        self.line_Q2_p, self.caps_Q2_p, self.bars_Q2_p = self.ax_Q2.errorbar([0], [0], yerr=[0], fmt='.-', rasterized=False, zorder=2)
        self.norm_Q2_p = self.ax_Q2.plot([0], [0], '--', rasterized=False, zorder=1)
        self.line_Q2_s, self.caps_Q2_s, self.bars_Q2_s = self.ax_Q2.errorbar([0], [0], yerr=[0], fmt='.-', rasterized=False, zorder=2)
        self.norm_Q2_s = self.ax_Q2.plot([0], [0], '--', rasterized=False, zorder=1)

        # ---

        self.im_Qu = self.ax_Qu.imshow([[0,1],[0,1]], interpolation='nearest',
                                       origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())
        self.im_Qu2 = self.ax_Qu2.imshow([[0,1],[0,1]], interpolation='nearest',
                                         origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.im_Qv = self.ax_Qv.imshow([[0,1],[0,1]], interpolation='nearest',
                                       origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())
        self.im_Qv2 = self.ax_Qv2.imshow([[0,1],[0,1]], interpolation='nearest',
                                         origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.im_uv = self.ax_uv.imshow([[0,1],[0,1]], interpolation='nearest',
                                       origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())
        self.im_uv2 = self.ax_uv2.imshow([[0,1],[0,1]], interpolation='nearest',
                                         origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.ax_Qu.set_rasterization_zorder(100)
        self.ax_Qu2.set_rasterization_zorder(100)

        self.ax_Qv.set_rasterization_zorder(100)
        self.ax_Qv2.set_rasterization_zorder(100)

        self.ax_uv.set_rasterization_zorder(100)
        self.ax_uv2.set_rasterization_zorder(100)

        self.im_Qu_fit = self.ax_Qu_fit.imshow([[0,1],[0,1]], interpolation='nearest',
                                               origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())
        self.im_Qu2_fit = self.ax_Qu2_fit.imshow([[0,1],[0,1]], interpolation='nearest',
                                                 origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.im_Qv_fit = self.ax_Qv_fit.imshow([[0,1],[0,1]], interpolation='nearest',
                                               origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())
        self.im_Qv2_fit = self.ax_Qv2_fit.imshow([[0,1],[0,1]], interpolation='nearest',
                                                 origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.im_uv_fit = self.ax_uv_fit.imshow([[0,1],[0,1]], interpolation='nearest',
                                               origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())
        self.im_uv2_fit = self.ax_uv2_fit.imshow([[0,1],[0,1]], interpolation='nearest',
                                                 origin='lower', extent=[0,1,0,1], norm=mpl.colors.LogNorm())

        self.ax_Qu_fit.set_rasterization_zorder(100)
        self.ax_Qu2_fit.set_rasterization_zorder(100)

        self.ax_Qv_fit.set_rasterization_zorder(100)
        self.ax_Qv2_fit.set_rasterization_zorder(100)

        self.ax_uv_fit.set_rasterization_zorder(100)
        self.ax_uv2_fit.set_rasterization_zorder(100)

        self.peak_pu = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.peak_pu2 = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.inner_pu = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.inner_pu2 = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.outer_pu = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.outer_pu2 = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.peak_pv = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.peak_pv2 = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.inner_pv = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.inner_pv2 = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.outer_pv = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.outer_pv2 = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.peak_uv = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.peak_uv2 = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.inner_uv = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.inner_uv2 = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.outer_uv = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.outer_uv2 = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.peak_pu_fit = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.peak_pu2_fit = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.inner_pu_fit = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.inner_pu2_fit = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.outer_pu_fit = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.outer_pu2_fit = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.peak_pv_fit = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.peak_pv2_fit = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.inner_pv_fit = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.inner_pv2_fit = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.outer_pv_fit = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.outer_pv2_fit = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.peak_uv_fit = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.peak_uv2_fit = Ellipse((0,0), width=1, height=1, linestyle='-', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.inner_uv_fit = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.inner_uv2_fit = Ellipse((0,0), width=1, height=1, linestyle=':', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.outer_uv_fit = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)
        self.outer_uv2_fit = Ellipse((0,0), width=1, height=1, linestyle='--', edgecolor='w', facecolor='none', rasterized=False, zorder=1000)

        self.trans_peak_pu = transforms.Affine2D()
        self.trans_peak_pu2 = transforms.Affine2D()

        self.trans_inner_pu = transforms.Affine2D()
        self.trans_inner_pu2 = transforms.Affine2D()

        self.trans_outer_pu = transforms.Affine2D()
        self.trans_outer_pu2 = transforms.Affine2D()

        self.trans_peak_pv = transforms.Affine2D()
        self.trans_peak_pv2 = transforms.Affine2D()

        self.trans_inner_pv = transforms.Affine2D()
        self.trans_inner_pv2 = transforms.Affine2D()

        self.trans_outer_pv = transforms.Affine2D()
        self.trans_outer_pv2 = transforms.Affine2D()

        self.trans_peak_uv = transforms.Affine2D()
        self.trans_peak_uv2 = transforms.Affine2D()

        self.trans_inner_uv = transforms.Affine2D()
        self.trans_inner_uv2 = transforms.Affine2D()

        self.trans_outer_uv = transforms.Affine2D()
        self.trans_outer_uv2 = transforms.Affine2D()

        self.trans_peak_pu_fit = transforms.Affine2D()
        self.trans_peak_pu2_fit = transforms.Affine2D()

        self.trans_inner_pu_fit = transforms.Affine2D()
        self.trans_inner_pu2_fit = transforms.Affine2D()

        self.trans_outer_pu_fit = transforms.Affine2D()
        self.trans_outer_pu2_fit = transforms.Affine2D()

        self.trans_peak_pv_fit = transforms.Affine2D()
        self.trans_peak_pv2_fit = transforms.Affine2D()

        self.trans_inner_pv_fit = transforms.Affine2D()
        self.trans_inner_pv2_fit = transforms.Affine2D()

        self.trans_outer_pv_fit = transforms.Affine2D()
        self.trans_outer_pv2_fit = transforms.Affine2D()

        self.trans_peak_uv_fit = transforms.Affine2D()
        self.trans_peak_uv2_fit = transforms.Affine2D()

        self.trans_inner_uv_fit = transforms.Affine2D()
        self.trans_inner_uv2_fit = transforms.Affine2D()

        self.trans_outer_uv_fit = transforms.Affine2D()
        self.trans_outer_uv2_fit = transforms.Affine2D()

        self.ax_Qu.add_patch(self.peak_pu)
        self.ax_Qu2.add_patch(self.peak_pu2)

        self.ax_Qu.add_patch(self.inner_pu)
        self.ax_Qu2.add_patch(self.inner_pu2)

        self.ax_Qu.add_patch(self.outer_pu)
        self.ax_Qu2.add_patch(self.outer_pu2)

        self.ax_Qv.add_patch(self.peak_pv)
        self.ax_Qv2.add_patch(self.peak_pv2)

        self.ax_Qv.add_patch(self.inner_pv)
        self.ax_Qv2.add_patch(self.inner_pv2)

        self.ax_Qv.add_patch(self.outer_pv)
        self.ax_Qv2.add_patch(self.outer_pv2)

        self.ax_uv.add_patch(self.peak_uv)
        self.ax_uv2.add_patch(self.peak_uv2)

        self.ax_uv.add_patch(self.inner_uv)
        self.ax_uv2.add_patch(self.inner_uv2)

        self.ax_uv.add_patch(self.outer_uv)
        self.ax_uv2.add_patch(self.outer_uv2)

        self.ax_Qu_fit.add_patch(self.peak_pu_fit)
        self.ax_Qu2_fit.add_patch(self.peak_pu2_fit)

        self.ax_Qu_fit.add_patch(self.inner_pu_fit)
        self.ax_Qu2_fit.add_patch(self.inner_pu2_fit)

        self.ax_Qu_fit.add_patch(self.outer_pu_fit)
        self.ax_Qu2_fit.add_patch(self.outer_pu2_fit)

        self.ax_Qv_fit.add_patch(self.peak_pv_fit)
        self.ax_Qv2_fit.add_patch(self.peak_pv2_fit)

        self.ax_Qv_fit.add_patch(self.inner_pv_fit)
        self.ax_Qv2_fit.add_patch(self.inner_pv2_fit)

        self.ax_Qv_fit.add_patch(self.outer_pv_fit)
        self.ax_Qv2_fit.add_patch(self.outer_pv2_fit)

        self.ax_uv_fit.add_patch(self.peak_uv_fit)
        self.ax_uv2_fit.add_patch(self.peak_uv2_fit)

        self.ax_uv_fit.add_patch(self.inner_uv_fit)
        self.ax_uv2_fit.add_patch(self.inner_uv2_fit)

        self.ax_uv_fit.add_patch(self.outer_uv_fit)
        self.ax_uv2_fit.add_patch(self.outer_uv2_fit)

        self.ax_Qu.set_ylabel('Q\u2081\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_Qu2.set_ylabel('Q\u2081\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        self.ax_Qv.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_Qv2.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        self.ax_uv.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_uv2.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        self.ax_Qv_fit.set_xlabel('Q\u209A [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_Qv2_fit.set_xlabel('Q\u209A [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        self.ax_Qu_fit.set_xlabel('Q\u209A [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_Qu2_fit.set_xlabel('Q\u209A [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        self.ax_uv_fit.set_xlabel('Q\u2081\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_uv2_fit.set_xlabel('Q\u2081\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        self.ax_Qu_fit.set_ylabel('Q\u2081\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_Qu2_fit.set_ylabel('Q\u2081\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        self.ax_Qv_fit.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_Qv2_fit.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        self.ax_uv_fit.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]
        self.ax_uv2_fit.set_ylabel('Q\u2082\u2032 [\u212B\u207B\u00B9]') # [\u212B\u207B\u00B9]

        ax = plt.subplot(gs2[:,3])
        ax2 = plt.subplot(gs3[:,3])

        ax.axis('off')
        ax2.axis('off')

        divider = make_axes_locatable(ax)
        divider2 = make_axes_locatable(ax2)

        width = axes_size.AxesY(ax, aspect=2.0)
        width2 = axes_size.AxesY(ax2, aspect=2.0)

        pad = axes_size.Fraction(0.5, width)
        pad2 = axes_size.Fraction(0.5, width2)

        cax = divider.append_axes('right', size=width, pad=pad)
        cax2 = divider2.append_axes('right', size=width2, pad=pad2)

        self.cb = self.fig.colorbar(self.im_uv, cax=cax)
        self.cb2 = self.fig.colorbar(self.im_uv2, cax=cax2)

        self.cb.ax.minorticks_on()
        self.cb2.ax.minorticks_on()

        self.__show_plots = False

    def update_plots(self, key, d):

        h, k, l, m, n, p = key

        if m**2+n**2+p**2 > 0:
            self.ax_Q.set_title('({} {} {}) ({} {} {})'.format(h,k,l,m,n,p))
        else:
            self.ax_Q.set_title('({} {} {})'.format(h,k,l))

        self.ax_p_proj.set_title('d = {:.4f} \u212B'.format(d))

    def clear_plots(self, key, d, lamda, two_theta, az_phi, n_runs):

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

        self.ax_p2_proj.set_title('')
        self.ax_s2_proj.set_title('')

        self.ax_p2_proj.set_aspect(1)
        self.ax_s2_proj.set_aspect(1)

        self.im_p2_proj.set_array(np.c_[[0,1],[1,0]])
        self.im_s2_proj.set_array(np.c_[[0,1],[1,0]])

        self.im_p2_proj.autoscale()
        self.im_s2_proj.autoscale()

        self.im_p2_proj.set_extent([0,1,0,1])
        self.im_s2_proj.set_extent([0,1,0,1])

        self.elli_p2.width = 1
        self.elli_s2.width = 1

        self.elli_p2.height = 1
        self.elli_s2.height = 1

        self.trans_elli_p2.clear()
        self.trans_elli_s2.clear()

        self.elli_p2.set_transform(self.trans_elli_p2+self.ax_p2_proj.transData)
        self.elli_s2.set_transform(self.trans_elli_s2+self.ax_s2_proj.transData)

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

            if np.isclose(az_phi[0],az_phi[1]):
                self.ax_Qu2_fit.set_title('\u03D5 = {:.1f}\u00B0'.format(az_phi[0]))
            else:
                self.ax_Qu2_fit.set_title('\u03D5 = {:.1f}-{:.1f}\u00B0'.format(*az_phi))

            self.ax_Qv.set_title('{} orientations'.format(n_runs))
        else:
            self.ax_Qu.set_title('\u03BB = {:.3f} \u212B'.format(lamda[0]))
            self.ax_Qu_fit.set_title('2\u03B8 = {:.1f}\u00B0'.format(two_theta[0]))
            self.ax_Qu2_fit.set_title('\u03D5 = {:.1f}\u00B0'.format(az_phi[0]))

            self.ax_Qv.set_title('1 orientation')
        
        self.ax_Qu2.set_title('')
        self.ax_Qv2.set_title('')

        self.ax_Qv_fit.set_title('')
        self.ax_Qv2_fit.set_title('')

        self.ax_uv.set_title('')
        self.ax_uv2.set_title('')

        self.ax_uv_fit.set_title('')
        self.ax_uv2_fit.set_title('')

        self.im_Qu.set_array(np.c_[[0,1],[1,0]])
        self.im_Qu2.set_array(np.c_[[0,1],[1,0]])

        self.im_Qv.set_array(np.c_[[0,1],[1,0]])
        self.im_Qv2.set_array(np.c_[[0,1],[1,0]])

        self.im_uv.set_array(np.c_[[0,1],[1,0]])
        self.im_uv2.set_array(np.c_[[0,1],[1,0]])

        self.im_Qu.autoscale()
        self.im_Qu2.autoscale()

        self.im_Qv.autoscale()
        self.im_Qv2.autoscale()

        self.im_uv.autoscale()
        self.im_uv2.autoscale()

        self.im_Qu.set_extent([0,1,0,1])
        self.im_Qu2.set_extent([0,1,0,1])

        self.im_Qv.set_extent([0,1,0,1])
        self.im_Qv2.set_extent([0,1,0,1])

        self.im_uv.set_extent([0,1,0,1])
        self.im_uv2.set_extent([0,1,0,1])

        self.ax_Qu.set_aspect(1)
        self.ax_Qu2.set_aspect(1)

        self.ax_Qv.set_aspect(1)
        self.ax_Qv2.set_aspect(1)

        self.ax_uv.set_aspect(1)
        self.ax_uv2.set_aspect(1)

        self.im_Qu_fit.set_array(np.c_[[0,1],[1,0]])
        self.im_Qu2_fit.set_array(np.c_[[0,1],[1,0]])

        self.im_Qv_fit.set_array(np.c_[[0,1],[1,0]])
        self.im_Qv2_fit.set_array(np.c_[[0,1],[1,0]])

        self.im_uv_fit.set_array(np.c_[[0,1],[1,0]])
        self.im_uv2_fit.set_array(np.c_[[0,1],[1,0]])

        self.im_Qu_fit.autoscale()
        self.im_Qu2_fit.autoscale()

        self.im_Qv_fit.autoscale()
        self.im_Qv2_fit.autoscale()

        self.im_uv_fit.autoscale()
        self.im_uv2_fit.autoscale()

        self.im_Qu_fit.set_extent([0,1,0,1])
        self.im_Qu2_fit.set_extent([0,1,0,1])

        self.im_Qv_fit.set_extent([0,1,0,1])
        self.im_Qv2_fit.set_extent([0,1,0,1])

        self.im_uv_fit.set_extent([0,1,0,1])
        self.im_uv2_fit.set_extent([0,1,0,1])

        self.ax_Qu_fit.set_aspect(1)
        self.ax_Qu2_fit.set_aspect(1)

        self.ax_Qv_fit.set_aspect(1)
        self.ax_Qv2_fit.set_aspect(1)

        self.ax_uv_fit.set_aspect(1)
        self.ax_uv2_fit.set_aspect(1)

        # ---

        self.peak_pu.width = 1
        self.peak_pu2.width = 1

        self.inner_pu.width = 1
        self.inner_pu2.width = 1

        self.outer_pu.width = 1
        self.outer_pu2.width = 1

        self.peak_pu.height = 1
        self.peak_pu2.height = 1

        self.inner_pu.height = 1
        self.inner_pu2.height = 1

        self.outer_pu.height = 1
        self.outer_pu2.height = 1

        self.trans_peak_pu.clear()
        self.trans_peak_pu2.clear()

        self.trans_inner_pu.clear()
        self.trans_inner_pu2.clear()

        self.trans_outer_pu.clear()
        self.trans_outer_pu2.clear()

        self.peak_pu.set_transform(self.trans_peak_pu+self.ax_Qu.transData)
        self.peak_pu2.set_transform(self.trans_peak_pu2+self.ax_Qu2.transData)

        self.inner_pu.set_transform(self.trans_inner_pu+self.ax_Qu.transData)
        self.inner_pu2.set_transform(self.trans_inner_pu2+self.ax_Qu2.transData)

        self.outer_pu.set_transform(self.trans_outer_pu+self.ax_Qu.transData)
        self.outer_pu2.set_transform(self.trans_outer_pu2+self.ax_Qu2.transData)

        # ---

        self.peak_pv.width = 1
        self.peak_pv2.width = 1

        self.inner_pv.width = 1
        self.inner_pv2.width = 1

        self.outer_pv.width = 1
        self.outer_pv2.width = 1

        self.peak_pv.height = 1
        self.peak_pv2.height = 1

        self.inner_pv.height = 1
        self.inner_pv2.height = 1

        self.outer_pv.height = 1
        self.outer_pv2.height = 1

        self.trans_peak_pv.clear()
        self.trans_peak_pv2.clear()

        self.trans_inner_pv.clear()
        self.trans_inner_pv2.clear()

        self.trans_outer_pv.clear()
        self.trans_outer_pv2.clear()

        self.peak_pv.set_transform(self.trans_peak_pv+self.ax_Qv.transData)
        self.peak_pv2.set_transform(self.trans_peak_pv2+self.ax_Qv2.transData)

        self.inner_pv.set_transform(self.trans_inner_pv+self.ax_Qv.transData)
        self.inner_pv2.set_transform(self.trans_inner_pv2+self.ax_Qv2.transData)

        self.outer_pv.set_transform(self.trans_outer_pv+self.ax_Qv.transData)
        self.outer_pv2.set_transform(self.trans_outer_pv2+self.ax_Qv2.transData)

        # ---

        self.peak_uv.width = 1
        self.peak_uv2.width = 1

        self.inner_uv.width = 1
        self.inner_uv2.width = 1

        self.outer_uv.width = 1
        self.outer_uv2.width = 1

        self.peak_uv.height = 1
        self.peak_uv2.height = 1

        self.inner_uv.height = 1
        self.inner_uv2.height = 1

        self.outer_uv.height = 1
        self.outer_uv2.height = 1

        self.trans_peak_uv.clear()
        self.trans_peak_uv2.clear()

        self.trans_inner_uv.clear()
        self.trans_inner_uv2.clear()

        self.trans_outer_uv.clear()
        self.trans_outer_uv2.clear()

        self.peak_uv.set_transform(self.trans_peak_uv+self.ax_uv.transData)
        self.peak_uv2.set_transform(self.trans_peak_uv2+self.ax_uv2.transData)

        self.inner_uv.set_transform(self.trans_inner_uv+self.ax_uv.transData)
        self.inner_uv2.set_transform(self.trans_inner_uv2+self.ax_uv2.transData)

        self.outer_uv.set_transform(self.trans_outer_uv+self.ax_uv.transData)
        self.outer_uv2.set_transform(self.trans_outer_uv2+self.ax_uv2.transData)

        # ---

        self.peak_pu_fit.width = 1
        self.peak_pu2_fit.width = 1

        self.inner_pu_fit.width = 1
        self.inner_pu2_fit.width = 1

        self.outer_pu_fit.width = 1
        self.outer_pu2_fit.width = 1

        self.peak_pu_fit.height = 1
        self.peak_pu2_fit.height = 1

        self.inner_pu_fit.height = 1
        self.inner_pu2_fit.height = 1

        self.outer_pu_fit.height = 1
        self.outer_pu2_fit.height = 1

        self.trans_peak_pu_fit.clear()
        self.trans_peak_pu2_fit.clear()

        self.trans_inner_pu_fit.clear()
        self.trans_inner_pu2_fit.clear()

        self.trans_outer_pu_fit.clear()
        self.trans_outer_pu2_fit.clear()

        self.peak_pu_fit.set_transform(self.trans_peak_pu_fit+self.ax_Qu_fit.transData)
        self.peak_pu2_fit.set_transform(self.trans_peak_pu2_fit+self.ax_Qu2_fit.transData)

        self.inner_pu_fit.set_transform(self.trans_inner_pu_fit+self.ax_Qu_fit.transData)
        self.inner_pu2_fit.set_transform(self.trans_inner_pu2_fit+self.ax_Qu2_fit.transData)

        self.outer_pu_fit.set_transform(self.trans_outer_pu_fit+self.ax_Qu_fit.transData)
        self.outer_pu2_fit.set_transform(self.trans_outer_pu2_fit+self.ax_Qu2_fit.transData)

        # ---

        self.peak_pv_fit.width = 1
        self.peak_pv2_fit.width = 1

        self.inner_pv_fit.width = 1
        self.inner_pv2_fit.width = 1

        self.outer_pv_fit.width = 1
        self.outer_pv2_fit.width = 1

        self.peak_pv_fit.height = 1
        self.peak_pv2_fit.height = 1

        self.inner_pv_fit.height = 1
        self.inner_pv2_fit.height = 1

        self.outer_pv_fit.height = 1
        self.outer_pv2_fit.height = 1

        self.trans_peak_pv_fit.clear()
        self.trans_peak_pv2_fit.clear()

        self.trans_inner_pv_fit.clear()
        self.trans_inner_pv2_fit.clear()

        self.trans_outer_pv_fit.clear()
        self.trans_outer_pv2_fit.clear()

        self.peak_pv_fit.set_transform(self.trans_peak_pv_fit+self.ax_Qv_fit.transData)
        self.peak_pv2_fit.set_transform(self.trans_peak_pv2_fit+self.ax_Qv2_fit.transData)

        self.inner_pv_fit.set_transform(self.trans_inner_pv_fit+self.ax_Qv_fit.transData)
        self.inner_pv2_fit.set_transform(self.trans_inner_pv2_fit+self.ax_Qv2_fit.transData)

        self.outer_pv_fit.set_transform(self.trans_outer_pv_fit+self.ax_Qv_fit.transData)
        self.outer_pv2_fit.set_transform(self.trans_outer_pv2_fit+self.ax_Qv2_fit.transData)

        # ---

        self.peak_uv_fit.width = 1
        self.peak_uv2_fit.width = 1

        self.inner_uv_fit.width = 1
        self.inner_uv2_fit.width = 1

        self.outer_uv_fit.width = 1
        self.outer_uv2_fit.width = 1

        self.peak_uv_fit.height = 1
        self.peak_uv2_fit.height = 1

        self.inner_uv_fit.height = 1
        self.inner_uv2_fit.height = 1

        self.outer_uv_fit.height = 1
        self.outer_uv2_fit.height = 1

        self.trans_peak_uv_fit.clear()
        self.trans_peak_uv2_fit.clear()

        self.trans_inner_uv_fit.clear()
        self.trans_inner_uv2_fit.clear()

        self.trans_outer_uv_fit.clear()
        self.trans_outer_uv2_fit.clear()

        self.peak_uv_fit.set_transform(self.trans_peak_uv_fit+self.ax_uv_fit.transData)
        self.peak_uv2_fit.set_transform(self.trans_peak_uv2_fit+self.ax_uv2_fit.transData)

        self.inner_uv_fit.set_transform(self.trans_inner_uv_fit+self.ax_uv_fit.transData)
        self.inner_uv2_fit.set_transform(self.trans_inner_uv2_fit+self.ax_uv2_fit.transData)

        self.outer_uv_fit.set_transform(self.trans_outer_uv_fit+self.ax_uv_fit.transData)
        self.outer_uv2_fit.set_transform(self.trans_outer_uv2_fit+self.ax_uv2_fit.transData)

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

        z[~np.isfinite(z)] = np.nan
        z0[~np.isfinite(z0)] = np.nan

        if np.any(z > 0) and np.any(z0 > 0) and np.diff(x_extents) > 0 and np.diff(y_extents) > 0 and np.all(sigma):

            self.im_p_proj.set_array(z0.T)
            self.im_s_proj.set_array(z.T)

            self.im_p_proj.autoscale()
            self.im_s_proj.autoscale()

            extents = [*x_extents, *y_extents]

            self.im_p_proj.set_extent(extents)
            self.im_s_proj.set_extent(extents)

            #self.ax_s_proj.set_title('\u03A7\u00B2 = {:.4f}'.format(chi_sq))

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

    def plot_extracted_projection(self, z, z0, x_extents, y_extents, mu, sigma, rho, chi_sq):

        z[z <= 0] = np.nan
        z0[z0 <= 0] = np.nan

        z[~np.isfinite(z)] = np.nan
        z0[~np.isfinite(z0)] = np.nan

        if np.any(z > 0) and np.any(z0 > 0)and np.diff(x_extents) > 0 and np.diff(y_extents) > 0 and np.all(sigma):

            self.im_p2_proj.set_array(z0.T)
            self.im_s2_proj.set_array(z.T)

            self.im_p2_proj.autoscale()
            self.im_s2_proj.autoscale()

            extents = [*x_extents, *y_extents]

            self.im_p2_proj.set_extent(extents)
            self.im_s2_proj.set_extent(extents)

            self.ax_p2_proj.set_title('\u03A7\u00B2 = {:.4f}'.format(chi_sq))

            mu_x, mu_y = mu
            sigma_x, sigma_y = sigma

            rx = np.sqrt(1+rho)
            ry = np.sqrt(1-rho)

            scale_x = 3*sigma_x
            scale_y = 3*sigma_y

            self.elli_p2.width = 2*rx
            self.elli_s2.width = 2*rx

            self.elli_p2.height = 2*ry
            self.elli_s2.height = 2*ry

            self.trans_elli_p2.rotate_deg(45).scale(scale_x,scale_y).translate(mu_x,mu_y)
            self.trans_elli_s2.rotate_deg(45).scale(scale_x,scale_y).translate(mu_x,mu_y)

            self.elli_p2.set_transform(self.trans_elli_p2+self.ax_p2_proj.transData)
            self.elli_s2.set_transform(self.trans_elli_s2+self.ax_s2_proj.transData)

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

    def update_ellipse2(self, mu, sigma, rho):

        mu_x, mu_y = mu
        sigma_x, sigma_y = sigma

        rx = np.sqrt(1+rho)
        ry = np.sqrt(1-rho)

        scale_x = 3*sigma_x
        scale_y = 3*sigma_y

        self.trans_elli_s2.clear()

        self.elli_s2.width = 2*rx

        self.elli_s2.height = 2*ry

        self.trans_elli_s2.rotate_deg(45).scale(scale_x,scale_y).translate(mu_x,mu_y)

        self.elli_s2.set_transform(self.trans_elli_s2+self.ax_s2_proj.transData)

    def plot_integration(self, signal, u_extents, v_extents, Q_extents, centers, radii, scales):

        self.trans_peak_pu.clear()
        self.trans_inner_pu.clear()
        self.trans_outer_pu.clear()

        self.trans_peak_pv.clear()
        self.trans_inner_pv.clear()
        self.trans_outer_pv.clear()

        self.trans_peak_uv.clear()
        self.trans_inner_uv.clear()
        self.trans_outer_uv.clear()

        self.trans_peak_pu_fit.clear()
        self.trans_inner_pu_fit.clear()
        self.trans_outer_pu_fit.clear()

        self.trans_peak_pv_fit.clear()
        self.trans_inner_pv_fit.clear()
        self.trans_outer_pv_fit.clear()

        self.trans_peak_uv_fit.clear()
        self.trans_inner_uv_fit.clear()
        self.trans_outer_uv_fit.clear()

        Qu = np.nansum(signal, axis=1)
        Qv = np.nansum(signal, axis=0)
        uv = np.nansum(signal, axis=2).T

        Qu[Qu <= 0] = np.nan
        Qv[Qv <= 0] = np.nan
        uv[uv <= 0] = np.nan

        Qu[~np.isfinite(Qu)] = np.nan
        Qv[~np.isfinite(Qv)] = np.nan
        uv[~np.isfinite(uv)] = np.nan

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

    def plot_extracted_integration(self, signal, u_extents, v_extents, Q_extents, centers, radii, scales):

        self.trans_peak_pu2.clear()
        self.trans_inner_pu2.clear()
        self.trans_outer_pu2.clear()

        self.trans_peak_pv2.clear()
        self.trans_inner_pv2.clear()
        self.trans_outer_pv2.clear()

        self.trans_peak_uv2.clear()
        self.trans_inner_uv2.clear()
        self.trans_outer_uv2.clear()

        self.trans_peak_pu2_fit.clear()
        self.trans_inner_pu2_fit.clear()
        self.trans_outer_pu2_fit.clear()

        self.trans_peak_pv2_fit.clear()
        self.trans_inner_pv2_fit.clear()
        self.trans_outer_pv2_fit.clear()

        self.trans_peak_uv2_fit.clear()
        self.trans_inner_uv2_fit.clear()
        self.trans_outer_uv2_fit.clear()

        Qu = np.nansum(signal, axis=1)
        Qv = np.nansum(signal, axis=0)
        uv = np.nansum(signal, axis=2).T

        Qu[Qu <= 0] = np.nan
        Qv[Qv <= 0] = np.nan
        uv[uv <= 0] = np.nan

        Qu[~np.isfinite(Qu)] = np.nan
        Qv[~np.isfinite(Qv)] = np.nan
        uv[~np.isfinite(uv)] = np.nan

        if np.any(Qu > 0) and np.any(Qv > 0) and np.any(uv > 0) and np.diff(u_extents[0::2]) > 0 and np.diff(v_extents[0::2]) > 0 and np.diff(Q_extents[0::2]) > 0:

            if np.nanmax(Qu)-np.nanmin(Qu) > 0 and np.nanmax(Qv)-np.nanmin(Qv) > 0 and np.nanmax(uv)-np.nanmin(uv) > 0:

                self.im_Qu2.set_array(Qu)
                self.im_Qv2.set_array(Qv)
                self.im_uv2.set_array(uv)

                self.im_Qu2.autoscale()
                self.im_Qv2.autoscale()
                self.im_uv2.autoscale()

                clim_Qu = self.im_Qu2.get_clim()
                clim_Qv = self.im_Qv2.get_clim()
                clim_uv = self.im_uv2.get_clim()

                self.im_Qu2_fit.set_clim(*clim_Qu)
                self.im_Qv2_fit.set_clim(*clim_Qv)
                self.im_uv2_fit.set_clim(*clim_uv)

            Qu_extents = [Q_extents[0],Q_extents[2],u_extents[0],u_extents[2]]
            Qv_extents = [Q_extents[0],Q_extents[2],v_extents[0],v_extents[2]]
            uv_extents = [u_extents[0],u_extents[2],v_extents[0],v_extents[2]]

            self.im_Qu2.set_extent(Qu_extents)
            self.im_Qv2.set_extent(Qv_extents)
            self.im_uv2.set_extent(uv_extents)

            self.im_Qu2_fit.set_extent(Qu_extents)
            self.im_Qv2_fit.set_extent(Qv_extents)
            self.im_uv2_fit.set_extent(uv_extents)

            Qu_aspect = Q_extents[1]/u_extents[1]
            Qv_aspect = Q_extents[1]/v_extents[1]
            uv_aspect = u_extents[1]/v_extents[1]

            self.ax_Qu2.set_aspect(Qu_aspect)
            self.ax_Qv2.set_aspect(Qv_aspect)
            self.ax_uv2.set_aspect(uv_aspect)

            self.ax_Qu2_fit.set_aspect(Qu_aspect)
            self.ax_Qv2_fit.set_aspect(Qv_aspect)
            self.ax_uv2_fit.set_aspect(uv_aspect)

            u_center, v_center, Q_center = centers

            u_size = 2*radii[0]
            v_size = 2*radii[1]
            Q_size = 2*radii[2]

            pk_scale, in_scale, out_scale = scales

            # --

            self.peak_pu2.width = Q_size*pk_scale
            self.inner_pu2.width = Q_size*in_scale
            self.outer_pu2.width = Q_size*out_scale

            self.peak_pu2.height = u_size*pk_scale
            self.inner_pu2.height = u_size*in_scale
            self.outer_pu2.height = u_size*out_scale

            self.trans_peak_pu2.rotate_deg(0).scale(1,1).translate(Q_center,u_center)
            self.trans_inner_pu2.rotate_deg(0).scale(1,1).translate(Q_center,u_center)
            self.trans_outer_pu2.rotate_deg(0).scale(1,1).translate(Q_center,u_center)

            self.peak_pu2.set_transform(self.trans_peak_pu2+self.ax_Qu2.transData)
            self.inner_pu2.set_transform(self.trans_inner_pu2+self.ax_Qu2.transData)
            self.outer_pu2.set_transform(self.trans_outer_pu2+self.ax_Qu2.transData)

            # ---

            self.peak_pv2.width = Q_size*pk_scale
            self.inner_pv2.width = Q_size*in_scale
            self.outer_pv2.width = Q_size*out_scale

            self.peak_pv2.height = v_size*pk_scale
            self.inner_pv2.height = v_size*in_scale
            self.outer_pv2.height = v_size*out_scale

            self.trans_peak_pv2.rotate_deg(0).scale(1,1).translate(Q_center,v_center)
            self.trans_inner_pv2.rotate_deg(0).scale(1,1).translate(Q_center,v_center)
            self.trans_outer_pv2.rotate_deg(0).scale(1,1).translate(Q_center,v_center)

            self.peak_pv2.set_transform(self.trans_peak_pv2+self.ax_Qv2.transData)
            self.inner_pv2.set_transform(self.trans_inner_pv2+self.ax_Qv2.transData)
            self.outer_pv2.set_transform(self.trans_outer_pv2+self.ax_Qv2.transData)

            # ---

            self.peak_uv2.width = u_size*pk_scale
            self.inner_uv2.width = u_size*in_scale
            self.outer_uv2.width = u_size*out_scale

            self.peak_uv2.height = v_size*pk_scale
            self.inner_uv2.height = v_size*in_scale
            self.outer_uv2.height = v_size*out_scale

            self.trans_peak_uv2.rotate_deg(0).scale(1,1).translate(u_center,v_center)
            self.trans_inner_uv2.rotate_deg(0).scale(1,1).translate(u_center,v_center)
            self.trans_outer_uv2.rotate_deg(0).scale(1,1).translate(u_center,v_center)

            self.peak_uv2.set_transform(self.trans_peak_uv2+self.ax_uv2.transData)
            self.inner_uv2.set_transform(self.trans_inner_uv2+self.ax_uv2.transData)
            self.outer_uv2.set_transform(self.trans_outer_uv2+self.ax_uv2.transData)

            # --

            self.peak_pu2_fit.width = Q_size*pk_scale
            self.inner_pu2_fit.width = Q_size*in_scale
            self.outer_pu2_fit.width = Q_size*out_scale

            self.peak_pu2_fit.height = u_size*pk_scale
            self.inner_pu2_fit.height = u_size*in_scale
            self.outer_pu2_fit.height = u_size*out_scale

            self.trans_peak_pu2_fit.rotate_deg(0).scale(1,1).translate(Q_center,u_center)
            self.trans_inner_pu2_fit.rotate_deg(0).scale(1,1).translate(Q_center,u_center)
            self.trans_outer_pu2_fit.rotate_deg(0).scale(1,1).translate(Q_center,u_center)

            self.peak_pu2_fit.set_transform(self.trans_peak_pu2_fit+self.ax_Qu2_fit.transData)
            self.inner_pu2_fit.set_transform(self.trans_inner_pu2_fit+self.ax_Qu2_fit.transData)
            self.outer_pu2_fit.set_transform(self.trans_outer_pu2_fit+self.ax_Qu2_fit.transData)

            # ---

            self.peak_pv2_fit.width = Q_size*pk_scale
            self.inner_pv2_fit.width = Q_size*in_scale
            self.outer_pv2_fit.width = Q_size*out_scale

            self.peak_pv2_fit.height = v_size*pk_scale
            self.inner_pv2_fit.height = v_size*in_scale
            self.outer_pv2_fit.height = v_size*out_scale

            self.trans_peak_pv2_fit.rotate_deg(0).scale(1,1).translate(Q_center,v_center)
            self.trans_inner_pv2_fit.rotate_deg(0).scale(1,1).translate(Q_center,v_center)
            self.trans_outer_pv2_fit.rotate_deg(0).scale(1,1).translate(Q_center,v_center)

            self.peak_pv2_fit.set_transform(self.trans_peak_pv2_fit+self.ax_Qv2_fit.transData)
            self.inner_pv2_fit.set_transform(self.trans_inner_pv2_fit+self.ax_Qv2_fit.transData)
            self.outer_pv2_fit.set_transform(self.trans_outer_pv2_fit+self.ax_Qv2_fit.transData)

            # ---

            self.peak_uv2_fit.width = u_size*pk_scale
            self.inner_uv2_fit.width = u_size*in_scale
            self.outer_uv2_fit.width = u_size*out_scale

            self.peak_uv2_fit.height = v_size*pk_scale
            self.inner_uv2_fit.height = v_size*in_scale
            self.outer_uv2_fit.height = v_size*out_scale

            self.trans_peak_uv2_fit.rotate_deg(0).scale(1,1).translate(u_center,v_center)
            self.trans_inner_uv2_fit.rotate_deg(0).scale(1,1).translate(u_center,v_center)
            self.trans_outer_uv2_fit.rotate_deg(0).scale(1,1).translate(u_center,v_center)

            self.peak_uv2_fit.set_transform(self.trans_peak_uv2_fit+self.ax_uv2_fit.transData)
            self.inner_uv2_fit.set_transform(self.trans_inner_uv2_fit+self.ax_uv2_fit.transData)
            self.outer_uv2_fit.set_transform(self.trans_outer_uv2_fit+self.ax_uv2_fit.transData)

            if self.__show_plots: self.fig.show()

    def plot_fitting(self, signal, I, sig, chi_sq):

        Qu = np.nansum(signal, axis=1)
        Qv = np.nansum(signal, axis=0)
        uv = np.nansum(signal, axis=2).T

        Qu[Qu <= 0] = np.nan
        Qv[Qv <= 0] = np.nan
        uv[uv <= 0] = np.nan

        if np.any(Qu > 0) and np.any(Qv > 0) and np.any(uv > 0):

            if np.nanmax(Qu)-np.nanmin(Qu) > 0 and np.nanmax(Qv)-np.nanmin(Qv) > 0 and np.nanmax(uv)-np.nanmin(uv) > 0:

                self.im_Qu_fit.set_array(Qu)
                self.im_Qv_fit.set_array(Qv)
                self.im_uv_fit.set_array(uv)

            self.ax_uv.set_title('I = {:.2e} \u00B1 {:.2e} [arb. unit]'.format(I[0], sig[0]))
            self.ax_uv_fit.set_title('I = {:.2e} \u00B1 {:.2e} [arb. unit]'.format(I[1], sig[1]))

            self.ax_Qv_fit.set_title('\u03A7\u00B2 = {:.4f}'.format(chi_sq))

            if self.__show_plots: self.fig.show()

    def plot_extracted_fitting(self, signal, I, sig, chi_sq):

        Qu = np.nansum(signal, axis=1)
        Qv = np.nansum(signal, axis=0)
        uv = np.nansum(signal, axis=2).T

        Qu[Qu <= 0] = np.nan
        Qv[Qv <= 0] = np.nan
        uv[uv <= 0] = np.nan

        if np.any(Qu > 0) and np.any(Qv > 0) and np.any(uv > 0):

            if np.nanmax(Qu)-np.nanmin(Qu) > 0 and np.nanmax(Qv)-np.nanmin(Qv) > 0 and np.nanmax(uv)-np.nanmin(uv) > 0:

                self.im_Qu2_fit.set_array(Qu)
                self.im_Qv2_fit.set_array(Qv)
                self.im_uv2_fit.set_array(uv)

            self.ax_uv2.set_title('I = {:.2e} \u00B1 {:.2e} [arb. unit]'.format(I[0], sig[0]))
            self.ax_uv2_fit.set_title('I = {:.2e} \u00B1 {:.2e} [arb. unit]'.format(I[1], sig[1]))

            self.ax_Qv2_fit.set_title('\u03A7\u00B2 = {:.4f}'.format(chi_sq))

            if self.__show_plots: self.fig.show()

    def update_individual(self, ind, no, lamda, two_theta, az_phi, run, bank):

        if type(lamda) is list:
            self.ax_Qu2.set_title('')
            self.ax_Qv2.set_title('')

            if np.isclose(lamda[0],lamda[1]):
                self.ax_Qu.set_title('\u03BB = {:.3f} \u212B'.format(lamda[0]))
            else:
                self.ax_Qu.set_title('\u03BB = {:.3f}-{:.3f} \u212B'.format(*lamda))

            if np.isclose(two_theta[0],two_theta[1]):
                self.ax_Qu_fit.set_title('2\u03B8 = {:.1f}\u00B0'.format(two_theta[0]))
            else:
                self.ax_Qu_fit.set_title('2\u03B8 = {:.1f}-{:.1f}\u00B0'.format(*two_theta))

            if np.isclose(az_phi[0],az_phi[1]):
                self.ax_Qu2_fit.set_title('\u03D5 = {:.1f}\u00B0'.format(az_phi[0]))
            else:
                self.ax_Qu2_fit.set_title('\u03D5 = {:.1f}-{:.1f}\u00B0'.format(*az_phi))

            self.ax_Qv.set_title('{} orientations'.format(no))
        else:
            self.ax_Qu2.set_title('{}'.format(run))
            self.ax_Qv2.set_title('Bank {}'.format(bank))

            self.ax_Qu.set_title('\u03BB = {:.3f} \u212B'.format(lamda))
            self.ax_Qu_fit.set_title('2\u03B8 = {:.1f}\u00B0'.format(two_theta))
            self.ax_Qu2_fit.set_title('\u03D5 = {:.1f}\u00B0'.format(az_phi))

            self.ax_Qv.set_title('{}/{}'.format(1+ind,no))

    def write_figure(self, figname):

        try:
            self.fig.savefig(figname, dpi=100, facecolor='white', transparent=False)
        except:
            pass

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

        self.__sat_keys = []
        self.__sat_Q = []

        # ---

        self.__ind_bin_size = []

        self.__ind_pk_data = []
        self.__ind_pk_norm = []

        self.__ind_bkg_data = []
        self.__ind_bkg_norm = []

        self.__ind_mu_x_3d = []
        self.__ind_mu_y_3d = []
        self.__ind_mu_z_3d = []
        self.__ind_sigma_x_3d = []
        self.__ind_sigma_y_3d = []
        self.__ind_sigma_z_3d = []
        self.__ind_rho_yz_3d = []
        self.__ind_rho_xz_3d = []
        self.__ind_rho_xy_3d = []

        self.__ind_pk_Q0 = []
        self.__ind_pk_Q1 = []
        self.__ind_pk_Q2 = []

        self.__ind_bkg_Q0 = []
        self.__ind_bkg_Q1 = []
        self.__ind_bkg_Q2 = []

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

    def get_individual_bin_size(self):

        return self.__ind_bin_size

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

    def get_ext_scale(self):

        if self.__ext_constants is not None:
            return self.__ext_constants
        else:
            return np.ones_like(self.__norm_scale)

    def set_data_scale(self, corr_scale):

        self.__data_scale = np.array(corr_scale)

    def get_data_scale(self):

        #if not hasattr(self, '_PeakInformation__data_scale'):
        #   self.__data_scale = np.ones_like(self.__norm_scale)

        return self.__data_scale

    def set_norm_scale(self, corr_scale):

        self.__norm_scale = np.array(corr_scale)

    def get_norm_scale(self):

        return self.__norm_scale

    def set_ext_scale(self, corr_scale):

        self.__ext_constants = np.array(corr_scale)

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

    def get_merged_wavelength(self):

        return np.mean(self.__wl)

    def get_merged_intensity(self):

        return self.__merge_intensity()

    def get_merged_intensity_error(self, contrib=True):

        return self.__merge_intensity_error(contrib)

    def get_partial_merged_peak_volume_fraction(self, indices):

        return self.__partial_merge_pk_vol_fract(indices)

    def get_partial_merged_background_volume_fraction(self, indices):

        return self.__partial_merge_bkg_vol_fract(indices)

    def get_partial_merged_intensity(self, indices):

        return self.__partial_merge_intensity(indices)

    def get_partial_merged_intensity_error(self, indices, contrib=True):

        return self.__partial_merge_intensity_error(indices, contrib)

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
              'MergedIntensityFit': self.__round(self.__intens_fit,2),
              'MergedSigmaFit': self.__round(self.__sig_fit,2),
              'MergedVolumeRatio': self.__round(self.__merge_pk_bkg_ratio(),2),
              'MergedVolumeFraction': self.__round(self.__merge_pk_vol_fract(),2),
              'Ellispoid': self.__round(self.get_A()[np.triu_indices(3)],2),
              'BinSize': self.__round(self.__bin_size,3),
              'Q': self.__round(self.__Q,3),
              'PeakQFit1d': self.__round(self.__peak_fit,2),
              'PeakQFit2d': self.__round(self.__peak_fit2d,2),
              'PeakBackgroundRatio1d': self.__round(self.__peak_bkg_ratio,2),
              'PeakBackgroundRatio2d': self.__round(self.__peak_bkg_ratio2d,2),
              'PeakScore1D': self.__round(self.__peak_score,2),
              'PeakScore2D': self.__round(self.__peak_score2d,2),
              'Intensity': self.__round(self.__intensity(),2),
              'IntensitySigma': self.__round(self.__intensity_error(),2),
              'Indices': self.__good_intensities(),
              'VolumeRatio': self.__round(self.__pk_bkg_ratio(),2),
              'PeakVolumeFraction': self.__round(self.__pk_vol_fract(),2),
              'BackgroundVolumeFraction': self.__round(self.__bkg_vol_fract(),2),
              'DataScale': self.__round(self.__data_scale,3),
              'NormalizationScale': self.__round(self.__norm_scale,2),
              'PeakDataSum': self.__round(self.__pk_data_sum(),2),
              'PeakNormalizationSum': self.__round(self.__pk_norm_sum(),2),
              'BackgroundDataSum': self.__round(self.__bkg_data_sum(),2),
              'BackgroundNormalizationSum': self.__round(self.__bkg_norm_sum(),2),
              'PeakScaleConstant': self.__round(self.get_peak_constant(),2),
              'Wavelength': self.__round(self.__wl,2),
              'ScatteringAngle': self.__round(np.rad2deg(self.__two_theta),2),
              'AzimuthalAngle': self.__round(np.rad2deg(self.__az_phi),2),
              'GoniometerPhiAngle': self.__round(self.__phi,2),
              'GoniometerChiAngle': self.__round(self.__chi,2),
              'GoniometerOmegaAngle': self.__round(self.__omega,2),
              'EstimatedIntensity': self.__round(self.__est_int,2),
              'EstimatedSigma': self.__round(self.__est_int_err,2),
              'IndividualIntensity': self.__round(self.get_individual_intensity(),2),
              'IndividualIntensitySigma': self.__round(self.get_individual_intensity_error(),2),
              'IndividualFittedIntensity': self.__round(self.get_individual_fitted_intensity(),2),
              'IndividualFittedIntensitySigma': self.__round(self.get_individual_fitted_intensity_error(),2),
              'IndividualBinSize': self.__round(self.get_individual_bin_size(),3),
            }

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

    def add_close_satellite(self, key, Q):

        if key not in self.__sat_keys:

            self.__sat_keys.append(key)
            self.__sat_Q.append(Q)

    def add_integration(self, Q, D, W, statistics, data_norm, pk_bkg, cntrs):

        peak_fit, peak_bkg_ratio, sig_noise_ratio, peak_fit2d, peak_bkg_ratio2d, sig_noise_ratio2d = statistics

        self.__Q = Q

        self.__D = D
        self.__W = W

        pk_data, pk_norm, bkg_data, bkg_norm, bin_size = pk_bkg
        Q0, Q1, Q2, data, norm = data_norm

        pk_Q0, pk_Q1, pk_Q2, bkg_Q0, bkg_Q1, bkg_Q2 = cntrs

        self.__pk_Q0 = pk_Q0
        self.__pk_Q1 = pk_Q1
        self.__pk_Q2 = pk_Q2

        self.__bkg_Q0 = bkg_Q0
        self.__bkg_Q1 = bkg_Q1
        self.__bkg_Q2 = bkg_Q2

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

    def add_individual_integration(self, pk_bkg, cntrs):

        pk_data, pk_norm, bkg_data, bkg_norm, bin_size = pk_bkg

        self.__ind_pk_data += pk_data
        self.__ind_pk_norm += pk_norm

        self.__ind_bkg_data += bkg_data
        self.__ind_bkg_norm += bkg_norm

        self.__ind_bin_size.append(bin_size)

        pk_Q0, pk_Q1, pk_Q2, bkg_Q0, bkg_Q1, bkg_Q2 = cntrs

        self.__ind_pk_Q0.append(pk_Q0)
        self.__ind_pk_Q1.append(pk_Q1)
        self.__ind_pk_Q2.append(pk_Q2)

        self.__ind_bkg_Q0.append(bkg_Q0)
        self.__ind_bkg_Q1.append(bkg_Q1)
        self.__ind_bkg_Q2.append(bkg_Q2)

    def update_individual_integration(self, pk_bkg, cntrs):

        pk_data, pk_norm, bkg_data, bkg_norm, bin_size = pk_bkg

        self.__ind_pk_data[-1] = pk_data[0]
        self.__ind_pk_norm[-1] = pk_norm[0]

        self.__ind_bkg_data[-1] = bkg_data[0]
        self.__ind_bkg_norm[-1] = bkg_norm[0]

        self.__ind_bin_size[-1] = bin_size

        pk_Q0, pk_Q1, pk_Q2, bkg_Q0, bkg_Q1, bkg_Q2 = cntrs

        self.__ind_pk_Q0[-1] = pk_Q0
        self.__ind_pk_Q1[-1] = pk_Q1
        self.__ind_pk_Q2[-1] = pk_Q2

        self.__ind_bkg_Q0[-1] = bkg_Q0
        self.__ind_bkg_Q1[-1] = bkg_Q1
        self.__ind_bkg_Q2[-1] = bkg_Q2

    def add_fit(self, fit_1d, fit_2d, fit_3d, chi_sq):

        mu_1d, sigma_1d = fit_1d 

        self.__mu_1d = mu_1d
        self.__sigma_1d = sigma_1d

        mu_x_2d, mu_y_2d, sigma_x_2d, sigma_y_2d, rho_2d = fit_2d

        self.__mu_x_2d = mu_x_2d
        self.__mu_y_2d = mu_y_2d
        self.__sigma_x_2d = sigma_x_2d
        self.__sigma_y_2d = sigma_y_2d
        self.__rho_2d = rho_2d

        mu_x_3d, mu_y_3d, mu_z_3d, sigma_x_3d, sigma_y_3d, sigma_z_3d, rho_yz_3d, rho_xz_3d, rho_xy_3d, *delta = fit_3d

        self.__mu_x_3d = mu_x_3d
        self.__mu_y_3d = mu_y_3d
        self.__mu_z_3d = mu_z_3d
        self.__sigma_x_3d = sigma_x_3d
        self.__sigma_y_3d = sigma_y_3d
        self.__sigma_z_3d = sigma_z_3d
        self.__rho_yz_3d = rho_yz_3d
        self.__rho_xz_3d = rho_xz_3d
        self.__rho_xy_3d = rho_xy_3d

        if len(delta) == 0:
            self.__delta = None
            self.__scale = None
        else:
            self.__delta = delta[0]
            self.__scale = delta[1]

        self.__chi_sq = chi_sq

    def add_individual_fit(self, fit_3d):

        mu_x_3d, mu_y_3d, mu_z_3d, sigma_x_3d, sigma_y_3d, sigma_z_3d, rho_yz_3d, rho_xz_3d, rho_xy_3d = fit_3d

        self.__ind_mu_x_3d.append(mu_x_3d)
        self.__ind_mu_y_3d.append(mu_y_3d)
        self.__ind_mu_z_3d.append(mu_z_3d)
        self.__ind_sigma_x_3d.append(sigma_x_3d)
        self.__ind_sigma_y_3d.append(sigma_y_3d)
        self.__ind_sigma_z_3d.append(sigma_z_3d)
        self.__ind_rho_yz_3d.append(rho_yz_3d)
        self.__ind_rho_xz_3d.append(rho_xz_3d)
        self.__ind_rho_xy_3d.append(rho_xy_3d)

    def update_individual_fit(self, fit_3d):

        mu_x_3d, mu_y_3d, mu_z_3d, sigma_x_3d, sigma_y_3d, sigma_z_3d, rho_yz_3d, rho_xz_3d, rho_xy_3d = fit_3d

        self.__ind_mu_x_3d[-1] = mu_x_3d
        self.__ind_mu_y_3d[-1] = mu_y_3d
        self.__ind_mu_z_3d[-1] = mu_z_3d
        self.__ind_sigma_x_3d[-1] = sigma_x_3d
        self.__ind_sigma_y_3d[-1] = sigma_y_3d
        self.__ind_sigma_z_3d[-1] = sigma_z_3d
        self.__ind_rho_yz_3d[-1] = rho_yz_3d
        self.__ind_rho_xz_3d[-1] = rho_xz_3d
        self.__ind_rho_xy_3d[-1] = rho_xy_3d

    def get_fitted_intensity(self):

        return self.__intens_fit

    def get_fitted_intensity_error(self):

        return self.__sig_fit

    def get_individual_fitted_intensity(self):

        return np.array(self.__ind_intens_fit)

    def get_individual_fitted_intensity_error(self):

        return np.array(self.__ind_sig_fit)

    def __covariance_matrix(self, sig0, sig1, sig2, rho12, rho02, rho01):

        sig = np.diag([sig0, sig1, sig2])

        rho = np.array([[1, rho01, rho02],
                        [rho01, 1, rho12],
                        [rho02, rho12, 1]])

        S = np.dot(np.dot(sig, rho), sig)

        return S

    def integrate(self):

        self.__sat_intens_fit = []
        self.__sat_sig_fit = []

        mu0, mu1, mu2 = self.__mu_x_3d, self.__mu_y_3d, self.__mu_z_3d
        sig0, sig1, sig2 = self.__sigma_x_3d, self.__sigma_y_3d, self.__sigma_z_3d
        rho12, rho02, rho01 = self.__rho_yz_3d, self.__rho_xz_3d, self.__rho_xy_3d

        delta = self.__delta
        scale = self.__scale

        S = self.__covariance_matrix(sig0, sig1, sig2, rho12, rho02, rho01)

        if np.linalg.det(S) > 0:
            inv_S = np.linalg.inv(S)
        else:
            inv_S = np.zeros((3,3))
            S = np.eye(3)

        indices = self.__good_intensities()

        if len(indices) == 0:
            self.__intens_fit = 0
            self.__sig_fit = 0

            if delta is not None:
                self.__sat_intens_fit = [0,0]
                self.__sat_sig_fit = [0,0]

            if delta is None:
                return 0, 0, 0, 0, 0
            else:
                return 0, 0, 0, 0, 0, 0, 0

        data = self.__get_peak_data_arrays()[indices]
        norm = self.__get_peak_norm_arrays()[indices]

        bkg_data = self.__get_background_data_arrays()[indices]
        bkg_norm = self.__get_background_norm_arrays()[indices]

        scale_data = self.get_data_scale()[indices]
        scale_norm = self.get_norm_scale()[indices]

        constant = self.get_peak_constant()

        data_norm = np.nansum(data*scale_data[:,np.newaxis], axis=0)/np.nansum(norm*scale_norm[:,np.newaxis], axis=0)
        bkg_data_norm = np.nansum(bkg_data, axis=0)/np.nansum(bkg_norm, axis=0)

        data_norm[~np.isfinite(data_norm)] = np.nan
        bkg_data_norm[~np.isfinite(bkg_data_norm)] = np.nan

        y = np.concatenate((data_norm,bkg_data_norm))

        data_norm = np.nansum(data*scale_data[:,np.newaxis], axis=0)/np.nansum(norm*scale_norm[:,np.newaxis], axis=0)**2
        bkg_data_norm = np.nansum(bkg_data, axis=0)/np.nansum(bkg_norm, axis=0)**2

        data_norm[~np.isfinite(data_norm)] = np.nan
        bkg_data_norm[~np.isfinite(bkg_data_norm)] = np.nan

        e = np.sqrt(np.concatenate((data_norm,bkg_data_norm)))

        data_Q0, data_Q1, data_Q2 = self.__get_peak_bin_centers()
        bkg_data_Q0, bkg_data_Q1, bkg_data_Q2 = self.__get_background_bin_centers()

        Q0 = np.concatenate((data_Q0,bkg_data_Q0))
        Q1 = np.concatenate((data_Q1,bkg_data_Q1))
        Q2 = np.concatenate((data_Q2,bkg_data_Q2))

        x0, x1, x2 = Q0-mu0, Q1-mu1, Q2-mu2

        norm = np.sqrt(np.linalg.det(2*np.pi*S))

        mask = np.isfinite(y) & np.isfinite(e) & (e > 0)

        x = np.exp(-0.5*(inv_S[0,0]*x0**2+inv_S[1,1]*x1**2+inv_S[2,2]*x2**2\
                     +2*(inv_S[1,2]*x1*x2+inv_S[0,2]*x0*x2+inv_S[0,1]*x0*x1)))/norm

        if delta is not None:
            factor = scale**3
            inv_s = inv_S/scale**2
            x_0 = np.exp(-0.5*(inv_s[0,0]*x0**2        +inv_s[1,1]*x1**2        +inv_s[2,2]*(x2-delta)**2\
                           +2*(inv_s[1,2]*x1*(x2-delta)+inv_s[0,2]*x0*(x2-delta)+inv_s[0,1]*x0*x1)))/norm/factor
            x_2 = np.exp(-0.5*(inv_s[0,0]*x0**2        +inv_s[1,1]*x1**2        +inv_s[2,2]*(x2+delta)**2\
                           +2*(inv_s[1,2]*x1*(x2+delta)+inv_s[0,2]*x0*(x2+delta)+inv_s[0,1]*x0*x1)))/norm/factor

        if delta is None:
            A = (np.array([x[mask], np.ones_like(x[mask]), x0[mask], x1[mask], x2[mask]])/e[mask]).T
        else:
            A = (np.array([x_0[mask], x[mask], x_2[mask], np.ones_like(x[mask]), x0[mask], x1[mask], x2[mask]])/e[mask]).T

        b = y[mask]/e[mask]

        if np.sum(mask) > 11 and np.all(np.isfinite(A)) and np.all(np.positive(x[mask])):
            coeff, r, rank, s = np.linalg.lstsq(A, b, rcond=None)
        else:
            if delta is None:
                coeff = [0, 0, 0, 0, 0]
            else:
                coeff = [0, 0, 0, 0, 0, 0, 0]

        if delta is None:
            intens, b, c0, c1, c2 = coeff
        else:
            intens_0, intens, intens_2, b, c0, c1, c2 = coeff

        cov = np.dot(A.T, A)

        if delta is None:
            if np.linalg.det(cov) > 0:
                sig = np.sqrt(np.linalg.inv(cov)[0,0])
            else:
                sig = intens
        else:
            det = np.linalg.det(cov)
            if det > 0 and not np.isclose(det,0):
                inv_cov = np.linalg.inv(cov)
                sig_0, sig, sig_2 = np.sqrt(inv_cov[0,0]), np.sqrt(inv_cov[1,1]), np.sqrt(inv_cov[2,2])
            else:
                sig_0, sig, sig_2 = intens_0, intens, intens_2  

        self.__intens_fit = intens*constant
        self.__sig_fit = sig*constant

        if delta is not None:
            self.__sat_intens_fit = [intens_0*constant, intens_2*constant]
            self.__sat_sig_fit = [sig_0*constant, sig_2*constant]

        if delta is None:
            values = intens, b, c0, c1, c2
        else:
            values = intens_0, intens, intens_2, b, c0, c1, c2

        return values
        
    def individual_integrate(self):

        self.__ind_intens_fit = []
        self.__ind_sig_fit = []

        mu0, mu1, mu2 = self.__ind_mu_x_3d, self.__ind_mu_y_3d, self.__ind_mu_z_3d
        sig0, sig1, sig2 = self.__ind_sigma_x_3d, self.__ind_sigma_y_3d, self.__ind_sigma_z_3d
        rho12, rho02, rho01 = self.__ind_rho_yz_3d, self.__ind_rho_xz_3d, self.__ind_rho_xy_3d
        
        data = self.__get_individual_peak_data_arrays()
        norm = self.__get_individual_peak_norm_arrays()

        bkg_data = self.__get_individual_background_data_arrays()
        bkg_norm = self.__get_individual_background_norm_arrays()

        scale_data = self.get_data_scale()
        scale_norm = self.get_norm_scale()

        constant = self.get_peak_constant()
        
        data_Q0, data_Q1, data_Q2 = self.__get_individual_peak_bin_centers()
        bkg_data_Q0, bkg_data_Q1, bkg_data_Q2 = self.__get_individual_background_bin_centers()
        
        intensities, bs, c0s, c1s, c2s = [], [], [], [], []

        #I_est = self.get_individual_intensity()
        #sig_est = self.get_individual_intensity_error()

        for j in range(len(mu0)):

            S = self.__covariance_matrix(sig0[j], sig1[j], sig2[j], rho12[j], rho02[j], rho01[j])

            if np.linalg.det(S) > 0:
                inv_S = np.linalg.inv(S)
            else:
                inv_S = np.zeros((3,3))
                S = np.eye(3)

            data_norm = data[j]*scale_data[j]/(norm[j]*scale_norm[j])
            bkg_data_norm = bkg_data[j]*scale_data[j]/(bkg_norm[j]*scale_norm[j])

            y = np.concatenate((data_norm,bkg_data_norm))

            data_norm = data[j]*scale_data[j]/(norm[j]*scale_norm[j])**2
            bkg_data_norm = bkg_data[j]*scale_data[j]/(bkg_norm[j]*scale_norm[j])**2

            e = np.sqrt(np.concatenate((data_norm,bkg_data_norm)))

            Q0 = np.concatenate((data_Q0[j],bkg_data_Q0[j]))
            Q1 = np.concatenate((data_Q1[j],bkg_data_Q1[j]))
            Q2 = np.concatenate((data_Q2[j],bkg_data_Q2[j]))

            x0, x1, x2 = Q0-mu0[j], Q1-mu1[j], Q2-mu2[j]

            norm_sig = np.sqrt(np.linalg.det(2*np.pi*S))

            mask = np.isfinite(y) & np.isfinite(e) & (e > 0)

            x = np.exp(-0.5*(inv_S[0,0]*x0**2+inv_S[1,1]*x1**2+inv_S[2,2]*x2**2\
                         +2*(inv_S[1,2]*x1*x2+inv_S[0,2]*x0*x2+inv_S[0,1]*x0*x1)))/norm_sig

            A = (np.array([x[mask], np.ones_like(x[mask])])/e[mask]).T #, x0[mask], x1[mask], x2[mask]
            b = y[mask]/e[mask]

            if np.sum(mask) > 11 and np.all(np.isfinite(A)) and np.all(np.positive(x[mask])):
                coeff, r, rank, s = np.linalg.lstsq(A, b, rcond=None)
            else:
                coeff = [0, 0] #, 0, 0, 0

            # intens, b, c0, c1, c2 = coeff
            intens, b = coeff
            c0, c1, c2 = 0, 0, 0

            cov = np.dot(A.T, A)

            if np.linalg.det(cov) > 0:
                sig = np.sqrt(np.linalg.inv(cov)[0,0])
            else:
                sig = intens
                
            self.__ind_intens_fit.append(intens*constant)
            self.__ind_sig_fit.append(sig*constant)

            intensities.append(intens)
            bs.append(b)
            c0s.append(c0)
            c1s.append(c1)
            c2s.append(c2)

        return intensities, bs, c0s, c1s, c2s

    def get_close_satellite_fit(self):

        return np.array(self.__sat_intens_fit), np.array(self.__sat_sig_fit)

    def get_individual_close_satellite_fit(self):

        return np.array(self.__sat_intens_ind_fit), np.array(self.__sat_sig_ind_fit)

    def set_close_satellites(self, keys, Qs):

        self.__sat_keys = keys 
        self.__sat_Q = Qs

    def get_close_satellites(self):

        return np.array(self.__sat_keys), np.array(self.__sat_Q)

    # ---

    def __merge_pk_vol_fract(self):

        return self.__partial_merge_pk_vol_fract(self.__good_intensities())

    def __merge_bkg_vol_fract(self):

        return self.__partial_merge_bkg_vol_fract(self.__good_intensities())

    def __merge_pk_bkg_ratio(self):

        return self.__partial_merge_pk_bkg_ratio(self.__good_intensities())

    def __merge_intensity(self):

        return self.__partial_merge_intensity(self.__good_intensities())

    def __merge_intensity_error(self, contrib=True):

        return self.__partial_merge_intensity_error(self.__good_intensities(), contrib)

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

    def partial_merge_sum(self, indices):

        if not self.__is_peak_integrated() or len(indices) == 0:

            return 0.0

        else:

            data = self.__get_partial_merged_peak_data_arrays(indices)
            norm = self.__get_partial_merged_peak_norm_arrays(indices)

            data[~np.isfinite(data)] = np.nan
            norm[~np.isfinite(norm)] = np.nan

            return np.nansum(data)/np.nansum(norm)

    def __partial_merge_intensity(self, indices):

        if not self.__is_peak_integrated() or len(indices) == 0:

            return 0.0

        else:

            data = self.__get_partial_merged_peak_data_arrays(indices)
            norm = self.__get_partial_merged_peak_norm_arrays(indices)

            bkg_data = self.__get_partial_merged_background_data_arrays(indices)
            bkg_norm = self.__get_partial_merged_background_norm_arrays(indices)

            scale_data = self.get_partial_merged_data_scale(indices)*self.get_partial_merged_extinction_scale(indices)
            scale_norm = self.get_partial_merged_norm_scale(indices)

            constant = self.get_peak_constant()*np.prod(self.get_bin_size())

            data[data <= 0] = np.nan
            norm[norm <= 0] = np.nan

            bkg_data[bkg_data <= 0] = np.nan
            bkg_norm[bkg_norm <= 0] = np.nan

            data_norm = np.nansum(data*scale_data[:,np.newaxis], axis=0)/np.nansum(norm*scale_norm[:,np.newaxis], axis=0)
            bkg_data_norm = np.nansum(bkg_data*scale_data[:,np.newaxis], axis=0)/np.nansum(bkg_norm*scale_norm[:,np.newaxis], axis=0)

            data_norm[~np.isfinite(data_norm)] = np.nan
            bkg_data_norm[~np.isfinite(bkg_data_norm)] = np.nan

#             if len(bkg_data_norm) > 0:
# 
#                 Q1, Q2, Q3 = np.nanpercentile(bkg_data_norm, [25,50,75])
#                 mask = (bkg_data_norm > Q3) | (bkg_data_norm < Q1)
# 
#                 bkg_data_norm[mask] = np.nan

            pk_vol_fract = np.isfinite(data_norm).sum()/data_norm.size

            pk_vol = np.sum(np.isfinite(data_norm))
            bkg_vol = np.sum(np.isfinite(bkg_data_norm))

            vol_ratio = pk_vol/bkg_vol

            return (np.nansum(data_norm)-vol_ratio*np.nanmean(bkg_data_norm)*bkg_vol)*constant#/pk_vol_fract

    def __partial_merge_intensity_error(self, indices, contrib=True):

        if not self.__is_peak_integrated() or len(indices) == 0:

            return 0.0

        else:

            sig_fit = self.__sig_fit if contrib else 0
            intens_fit = self.__intens_fit if contrib else 0

            data = self.__get_partial_merged_peak_data_arrays(indices)
            norm = self.__get_partial_merged_peak_norm_arrays(indices)

            bkg_data = self.__get_partial_merged_background_data_arrays(indices)
            bkg_norm = self.__get_partial_merged_background_norm_arrays(indices)

            scale_data = self.get_partial_merged_data_scale(indices)*self.get_partial_merged_extinction_scale(indices)
            scale_norm = self.get_partial_merged_norm_scale(indices)

            constant = self.get_peak_constant()*np.prod(self.get_bin_size())

            data[data <= 0] = np.nan
            norm[norm <= 0] = np.nan

            bkg_data[bkg_data <= 0] = np.nan
            bkg_norm[bkg_norm <= 0] = np.nan

            data_norm = np.nansum(data*scale_data[:,np.newaxis], axis=0)/np.nansum(norm*scale_norm[:,np.newaxis], axis=0)**2
            bkg_data_norm = np.nansum(bkg_data*scale_data[:,np.newaxis], axis=0)/np.nansum(bkg_norm*scale_norm[:,np.newaxis], axis=0)**2

            data_norm[~np.isfinite(data_norm)] = np.nan
            bkg_data_norm[~np.isfinite(bkg_data_norm)] = np.nan

#             if len(bkg_data_norm) > 0:
# 
#                 Q1, Q2, Q3 = np.nanpercentile(bkg_data_norm, [25,50,75])
#                 mask = (bkg_data_norm > Q3) | (bkg_data_norm < Q1)
# 
#                 bkg_data_norm[mask] = np.nan

            pk_vol_fract = np.isfinite(data_norm).sum()/data_norm.size

            ind_intens = np.concatenate((self.__intensity()[indices], [intens_fit]))

            if contrib:

#                 if len(ind_intens) > 0:
# 
#                     Q1, Q2, Q3 = np.nanpercentile(ind_intens, [25,50,75])
#                     IQR = Q3-Q1
#                     outlier = (ind_intens > Q3+1.5*IQR) | (ind_intens < Q1-1.5*IQR)
# 
#                     ind_intens[outlier] = np.nan

                var_ind = np.nanvar(ind_intens)

                if np.isfinite(ind_intens).sum() > 1:
                   var_ind /= np.isfinite(ind_intens).sum()

            else:

                var_ind = 0

            pk_vol = np.sum(np.isfinite(data_norm))
            bkg_vol = np.sum(np.isfinite(bkg_data_norm))

            vol_ratio =  pk_vol/bkg_vol

            return np.sqrt((np.nansum(data_norm)+vol_ratio**2*np.nansum(bkg_data_norm))*constant**2+sig_fit**2+var_ind)

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

            pk_vol = np.sum(np.isfinite(data/norm),axis=1)
            bkg_vol = np.sum(np.isfinite(bkg_data/bkg_norm),axis=1)

            return pk_vol/bkg_vol

    def __intensity(self):

        if not self.__is_peak_integrated():

            return np.array([])

        else:

            data = self.__get_peak_data_arrays()
            norm = self.__get_peak_norm_arrays()

            bkg_data = self.__get_background_data_arrays()
            bkg_norm = self.__get_background_norm_arrays()

            scale_data = self.get_data_scale()*self.get_ext_scale()
            scale_norm = self.get_norm_scale()

            pk_vol_fract = self.__pk_vol_fract()

            constant = self.get_peak_constant()*np.prod(self.get_bin_size())

            data_norm = (data*scale_data[:,np.newaxis])/(norm*scale_norm[:,np.newaxis])
            bkg_data_norm = (bkg_data*scale_data[:,np.newaxis])/(bkg_norm*scale_norm[:,np.newaxis])

            data_norm[data_norm <= 0] = np.nan
            bkg_data_norm[bkg_data_norm <= 0] = np.nan

#             if bkg_data_norm.shape[1] > 0:
# 
#                 Q1, Q2, Q3 = np.nanpercentile(bkg_data_norm, [25,50,75], axis=0)
#                 mask = (bkg_data_norm > Q3) | (bkg_data_norm < Q1)
# 
#                 bkg_data_norm[mask] = np.nan
#                 # bkg_data_norm[:,:] = np.nanmean(bkg_data_norm, axis=0)

            intens = np.nansum(data_norm, axis=1)#/pk_vol_fract

            pk_vol = np.sum(np.isfinite(data_norm), axis=1)
            bkg_vol = np.sum(np.isfinite(bkg_data_norm), axis=1)

            bkg_intens = np.nanmean(bkg_data_norm, axis=1)*bkg_vol

            vol_ratio =  pk_vol/bkg_vol

            intensity = (intens-np.multiply(bkg_intens,vol_ratio))*constant#/pk_vol_fract

            return intensity

    def __intensity_error(self):

        if not self.__is_peak_integrated():

            return np.array([])

        else:

            data = self.__get_peak_data_arrays()
            norm = self.__get_peak_norm_arrays()

            bkg_data = self.__get_background_data_arrays()
            bkg_norm = self.__get_background_norm_arrays()

            scale_data = self.get_data_scale()*self.get_ext_scale()
            scale_norm = self.get_norm_scale()

            pk_vol_fract = self.__pk_vol_fract()

            constant = self.get_peak_constant()*np.prod(self.get_bin_size())

            data_norm = (data*scale_data[:,np.newaxis])/(norm*scale_norm[:,np.newaxis])**2
            bkg_data_norm = (bkg_data*scale_data[:,np.newaxis])/(bkg_norm*scale_norm[:,np.newaxis])**2

            data_norm[data_norm <= 0] = np.nan
            bkg_data_norm[bkg_data_norm <= 0] = np.nan

#             if bkg_data_norm.shape[1] > 0:
# 
#                 Q1, Q2, Q3 = np.nanpercentile(bkg_data_norm, [25,50,75], axis=0)
#                 mask = (bkg_data_norm > Q3) | (bkg_data_norm < Q1)
# 
#                 bkg_data_norm[mask] = np.nan
#                 # bkg_data_norm[:,:] = np.nanmean(bkg_data_norm, axis=0)

            intens = np.nansum(data_norm, axis=1)#/pk_vol_fract**2
            bkg_intens = np.nansum(bkg_data_norm, axis=1)

            pk_vol = np.sum(np.isfinite(data_norm), axis=1)
            bkg_vol = np.sum(np.isfinite(bkg_data_norm), axis=1)

            vol_ratio =  pk_vol/bkg_vol

            intensity = np.sqrt(intens+np.multiply(bkg_intens,vol_ratio**2))*constant#/pk_vol_fract

            return intensity

    def get_individual_intensity(self):

        if not self.__is_peak_integrated() or np.prod(np.shape(self.get_individual_bin_size())) == 0:

            return np.array([])

        else:

            data = self.__get_individual_peak_data_arrays()
            norm = self.__get_individual_peak_norm_arrays()

            bkg_data = self.__get_individual_background_data_arrays()
            bkg_norm = self.__get_individual_background_norm_arrays()

            scale_data = self.get_data_scale()
            scale_norm = self.get_norm_scale()

            constant = self.get_peak_constant()*np.prod(self.get_individual_bin_size(), axis=1)

            data_norm = [d*sd/(n*sn) for d, n, sd, sn in zip(data, norm, scale_data, scale_norm)]
            bkg_data_norm = [d*sd/(n*sn) for d, n, sd, sn in zip(bkg_data, bkg_norm, scale_data, scale_norm)]

            intens = np.array([np.nansum(dn) for dn in data_norm])
            #bkg_intens = np.array([np.nanpercentile(dn, 15)*np.isfinite(dn).sum() for dn in bkg_data_norm])
            bkg_intens = np.array([np.nansum(dn) for dn in bkg_data_norm])

            pk_vol = np.array([np.isfinite(dn).sum() for dn in data_norm])
            bkg_vol = np.array([np.isfinite(dn).sum() for dn in bkg_data_norm])

            vol_ratio = pk_vol/bkg_vol

            intensity = (intens-bkg_intens*vol_ratio)*constant

            return intensity

    def get_individual_intensity_error(self):

        if not self.__is_peak_integrated() or np.prod(np.shape(self.get_individual_bin_size())) == 0:

            return np.array([])

        else:

            data = self.__get_individual_peak_data_arrays()
            norm = self.__get_individual_peak_norm_arrays()

            bkg_data = self.__get_individual_background_data_arrays()
            bkg_norm = self.__get_individual_background_norm_arrays()

            scale_data = self.get_data_scale()
            scale_norm = self.get_norm_scale()

            constant = self.get_peak_constant()*np.prod(self.get_individual_bin_size(), axis=1)

            data_norm = [d*sd/(n*sn)**2 for d, n, sd, sn in zip(data, norm, scale_data, scale_norm)]
            bkg_data_norm = [d*sd/(n*sn)**2 for d, n, sd, sn in zip(bkg_data, bkg_norm, scale_data, scale_norm)]

            intens = np.array([np.nansum(dn) for dn in data_norm])
            bkg_intens = np.array([np.nansum(dn) for dn in bkg_data_norm])

            pk_vol = np.array([np.isfinite(dn).sum() for dn in data_norm])
            bkg_vol = np.array([np.isfinite(dn).sum() for dn in bkg_data_norm])

            vol_ratio = pk_vol/bkg_vol

            pk_vol_fract = np.array([np.sum(np.isfinite(dn))/dn.size for dn in data_norm])

            intensity = np.sqrt(intens+bkg_intens*vol_ratio**2)*constant

            return intensity

    def get_individual_peak_volume_fraction(self):

        if not self.__is_peak_integrated() or np.prod(np.shape(self.get_individual_bin_size())) == 0:

            return np.array([])

        else:

            data = self.__get_individual_peak_data_arrays()
            norm = self.__get_individual_peak_norm_arrays()

            data_norm = [d/n for d, n in zip(data, norm)]
            fract = np.array([np.sum(np.isfinite(dn))/dn.size for dn in data_norm])

            return fract

    def get_individual_background_volume_fraction(self):

        if not self.__is_peak_integrated() or np.prod(np.shape(self.get_individual_bin_size())) == 0:

            return np.array([])

        else:

            data = self.__get_individual_background_data_arrays()
            norm = self.__get_individual_background_norm_arrays()

            data_norm = [d/n for d, n in zip(data, norm)]
            fract = np.array([np.sum(np.isfinite(dn))/dn.size for dn in data_norm])

            return fract

    def __pk_data_sum(self, indices=None):

        if not self.__is_peak_integrated():

            return np.array([])

        else:

            data = self.__get_peak_data_arrays()

            scale_data = self.get_data_scale()[:,np.newaxis]*self.get_ext_scale()[:,np.newaxis]

            data_scale = np.multiply(data, scale_data)

            data_norm = data_scale

            data_norm[~np.isfinite(data_norm)] = np.nan

            intens = np.nansum(data_norm, axis=1)

            return intens

    def __pk_norm_sum(self):

        if not self.__is_peak_integrated():

            return np.array([])

        else:

            norm = self.__get_peak_norm_arrays()

            scale_norm = self.get_norm_scale()[:,np.newaxis]

            norm_scale = np.multiply(norm, scale_norm)

            data_norm = norm_scale

            data_norm[~np.isfinite(data_norm)] = np.nan

            intens = np.nansum(data_norm, axis=1)

            return intens

    def __bkg_data_sum(self):

        if not self.__is_peak_integrated():

            return np.array([])

        else:

            data = self.__get_background_data_arrays()

            scale_data = self.get_data_scale()[:,np.newaxis]*self.get_ext_scale()[:,np.newaxis]

            data_scale = np.multiply(data, scale_data)

            data_norm = data_scale

            data_norm[~np.isfinite(data_norm)] = np.nan

            intens = np.nansum(data_norm, axis=1)

            return intens

    def __bkg_norm_sum(self):

        if not self.__is_peak_integrated():

            return np.array([])

        else:

            norm = self.__get_background_norm_arrays()

            scale_norm = self.get_norm_scale()[:,np.newaxis]

            norm_scale = np.multiply(norm, scale_norm)

            data_norm = norm_scale

            data_norm[~np.isfinite(data_norm)] = np.nan

            intens = np.nansum(data_norm, axis=1)

            return intens

    def get_peak_clusters(self, step=0.5):

        indices = self.__good_intensities()

        if len(indices) > 0:

            lamda = self.get_wavelengths()[indices]

            k = 2*np.pi/lamda

            n_orient = k.size

            n_bins = int(round((k.max()-k.min())/step))+1

            if n_bins > 1 and n_bins < n_orient:

                k_count, k_bin = np.histogram(k, bins=n_bins, range=[k.min()-step/2,k.max()+step/2])

                k_indices = np.searchsorted(k_bin, k)

                clusters = [np.argwhere(index == k_indices).flatten() for index in range(1,1+n_bins)]
                clusters = [cluster for cluster in clusters if len(cluster) > 0]
                clusters = [[c for c in cluster if c in indices] for cluster in clusters]

            else:

                clusters = [indices]

        else:

            clusters = [[]]

        return clusters

    def is_peak_integrated(self):

        return self.__is_peak_integrated() and self.__has_good_fit()

    def __is_peak_integrated(self):

        return not (self.__peak_num == 0 or self.__pk_norm is None)

    def prune_peaks(self, min_vol_fract=0.6):

        self.__good_indices = np.arange(len(self.get_wavelengths()))

        if len(self.__ind_bin_size) > 0 and len(self.__ind_bin_size) == len(self.__good_indices):

            intens = np.array(self.get_intensity())
            sig_intens = np.array(self.get_intensity_error())

            #ind_intens = np.array(self.get_individual_intensity())
            #ind_sig_intens = np.array(self.get_individual_intensity_error())

            pk_vol_fract = np.array(self.__pk_vol_fract())

            #fit_ind_intens = np.array(self.get_individual_fitted_intensity())
            #fit_ind_sig_intens = np.array(self.get_individual_fitted_intensity_error())

            indices = np.arange(len(pk_vol_fract))

#             if np.allclose([len(intens),len(sig_intens)],len(pk_vol_fract)): #,len(ind_intens),len(ind_sig_intens),len(fit_ind_intens),len(fit_ind_sig_intens)
# 
#                 mask = (pk_vol_fract > min_vol_fract) & (intens > 0*sig_intens) #& (ind_intens > 3*ind_sig_intens) & (fit_ind_intens > 3*fit_ind_sig_intens) & (sig_intens > 0) & np.isfinite(sig_intens)
# 
#                 indices = indices[mask]
# 
#             else:
# 
#                 indices = np.array([])

            self.__good_indices = indices

    def good_indices(self):

        return self.__good_intensities()

    def __good_intensities(self):

        if not hasattr(self, '_PeakInformation__good_indices'):
            self.prune_peaks()

        return self.__good_indices

    def __dbscan_1d(self, array, eps=0.5):

        sort = np.argsort(array)
        array_sorted = array[sort]

        value = array_sorted[0]
        cluster = [0]

        clusters = []

        for i, val in zip(sort[1:], array_sorted[1:]):
            if val <= value+eps:
                cluster.append(i)
            else:
                clusters.append(cluster)
                cluster = [i]
            value = val

        clusters.append(cluster)

        return clusters

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
                if self.__peak_score2d/self.__peak_score > 3:

                    good = False

                # powder line in projection
                if self.__peak_score/self.__peak_score2d > 3:

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

    def get_partial_merged_extinction_scale(self, indices):

        scale_data = self.get_ext_scale()

        return np.array([scale_data[ind] for ind in indices])

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

    # ---

    def __get_individual_peak_data_arrays(self):

        return self.__ind_pk_data

    def __get_individual_peak_norm_arrays(self):

        return self.__ind_pk_norm

    def __get_individual_background_data_arrays(self):

        return self.__ind_bkg_data

    def __get_individual_background_norm_arrays(self):

        return self.__ind_bkg_norm
        
    def __get_individual_peak_bin_centers(self):

        return self.__ind_pk_Q0, self.__ind_pk_Q1, self.__ind_pk_Q2

    def __get_individual_background_bin_centers(self):

        return self.__ind_bkg_Q0, self.__ind_bkg_Q1, self.__ind_bkg_Q2
    
class PeakDictionary:

    def __init__(self, a=5, b=5, c=5, alpha=90, beta=90, gamma=90, sample=None):

        self.peak_dict = { }

        self.scale_constant = 1e+4

        self.sample_mass = 0
        self.z_parameter = 0
        self.chemical_formula = None

        self.sample_name = sample+'_' if type(sample) is str else ''

        CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace=self.sample_name+'pws')
        CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace=self.sample_name+'iws')
        CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace=self.sample_name+'cws')

        CreateSingleValuedWorkspace(OutputWorkspace=self.sample_name+'nws')

        self.pws = mtd[self.sample_name+'pws']
        self.iws = mtd[self.sample_name+'iws']
        self.cws = mtd[self.sample_name+'cws']
        self.nws = mtd[self.sample_name+'nws']

        self.set_constants(a, b, c, alpha, beta, gamma)
        self.set_satellite_info([0,0,0], [0,0,0], [0,0,0], 0)

        chemical_formula = 'V'
        unit_cell_volume = 27.642
        z_parameter = 2

        vanadium = CrystalStructure('3.0278 3.0278 3.0278', 'I m -3 m', 'V 0 0 0 1.0 0.00605')

        SetSampleMaterial(InputWorkspace=self.nws,
                          ChemicalFormula=chemical_formula,
                          ZParameter=z_parameter,
                          UnitCellVolume=unit_cell_volume,
                          SampleMass=0)

        self.nws.sample().setCrystalStructure(vanadium)

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

        if chemical_formula is not None and z_parameter > 0 and sample_mass > 0:

            self.chemical_formula = chemical_formula
            self.z_parameter = z_parameter
            self.sample_mass = sample_mass

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

        DeleteTableRows(TableWorkspace=self.pws, Rows=range(self.pws.getNumberPeaks()))

        ol = self.pws.sample().getOrientedLattice()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

        self.iws.run().getGoniometer().setR(np.eye(3))
        pk = self.iws.createPeakHKL(V3D(0,0,0))

        for key in self.peak_dict.keys():

            peaks = self.peak_dict.get(key)

            h, k, l, m, n, p = key

            for peak in peaks:

                if peak.is_peak_integrated():
                    peak.individual_integrate()
                    peak.prune_peaks()
                    peak.integrate()

                peak_num = peak.get_peak_number()
                intens = peak.get_merged_intensity()
                sig_intens = peak.get_merged_intensity_error()
                pk_vol_fract = peak.get_merged_peak_volume_fraction()
                bkg_vol_fract = peak.get_merged_background_volume_fraction()

                R = peak.get_goniometers()[0]

                dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)

                pk.setGoniometerMatrix(R)
                pk.setHKL(h+dh,k+dk,l+dl)
                pk.setIntHKL(V3D(h,k,l))
                pk.setIntMNP(V3D(m,n,p))
                pk.setIntensity(intens)
                pk.setSigmaIntensity(sig_intens)
                pk.setPeakNumber(peak_num)
                pk.setBinCount(pk_vol_fract)
                pk.setAbsorptionWeightedPathLength(bkg_vol_fract)
                self.pws.addPeak(pk)

    def add_peaks(self, ws, cluster=False, lamda_min=None, lamda_max=None):

        if mtd.doesExist(ws):

            pws = mtd[ws]

            ol = pws.sample().getOrientedLattice()

            UB = ol.getUB()

            mod_vec_1 = ol.getModVec(0)
            mod_vec_2 = ol.getModVec(1)
            mod_vec_3 = ol.getModVec(2)

            for pn in range(pws.getNumberPeaks()):

                peak = pws.getPeak(pn)
                wl = peak.getWavelength()

                h, k, l = peak.getIntHKL()
                m, n, p = peak.getIntMNP()

                h, k, l, m, n, p = int(h), int(k), int(l), int(m), int(n), int(p)

                if lamda_min is not None and lamda_max is not None:
                    add_peak = (wl > lamda_min) and (wl < lamda_max)
                elif lamda_min is not None:
                    add_peak = wl > lamda_min
                elif lamda_max is not None:
                    add_peak = wl < lamda_max
                else:
                    add_peak = True

                if add_peak:

                    bank = pws.row(pn)['BankName']
                    row = int(pws.row(pn)['Row'])
                    col = int(pws.row(pn)['Col'])

                    intens = peak.getIntensity()
                    sig_intens = peak.getSigmaIntensity()

                    if bank != 'None' and bank != '' and intens > 0 and sig_intens > 0 and intens > sig_intens:

                        # print('Adding peak: ({},{},{},{},{},{})'.format(h,k,l,m,n,p))

                        if cluster and m**2+n**2+p**2 != 0:

                            dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)            

                            sat_Q = 2*np.pi*np.dot(UB, np.array([h+dh,k+dk,l+dl]))
                            sat_key = (h,k,l,m,n,p)

                            key = (h,k,l,0,0,0)
                            h, k, l, m, n, p = key

                        else:

                            key = (h,k,l,m,n,p)
                            sat_key = None

                        run = peak.getRunNumber()

                        bank = int(round(peak.getBinCount())) if bank == 'panel' else int(bank.strip('bank'))
                        ind = peak.getPeakNumber()

                        dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)            

                        Q = 2*np.pi*np.dot(UB, np.array([h+dh,k+dk,l+dl]))
                        #Q = peak.getQSampleFrame()

                        d = 2*np.pi/np.linalg.norm(Q)

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

                        if sat_key is None:

                            Ql = np.dot(R, Q)

                            sign = -1 if config.get('Q.convention') == 'Inelastic' else 1

                            two_theta = 2*np.abs(np.arcsin(Ql[2]/np.linalg.norm(Ql)))
                            az_phi = np.arctan2(sign*Ql[1],sign*Ql[0])

                            self.peak_dict[key][0].add_information(run, bank, ind, row, col, wl, two_theta, az_phi,
                                                                   phi, chi, omega, intens, sig_intens)

                        else:

                            self.peak_dict[key][0].add_close_satellite(sat_key, sat_Q)

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

        keys = list(self.peak_dict.keys())

        for key in keys:

            peaks = self.peak_dict.get(key)

            h, k, l, m, n, p = key

            split = []

            for peak in peaks:

                if len(peak.get_run_numbers()) > 0:

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

                    sat_keys, sat_Qs = peak.get_close_satellites()

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

                                new_peak.set_close_satellites(sat_keys, sat_Qs)

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

                else:

                    self.peak_dict.pop(key, None)

    def clone_peak(self, peak, new_key, Q):

        h, k, l, m, n, p = new_key

        h, k, l, m, n, p = int(h), int(k), int(l), int(m), int(n), int(p)

        key = (h,k,l,m,n,p)

        ol = self.pws.sample().getOrientedLattice()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

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

        varphi = peak.get_rotation_angle()
        u = peak.get_rotation_axis()

        sat_keys, sat_Qs = peak.get_close_satellites()

        split = self.peak_dict.get(key)
        if split is None:
            split = []

        new_peak = PeakInformation(self.scale_constant)

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

        new_peak.set_close_satellites([], [])

        new_peak.set_rows(rows)
        new_peak.set_cols(cols)
        new_peak.set_run_numbers(runs)
        new_peak.set_bank_numbers(banks)
        new_peak.set_peak_indices(indices)
        new_peak.set_wavelengths(wl)
        new_peak.set_scattering_angles(two_theta)
        new_peak.set_azimuthal_angles(az)
        new_peak.set_phi_angles(phi)
        new_peak.set_chi_angles(chi)
        new_peak.set_omega_angles(omega)
        new_peak.set_estimated_intensities(intens)
        new_peak.set_estimated_intensity_errors(sig_intens)

        split.append(new_peak)

        self.peak_dict[key] = split

        peak.set_close_satellites([], [])

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

                if peaks is not None:

                    peak_dict[key] = peaks

        return peak_dict

    def construct_tree(self):

        peak_dict = self.to_be_integrated()

        keys = peak_dict.keys()

        Q_points = []

        for key in keys:

            peaks = peak_dict.get(key)

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

    def integrated_result(self, key, Q, D, W, statistics, data_norm, pkg_bk, cntrs, index=0):

        peaks = self.peak_dict[key]

        peak = peaks[index]
        peak.add_integration(Q, D, W, statistics, data_norm, pkg_bk, cntrs)

    def partial_result(self, key, Q, A, peak_fit, peak_bkg_ratio, peak_score, index=0):

        peaks = self.peak_dict[key]

        peak = peaks[index]
        peak.add_partial_integration(Q, A, peak_fit, peak_bkg_ratio, peak_score)

    def fitted_result(self, key, fit_1d, fit_2d, fit_3d, chi_sq, index=0):

        peaks = self.peak_dict[key]

        peak = peaks[index]
        peak.add_fit(fit_1d, fit_2d, fit_3d, chi_sq)

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

    def save_envelopes(self, filename='envelopes.txt', min_sig_noise_ratio=3, min_pk_vol_fract=0.85, min_bkg_vol_fract=0.15):

        SortPeaksWorkspace(self.iws, ColumnNameToSortBy='l', SortAscending=False, OutputWorkspace=self.iws)
        SortPeaksWorkspace(self.iws, ColumnNameToSortBy='k', SortAscending=False, OutputWorkspace=self.iws)
        SortPeaksWorkspace(self.iws, ColumnNameToSortBy='h', SortAscending=False, OutputWorkspace=self.iws)
        SortPeaksWorkspace(self.iws, ColumnNameToSortBy='DSpacing', SortAscending=False, OutputWorkspace=self.iws)

        ol = self.iws.sample().getOrientedLattice()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

        key_set = {}

        for pn in range(self.iws.getNumberPeaks()):

            pk = self.iws.getPeak(pn)

            h, k, l = pk.getIntHKL()
            m, n, p = pk.getIntMNP()

            h, k, l, m, n, p = int(h), int(k), int(l), int(m), int(n), int(p)

            key = (h, k, l, m, n, p)

            key_set[key] = key

        keys = list(key_set.keys())

        env_format = 3*'{:8.3f}'+'{:8.4f}'+3*'{:9.5f}'+3*'{:9.2f}'+9*'{:9.5f}'+6*'{:8.4f}'+'{:4.0f}'+'\n'

        with open(filename, 'w') as f:

            hdr_env = ['#      h', '       k', '       l', '    d-sp',
                       '      mu1','      mu2','      mu3',
                       '       D1','       D2','       D3',
                       '      W11','      W12','      W13',
                       '      W21','      W22','      W23',
                       '      W31','      W32','      W33',
                       '  min-wl','  max-wl','  min-tt','  max-tt','  min-az','  max-az','   n']
            fmt_env = 4*'{:8}'+15*'{:9}'+6*'{:8}'+'{:4}'+'\n'

            f.write(fmt_env.format(*hdr_env))

            for key in keys:

                peaks = self.peak_dict.get(key)

                h, k, l, m, n, p = key

                for peak in peaks:

                    if peak.is_peak_integrated():

                        pk_vol_fract = peak.get_merged_peak_volume_fraction()
                        bkg_vol_fract = peak.get_merged_background_volume_fraction()

                        intens = peak.get_merged_intensity()
                        sig_intens = peak.get_merged_intensity_error()
                        
                        lamda = peak.get_wavelengths()
                        az_phi = peak.get_azimuthal_angles()
                        two_theta = peak.get_scattering_angles()

                        if (intens > 0 and sig_intens > 0 and intens > min_sig_noise_ratio*sig_intens and pk_vol_fract > min_pk_vol_fract and bkg_vol_fract > min_bkg_vol_fract):

                            dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)

                            d_spacing = ol.d(V3D(h+dh,k+dk,l+dl))

                            Q = peak.get_Q()
                            D = peak.get_D()
                            W = peak.get_W()

                            f.write(env_format.format(*[h+dh,k+dk,l+dl,d_spacing,*Q,*np.diagonal(D),*W.flatten(),lamda.min(),lamda.max(),two_theta.min(),two_theta.max(),az_phi.min(),az_phi.max(),len(lamda)]))

    def save_hkl(self, filename, min_sig_noise_ratio=3,
                       min_pk_vol_fract=0.85, min_bkg_vol_fract=0.15,
                       adaptive_scale=True, scale=1, cross_terms=False):

        ol = self.iws.sample().getOrientedLattice()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

        max_order = ol.getMaxOrder()

        mod_vecs = []

        for i in range(3):
            x, y, z = ol.getModVec(i)
            if np.linalg.norm([x,y,z]) > 0:
                mod_vecs.append([x,y,z])
        n_mod = len(mod_vecs)

        satellite = True if max_order > 0 else False

        key_set = {}

        for pn in range(self.iws.getNumberPeaks()):

            pk = self.iws.getPeak(pn)

            h, k, l = pk.getIntHKL()
            m, n, p = pk.getIntMNP()

            pk_no = pk.getPeakNumber()

            h, k, l, m, n, p = int(h), int(k), int(l), int(m), int(n), int(p)

            key = (h, k, l, m, n, p)

            if key_set.get(key) is None:
                key_set[key] = [pk_no]
            else:
                key_set[key].append(pk_no)

        for int_type in ['est']: # 'fit'

            hkl_intensity = []

            sat_info = []
            peak_info = []
            wavelength_info = []

            keys = list(key_set.keys())

            I_max = None

            for key in keys:

                peaks = self.peak_dict.get(key)

                for peak in peaks:

                    if peak.is_peak_integrated() and peak.get_peak_number() in key_set[key]:

                        # peak.integrate()

                        pk_vol_fract = peak.get_merged_peak_volume_fraction()
                        bkg_vol_fract = peak.get_merged_background_volume_fraction()

                        lamda = peak.get_merged_wavelength()

                        if int_type == 'est':
                            intens = peak.get_merged_intensity()
                            sig_intens = peak.get_merged_intensity_error()
                            I, sig, indices = [intens], [sig_intens], [key]
                        else:
                            intens = peak.get_fitted_intensity()
                            sig_intens = peak.get_fitted_intensity_error()

                            sat_keys, sat_Qs = peak.get_close_satellites()
                            if len(sat_keys) > 0:
                                sat_intens_fit, sat_sig_fit = peak.get_close_satellite_fit()
                                I, sig, indices = np.concatenate(([intens],sat_intens_fit)), np.concatenate(([sig_intens],sat_sig_fit)), np.concatenate(([key],sat_keys))
                            else:
                                I, sig, indices = [intens], [sig_intens], [key]

                        for intens, sig_intens, (h,k,l,m,n,p) in zip(I, sig, indices):

                            if (intens > 0 and sig_intens > 0 and intens > min_sig_noise_ratio*sig_intens and pk_vol_fract > min_pk_vol_fract and bkg_vol_fract > min_bkg_vol_fract):

                                if int_type == 'est':

                                    if I_max is None:
                                        I_max = intens
                                    elif intens > I_max:
                                        I_max = intens

                                dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)

                                d_spacing = ol.d(V3D(h+dh,k+dk,l+dl))

                                hkl_intensity.append([h,k,l,intens,sig_intens])

                                peak_info.append(d_spacing)
                                wavelength_info.append(lamda)

                                mv = 0
                                mnp = []
                                if n_mod > 0:
                                    mnp.append(m)
                                    mv = 0+(3*m+1)*m//2
                                if n_mod > 1: 
                                    mnp.append(n)
                                    mv = 2+(3*n+1)*n//2
                                if n_mod > 2:
                                    mnp.append(p)
                                    mv = 4+(3*p+1)*p//2
                                mnp.append(mv)

                                sat_info.append(mnp)

            if adaptive_scale:
                if I_max is not None:
                    scale = 1000/I_max
                else:
                    scale = 1

            sort = np.argsort(peak_info)[::-1]

            for i in sort:
                hkl_intensity[i][3] *= scale
                hkl_intensity[i][4] *= scale

            fname, ext = os.path.splitext(filename)

            app = '' if int_type == 'est' else '_fit'

            if not satellite:

                with open(fname+app+'.hkl', 'w') as f:

                    hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:8.4f}\n'
                    for i in sort:
                        f.write(hkl_format.format(*[*hkl_intensity[i],peak_info[i]]))

                with open(fname+app+'.int', 'w') as f:
                    hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:5.0f}{:10.4f}\n'
                    f.write('Single crystal integrated intensity file\n')
                    f.write('(3i4,2f8.2,i5,2f10.4)\n')
                    f.write('  0 0 0\n')
                    f.write('#  h   k   l  Fsqr   s(Fsqr) Cod   DSpace\n')
                    for i in sort:
                        f.write(hkl_format.format(*[*hkl_intensity[i],1,peak_info[i]]))

            else:

                with open(fname+app+'.hkl', 'w') as f:
                    hkl_format = '{:4.0f}{:4.0f}{:4.0f}'\
                               + '{:4.0f}'*n_mod\
                               + '{:8.2f}{:8.2f}{:8.4f}\n'
                    #f.write('# Structural propagation vectors used\n')
                    #f.write('           {}\n'.format(n_mod))
                    #for i, mod_vec in enumerate(mod_vecs):
                    #    x, y, z = mod_vec
                    #    f.write('       {}{: >13.6f}{: >13.6f}{: >13.6f}\n'.format(i+1,x,y,z))
                    #f.write('#  h   k   l')
                    #f.write(''.join(['  m{}'.format(i+1) for i in range(n_mod)]))
                    #f.write('    Fsqr s(Fsqr)       d\n')
                    for i in sort:
                        f.write(hkl_format.format(*[*hkl_intensity[i][:-2],*sat_info[i][:-1],*hkl_intensity[i][-2:],peak_info[i]]))

                with open(fname+app+'_sat.hkl', 'w') as f:
                    hkl_format = '{:4.0f}{:4.0f}{:4.0f}'\
                               + '{:4.0f}'*n_mod\
                               + '{:8.2f}{:8.2f}{:8.4f}\n'
                    #f.write('# Structural propagation vectors used\n')
                    #f.write('           {}\n'.format(n_mod))
                    #for i, mod_vec in enumerate(mod_vecs):
                    #    x, y, z = mod_vec
                    #    f.write('       {}{: >13.6f}{: >13.6f}{: >13.6f}\n'.format(i+1,x,y,z))
                    #f.write('#  h   k   l')
                    #f.write(''.join(['  m{}'.format(i+1) for i in range(n_mod)]))
                    #f.write('    Fsqr s(Fsqr)       d\n')
                    for i in sort:
                        if not np.all(np.array(sat_info[i][:-1]) == 0):
                            f.write(hkl_format.format(*[*hkl_intensity[i][:-2],*sat_info[i][:-1],*hkl_intensity[i][-2:],peak_info[i]]))

                with open(fname+app+'_nuc.hkl', 'w') as f:
                    hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:8.4f}\n'
                    for i in sort:
                        if np.all(np.array(sat_info[i][:-1]) == 0):
                            f.write(hkl_format.format(*[*hkl_intensity[i],peak_info[i]]))

                with open(fname+app+'_nuc.int', 'w') as f:
                    hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:5.0f}{:8.4f}\n'
                    f.write('Single crystal integrated intensity file\n')
                    f.write('(3i4,2f8.2,i5,2f8.2)\n')
                    f.write('  {:.2f} 0 0\n'.format(np.mean(wavelength_info)))
                    f.write('#  h   k   l    Fsqr s(Fsqr)  Cod       d\n')
                    for i in sort:
                        if np.all(np.array(sat_info[i][:-1]) == 0):
                            f.write(hkl_format.format(*[*hkl_intensity[i],1,peak_info[i]]))

                if not cross_terms:
                    with open(fname+app+'_sat.int', 'w') as f:
                        hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:5.0f}{:8.4f}\n'
                        f.write('Single crystal integrated intensity file\n')
                        f.write('(4i4,2f8.2,i5,2f8.2)\n')
                        f.write('  1 0 0\n')
                        f.write('           {}\n'.format(2*n_mod))
                        for i, mod_vec in enumerate(mod_vecs):
                            x, y, z = mod_vec
                            f.write('       {}{: >13.6f}{: >13.6f}{: >13.6f}\n'.format(2*i+1,x,y,z))
                            f.write('       {}{: >13.6f}{: >13.6f}{: >13.6f}\n'.format(2*i+2,-x,-y,-z))
                        f.write('#  h   k   l   m    Fsqr s(Fsqr)  Cod       d\n')
                        for i in sort:
                            if not np.all(np.array(sat_info[i][:-1]) == 0):
                                f.write(hkl_format.format(*[*hkl_intensity[i][:-2],sat_info[i][-1],*hkl_intensity[i][-2:],1,peak_info[i]]))

        # SortPeaksWorkspace(InputWorkspace=self.iws,
        #                    ColumnNameToSortBy='PeakNumber',
        #                    SortAscending=True,
        #                    OutputWorkspace=ws)
        # SortPeaksWorkspace(InputWorkspace=self.cws,
        #                    ColumnNameToSortBy='PeakNumber',
        #                    SortAscending=True,
        #                    OutputWorkspace=ws)

        return scale

    def save_reflections(self, filename, min_sig_noise_ratio=3, min_pk_vol_fract=0.85, min_bkg_vol_fract=0.15, adaptive_scale=True, scale=1, cross_terms=False):

        scale_fit = scale

        UB = self.cws.sample().getOrientedLattice().getUB()

        SortPeaksWorkspace(self.iws, ColumnNameToSortBy='l', SortAscending=False, OutputWorkspace=self.iws)
        SortPeaksWorkspace(self.iws, ColumnNameToSortBy='k', SortAscending=False, OutputWorkspace=self.iws)
        SortPeaksWorkspace(self.iws, ColumnNameToSortBy='h', SortAscending=False, OutputWorkspace=self.iws)
        SortPeaksWorkspace(self.iws, ColumnNameToSortBy='DSpacing', SortAscending=False, OutputWorkspace=self.iws)

        ol = self.iws.sample().getOrientedLattice()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

        max_order = ol.getMaxOrder()

        mod_vecs = []

        for i in range(3):
            x, y, z = ol.getModVec(i)
            if np.linalg.norm([x,y,z]) > 0:
                mod_vecs.append([x,y,z])
        n_mod = len(mod_vecs)

        hkl_intensity = []
        mnp_vector = []
        pk_info_1 = []
        pk_info_2 = []
        sat_info = []
        sup_info = []

        run_bank_dict = {}
        bank_run_dict = {}

        key_dict = {}

        i, j = 0, 0

        key_set = {}

        for pn in range(self.iws.getNumberPeaks()):

            pk = self.iws.getPeak(pn)

            h, k, l = pk.getIntHKL()
            m, n, p = pk.getIntMNP()

            pk_no = pk.getPeakNumber()

            h, k, l, m, n, p = int(h), int(k), int(l), int(m), int(n), int(p)

            key = (h, k, l, m, n, p)

            if key_set.get(key) is None:
                key_set[key] = [pk_no]
            else:
                key_set[key].append(pk_no)

        keys = list(key_set.keys())

        I_max = None

        run_min = None

        for key in keys:

            peaks = self.peak_dict.get(key)

            h, k, l, m, n, p = key

            for peak in peaks:

                if peak.is_peak_integrated() and peak.get_peak_number() in key_set[key]:

                    intens_merge = peak.get_merged_intensity()
                    sig_intens_merge = peak.get_merged_intensity_error()

                    pk_vol_fract_merge = peak.get_merged_peak_volume_fraction()
                    bkg_vol_fract_merge = peak.get_merged_background_volume_fraction()

                    if (intens_merge > 0 and sig_intens_merge > 0 and intens_merge > min_sig_noise_ratio*sig_intens_merge and pk_vol_fract_merge > min_pk_vol_fract and bkg_vol_fract_merge > min_bkg_vol_fract):

                        rows = peak.get_rows()
                        cols = peak.get_cols()

                        runs = peak.get_run_numbers()
                        banks = peak.get_bank_numbers()

                        Q = peak.get_Q()

                        lamda = peak.get_wavelengths()
                        two_theta = peak.get_scattering_angles()
                        az_phi = peak.get_azimuthal_angles()

                        omega = peak.get_omega_angles()
                        chi = peak.get_chi_angles()
                        phi = peak.get_phi_angles()

                        T = peak.get_transmission_coefficient()
                        Tbar = peak.get_weighted_mean_path_length()

                        R = peak.get_goniometers()
                        R = np.array(R)

                        # clusters = peak.get_peak_clusters()

                        ind_intens = peak.get_individual_intensity()
                        ind_sig_intens = peak.get_individual_intensity_error()

                        ind_fit_intens = peak.get_individual_fitted_intensity()
                        ind_fit_sig_intens = peak.get_individual_fitted_intensity_error()

                        ind_pk_vol_fract = peak.get_individual_peak_volume_fraction()
                        ind_bkg_vol_fract = peak.get_individual_background_volume_fraction()

                        intens = peak.get_intensity()
                        sig_intens = peak.get_intensity_error()

                        fit_intens = peak.get_fitted_intensity()
                        fit_sig_intens = peak.get_fitted_intensity_error()

                        pk_vol_fract = peak.get_peak_volume_fraction()
                        bkg_vol_fract = peak.get_background_volume_fraction()

                        n_ind = len(ind_intens)

                        for ind in range(n_ind):

                            sig_prop = np.sqrt(ind_sig_intens[ind]**2+ind_fit_sig_intens[ind]**2+sig_intens_merge**2)

                            good = False
                            if (intens[ind] > 0 and sig_intens[ind] > 0 and intens[ind] > min_sig_noise_ratio*sig_prop and \
                                ind_intens[ind] > 0 and ind_sig_intens[ind] > 0 and ind_intens[ind] > min_sig_noise_ratio*sig_prop and \
                                fit_intens > min_sig_noise_ratio*fit_sig_intens and pk_vol_fract[ind] > min_pk_vol_fract and \
                                ind_fit_intens[ind] > min_sig_noise_ratio*ind_fit_sig_intens[ind] and ind_pk_vol_fract[ind] > min_pk_vol_fract and \
                                bkg_vol_fract[ind] > min_bkg_vol_fract and \
                                ind_bkg_vol_fract[ind] > min_bkg_vol_fract):
                                good = True

                            intensity = ind_intens[ind]

                            if good:
                            #if (intens[ind] > 0 and sig_intens[ind] > 0 and intens[ind] > min_sig_noise_ratio*sig_intens[ind] and pk_vol_fract[ind] > min_pk_vol_fract and bkg_vol_fract[ind] > min_bkg_vol_fract):

                                if I_max is None: 
                                    I_max = intensity#[i].copy()
                                elif intens[ind] > I_max: 
                                    I_max = intensity#[i].copy()

                                ki_norm = np.array([0, 0, 1])
                                kf_norm = np.array([np.cos(az_phi[ind])*np.sin(two_theta[ind]),
                                                    np.sin(az_phi[ind])*np.sin(two_theta[ind]),
                                                    np.cos(two_theta[ind])])

                                # RU = np.dot(R[cluster][0],U)
                                # 
                                # incident = -np.dot(RU.T, ki_norm)
                                # reflected = np.dot(RU.T, kf_norm)

                                t1 = UB[:,0].copy()
                                t2 = UB[:,1].copy()
                                t3 = UB[:,2].copy()

                                t1 /= np.linalg.norm(t1)
                                t2 /= np.linalg.norm(t2)
                                t3 /= np.linalg.norm(t3)

                                up = np.dot(R[ind].T, -ki_norm)
                                us = np.dot(R[ind].T, +kf_norm)

                                incident = np.dot(up,t1), np.dot(up,t2), np.dot(up,t3)
                                reflected = np.dot(us,t1), np.dot(us,t2), np.dot(us,t3)

                                dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)

                                d = ol.d(V3D(h+dh,k+dk,l+dl))

                                hkl_intensity.append([h, k, l, intensity, sig_prop])
                                mnp_vector.append([m, n, p])

                                run = runs[ind]
                                bank = banks[ind]

                                if run_min is None:
                                    run_min = run
                                elif run < run_min: 
                                    run_min = run

                                pk_info_1.append([lamda[ind], Tbar[ind], incident[0], reflected[0], incident[1], reflected[1], incident[2], reflected[2], run])
                                pk_info_2.append([T[ind], bank, two_theta[ind], d, cols[ind], rows[ind]])

                                sup_info.append([az_phi[ind], omega[ind], phi[ind], chi[ind]])

                                mv = 0
                                mnp = []
                                if n_mod > 0:
                                    mnp.append(m)
                                    mv = 0+(3*m+1)*m//2
                                if n_mod > 1: 
                                    mnp.append(n)
                                    mv = 2+(3*n+1)*n//2
                                if n_mod > 2:
                                    mnp.append(p)
                                    mv = 4+(3*p+1)*p//2
                                mnp.append(mv)

                                sat_info.append(mnp)

                                sort_key = (run)

                                if run_bank_dict.get(sort_key) is None:
                                    run_bank_dict[sort_key] = [j]
                                else:
                                    index = run_bank_dict[sort_key]
                                    index.append(j)
                                    run_bank_dict[sort_key] = index

                                sort_key = (bank)

                                if bank_run_dict.get(sort_key) is None:
                                    bank_run_dict[sort_key] = [j]
                                else:
                                    index = bank_run_dict[sort_key]
                                    index.append(j)
                                    bank_run_dict[sort_key] = index

                                sort_key = (h,k,l,m,n,p)

                                if key_dict.get(sort_key) is None:
                                    key_dict[sort_key] = [j]
                                else:
                                    index = key_dict[sort_key]
                                    index.append(j)
                                    key_dict[sort_key] = index

                                j += 1

        if adaptive_scale:
            if I_max is not None:
                scale = 1000/I_max
            else:
                scale = 1

        filename, ext = os.path.splitext(filename)

        with open(filename+'_norm'+ext, 'w') as f:

            if max_order == 0:
                hkl_fmt = 3*'{:4d}'+2*'{:8.2f}'+'{:4d}'+2*'{:8.5f}'+6*'{:9.5f}'+\
                          '{:6d}{:7d}{:7.4f}{:4d}{:9.5f}{:8.4f}'+2*'{:7.2f}'+'\n'
            else:
                hkl_fmt = 6*'{:4d}'+2*'{:8.2f}'+'{:4d}'+2*'{:8.5f}'+6*'{:9.5f}'+\
                          '{:6d}{:7d}{:7.4f}{:4d}{:9.5f}{:8.4f}'+2*'{:7.2f}'+'\n'

            pk_num = 0

            sort = np.argsort([pk_info_2[i][3] for i in range(len(pk_info_2))])[::-1]

            for i in sort:

                seq_num = 1

                pk_info_1[i][-1] -= (run_min-1)

                hkl_intensity[i][3] *= scale
                hkl_intensity[i][4] *= scale

                if max_order == 0:
                    f.write(hkl_fmt.format(*[*hkl_intensity[i], seq_num, *pk_info_1[i], pk_num, *pk_info_2[i]]))
                else:
                    f.write(hkl_fmt.format(*[*hkl_intensity[i][:3], *mnp_vector[i], *hkl_intensity[i][3:], seq_num, *pk_info_1[i], pk_num, *pk_info_2[i]]))

                pk_num += 1

            if max_order > 0:
                f.write(hkl_fmt.format(*[0]*25))
            else:
                f.write(hkl_fmt.format(*[0]*22))

        with open(filename+'_norm.csv', 'w') as f:

            if max_order == 0:
                hkl_fmt = 3*'{:4d},'+2*'{:8.2f},'+6*'{:8.5f},'+'{:8.5f}\n'
            else:
                hkl_fmt = 6*'{:4d},'+2*'{:8.2f},'+6*'{:8.5f},'+'{:8.5f}\n'

            # sort = np.argsort([pk_info_2[i][3] for i in range(len(pk_info_2))])[::-1]

            for i in sort:

                if max_order == 0:
                    f.write(hkl_fmt.format(*[*hkl_intensity[i], pk_info_2[i][3], pk_info_1[i][0], np.rad2deg(pk_info_2[i][2]), np.rad2deg(sup_info[i][0]), *sup_info[i][1:]]))
                else:
                    f.write(hkl_fmt.format(*[*hkl_intensity[i][:3], *mnp_vector[i], *hkl_intensity[i][3:], pk_info_2[i][3], pk_info_1[i][0], np.rad2deg(pk_info_2[i][2]), np.rad2deg(sup_info[i][0]), *sup_info[i][1:]]))

        if max_order == 0:

            with open(filename+'_sn'+ext, 'w') as f:

                hkl_fmt = 3*'{:4d}'+2*'{:8.2f}'+'{:4d}'+2*'{:8.5f}'+6*'{:9.5f}'+\
                          '{:6d}{:7d}{:7.4f}{:4d}{:9.5f}{:8.4f}'+2*'{:7.2f}'+'\n' #(3i4,2f8.2,i4,2f8.5,6f9.5,i6,i7,f7.4,i4,f9.5,f8.4)

                pk_num = 0

                for j, (run) in enumerate(sorted(run_bank_dict.keys())):

                    seq_num = 1

                    for i in run_bank_dict[(run)]:

                        f.write(hkl_fmt.format(*[*hkl_intensity[i], seq_num, *pk_info_1[i], pk_num, *pk_info_2[i]]))

                        pk_num += 1

                f.write(hkl_fmt.format(*[0]*22))

            with open(filename+'_dn'+ext, 'w') as f:

                hkl_fmt = 3*'{:4d}'+2*'{:8.2f}'+'{:4d}'+2*'{:8.5f}'+6*'{:9.5f}'+\
                          '{:6d}{:7d}{:7.4f}{:4d}{:9.5f}{:8.4f}'+2*'{:7.2f}'+'\n'

                pk_num = 0

                for j, (bank) in enumerate(sorted(bank_run_dict.keys())):

                    seq_num = 1

                    for i in bank_run_dict[(bank)]:

                        f.write(hkl_fmt.format(*[*hkl_intensity[i], seq_num, *pk_info_1[i], pk_num, *pk_info_2[i]]))

                        pk_num += 1

                f.write(hkl_fmt.format(*[0]*22))

        else:

            with open(filename+'_norm_nuc.hkl', 'w') as f:
                hkl_fmt = 3*'{:4d}'+2*'{:8.2f}'+'{:4d}'+2*'{:8.5f}'+6*'{:9.5f}'+\
                          '{:6d}{:7d}{:7.4f}{:4d}{:9.5f}{:8.4f}'+2*'{:7.2f}'+'\n'
                pk_num = 0
                for i in sort:
                    if np.all(np.array(sat_info[i][:-1]) == 0):
                        f.write(hkl_fmt.format(*[*hkl_intensity[i], seq_num, *pk_info_1[i], pk_num, *pk_info_2[i]]))
                        pk_num += 1
                f.write(hkl_fmt.format(*[0]*22))

            with open(filename+'_norm_sat.hkl', 'w') as f:
                hkl_fmt = 6*'{:4d}'+2*'{:8.2f}'+'{:4d}'+2*'{:8.5f}'+6*'{:9.5f}'+\
                          '{:6d}{:7d}{:7.4f}{:4d}{:9.5f}{:8.4f}'+2*'{:7.2f}'+'\n'
                pk_num = 0
                for i in sort:
                    if not np.all(np.array(sat_info[i][:-1]) == 0):
                        f.write(hkl_fmt.format(*[*hkl_intensity[i][:3], *mnp_vector[i], *hkl_intensity[i][3:], seq_num, *pk_info_1[i], pk_num, *pk_info_2[i]]))
                        pk_num += 1
                f.write(hkl_fmt.format(*[0]*25))

            with open(filename+'_norm_nuc.int', 'w') as f:
                hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:5.0f}{:8.4f}\n'
                f.write('Single crystal integrated intensity file\n')
                f.write('(3i4,2f8.2,i5,2f8.2)\n')
                f.write('  0 0 0\n')
                f.write('#  h   k   l    Fsqr s(Fsqr)  Cod  Lambda\n')
                for i in sort:
                    if np.all(np.array(sat_info[i][:-1]) == 0):
                        f.write(hkl_format.format(*[*hkl_intensity[i][:-2],*hkl_intensity[i][-2:],1,pk_info_1[i][0]]))

            # sort = np.argsort([pk_info_2[i][3] for i in range(len(pk_info_2))])[::-1]

            if not cross_terms:
                with open(filename+'_norm.int', 'w') as f:
                    hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:5.0f}{:8.4f}\n'
                    f.write('Single crystal integrated intensity file\n')
                    f.write('(4i4,2f8.2,i5,2f8.2)\n')
                    f.write('  0 0 0\n')
                    f.write('           {}\n'.format(2*n_mod))
                    for i, mod_vec in enumerate(mod_vecs):
                        x, y, z = mod_vec
                        f.write('       {}{: >13.6f}{: >13.6f}{: >13.6f}\n'.format(2*i+1,x,y,z))
                        f.write('       {}{: >13.6f}{: >13.6f}{: >13.6f}\n'.format(2*i+2,-x,-y,-z))
                        f.write('#  h   k   l   m    Fsqr s(Fsqr)  Cod  Lambda\n')
                    for i in sort:
                        f.write(hkl_format.format(*[*hkl_intensity[i][:-2],*sat_info[i][:-1],*hkl_intensity[i][-2:],1,pk_info_1[i][0]]))

                with open(filename+'_norm_nuc.csv', 'w') as f:
                    hkl_fmt = 3*'{:4d},'+2*'{:8.2f},'+6*'{:8.5f},'+'{:8.5f}\n'
                    for i in sort:
                        if np.all(np.array(sat_info[i][:-1]) == 0):
                            f.write(hkl_fmt.format(*[*hkl_intensity[i], pk_info_2[i][3], pk_info_1[i][0], np.rad2deg(pk_info_2[i][2]), np.rad2deg(sup_info[i][0]), *sup_info[i][1:]]))

                with open(filename+'_norm_sat.csv', 'w') as f:
                    hkl_fmt = 6*'{:4d},'+2*'{:8.2f},'+6*'{:8.5f},'+'{:8.5f}\n'
                    for i in sort:
                        if not np.all(np.array(sat_info[i][:-1]) == 0):
                            f.write(hkl_fmt.format(*[*hkl_intensity[i][:3], *mnp_vector[i], *hkl_intensity[i][3:], pk_info_2[i][3], pk_info_1[i][0], np.rad2deg(pk_info_2[i][2]), np.rad2deg(sup_info[i][0]), *sup_info[i][1:]]))


        SortPeaksWorkspace(InputWorkspace=self.iws,
                           ColumnNameToSortBy='PeakNumber',
                           SortAscending=True,
                           OutputWorkspace=self.iws)
        SortPeaksWorkspace(InputWorkspace=self.cws,
                           ColumnNameToSortBy='PeakNumber',
                           SortAscending=True,
                           OutputWorkspace=self.cws)

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

        n_pks = cal.getNumberPeaks()

        if n_pks > 20:

            ol = self.iws.sample().getOrientedLattice()

            mod_vec_1 = ol.getModVec(0)
            mod_vec_2 = ol.getModVec(1)
            mod_vec_3 = ol.getModVec(2)

            max_order = ol.getMaxOrder()

            Q, hkl = [], []

            for pn in range(n_pks):

                pk = cal.getPeak(pn)

                #h, k, l = pk.getIntHKL()
                #m, n, p = pk.getIntMNP()

                #h, k, l, m, n, p = int(h), int(k), int(l), int(m), int(n), int(p)

                #dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)

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
            # elif np.allclose([alpha, beta], 90):
            #     fun = self.__mono1
            #     x0 = (a, b, c, np.deg2rad(gamma))
            elif np.allclose([alpha, gamma], 90):
                fun = self.__mono2
                x0 = (a, b, c, np.deg2rad(beta))
            else:
                fun = self.__tri
                x0 = (a, b, c, np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma))

            U = self.cws.sample().getOrientedLattice().getU()

            omega = np.arccos((np.trace(U)-1)/2)

            val, vec = np.linalg.eig(U)

            ux, uy, uz = vec[:,np.argwhere(np.isclose(val, 1))[0][0]].real

            theta = np.arccos(uz)
            phi = np.arctan2(uy,ux)

            sol = scipy.optimize.least_squares(self.__res, x0=x0+(phi,theta,omega), args=(hkl,Q,fun)) #, method='lm'

            a, b, c, alpha, beta, gamma, phi, theta, omega = fun(sol.x)

            B = self.__B_matrix(a, b, c, alpha, beta, gamma)
            U = self.__U_matrix(phi, theta, omega)

            UB = np.dot(U,B)

            J = sol.jac
            cov = np.linalg.inv(J.T.dot(J))

            chi2dof = np.sum(sol.fun**2)/(sol.fun.size-sol.x.size)
            cov *= chi2dof

            sig = np.sqrt(np.diagonal(cov))

            sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma, *sig_angles = fun(sig)

            alpha, beta, gamma = np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)
            sig_alpha, sig_beta, sig_gamma = np.rad2deg(sig_alpha), np.rad2deg(sig_beta), np.rad2deg(sig_gamma)

            if np.isclose(a, sig_a):
                sig_a = 0
            if np.isclose(b, sig_b):
                sig_b = 0
            if np.isclose(c, sig_c):
                sig_c = 0

            if np.isclose(alpha, sig_alpha):
                sig_alpha = 0
            if np.isclose(beta, sig_beta):
                sig_beta = 0
            if np.isclose(gamma, sig_gamma):
                sig_gamma = 0

            SetUB(Workspace='cal', UB=UB)

            mtd['cal'].sample().getOrientedLattice().setError(sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma)

            self.__set_satellite_info(cal, mod_vec_1, mod_vec_2, mod_vec_3, max_order)

        SaveNexus(InputWorkspace='cal', Filename=filename)
        SaveIsawUB(InputWorkspace='cal', Filename=filename.replace('nxs','mat'))

        #if n_pks <= 20:

            # if mtd.doesExist('cal'):
            #     DeleteWorkspace('cal')

    def recalculate_hkl(self, tol=0.08, fname=None):

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
                           '  pk-vol%', ' bkg-vol%']
                fmt_hkl = 12*'{:8}'+2*'{:9}'+'\n'

                hkl_file = open(fname, 'w')
                hkl_file.write(fmt_hkl.format(*hdr_hkl))

                fmt_khl = 3*'{:8.3f}'+'{:8.4f}'+3*'{:8.3f}'+'{:8.4f}'+3*'{:8.3f}'+'{:8.4f}'+2*'{:9.2f}'+'\n'

            mod_vec_1 = ol.getModVec(0)
            mod_vec_2 = ol.getModVec(1)
            mod_vec_3 = ol.getModVec(2)

            lines_hkl = []

            for pn in range(mtd['out'].getNumberPeaks()-1,-1,-1):

                ipk, opk = mtd['in'].getPeak(pn), mtd['out'].getPeak(pn)

                # h, k, l = ipk.getIntHKL()
                # m, n, p = ipk.getIntMNP()

                # h, k, l, m, n, p = int(h), int(k), int(l), int(m), int(n), int(p)

                # dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)

                HKL = ipk.getHKL()
                hkl = opk.getHKL()

                dHKL = np.abs(HKL-hkl)
                d0 = ol.d(V3D(*HKL))

                Q = opk.getQSampleFrame()
                d = opk.getDSpacing()

                pk_vol_perc = np.round(100*opk.getBinCount(),2)
                bkg_vol_perc = np.round(100*opk.getAbsorptionWeightedPathLength(),2)

                delta_d = np.abs(d-d0)/d0*100

                line_hkl = [*HKL, d0, *hkl, d, *dHKL, delta_d, pk_vol_perc, bkg_vol_perc]
                lines_hkl.append(line_hkl)

                if np.any(dHKL > tol) or delta_d > 5:

                    self.iws.removePeak(pn)
                    self.cws.removePeak(pn)

            if fname is not None:

                sort = np.argsort([line_hkl[3] for line_hkl in lines_hkl])[::-1]

                for i in sort:
                    hkl_file.write(fmt_khl.format(*lines_hkl[i]))

                hkl_file.close()

            # SaveNexus(InputWorkspace='out', Filename='/tmp/out.nxs')
            # SaveNexus(InputWorkspace='in', Filename='/tmp/in.nxs')
            # SaveNexus(InputWorkspace='cws', Filename='/tmp/cws.nxs')
            # SaveNexus(InputWorkspace='iws', Filename='/tmp/iws.nxs')

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

                #peak.integrate()
                peak.individual_integrate()

                intens = peak.get_merged_intensity()
                sig_intens = peak.get_merged_intensity_error()

                pk_vol_fract = peak.get_merged_peak_volume_fraction()
                bkg_vol_fract = peak.get_merged_background_volume_fraction()

                run = peak.get_run_numbers().tolist()[0]
                R = peak.get_goniometers()[0]

                h, k, l, m, n, p = key
                Qx, Qy, Qz = peak.get_Q()

                if peak.is_peak_integrated():

                    peak_num = peak.get_peak_number()

                    self.cws.run().getGoniometer().setR(R)
                    self.iws.run().getGoniometer().setR(R)

                    dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)            

                    pk = self.iws.createPeakHKL(V3D(h+dh,k+dk,l+dl))
                    pk.setGoniometerMatrix(R)
                    pk.setIntHKL(V3D(h,k,l))
                    pk.setIntMNP(V3D(m,n,p))
                    #pk.setQSampleFrame(V3D(Qx,Qy,Qz))
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
                    pk.setQSampleFrame(V3D(Qx,Qy,Qz))
                    pk.setPeakNumber(peak_num)
                    pk.setIntensity(intens)
                    pk.setSigmaIntensity(sig_intens)
                    pk.setBinCount(pk_vol_fract)
                    pk.setAbsorptionWeightedPathLength(bkg_vol_fract)
                    pk.setRunNumber(run)
                    self.cws.addPeak(pk)

        #SortPeaksWorkspace(self.iws, ColumnNameToSortBy='DSpacing', SortAscending=False, OutputWorkspace=self.iws)
        #SortPeaksWorkspace(self.cws, ColumnNameToSortBy='DSpacing', SortAscending=False, OutputWorkspace=self.cws)

    def clear_peaks(self):

        for pws in [self.iws, self.cws]:
            DeleteTableRows(TableWorkspace=pws, Rows=range(pws.getNumberPeaks()))

    def __equivalent_sphere(self):

        chemical_formula = self.chemical_formula

        if chemical_formula is not None:

            m = self.sample_mass # g
            z = self.z_parameter

            mat_dict = self.__material_constants()

            atms = [atm.replace('(','').replace(')','').rstrip('1234567890.') for atm in chemical_formula.split(' ')]
            pres = [re.findall('(?:\d+)', atm)[0] if '(' in atm else '' for atm in chemical_formula.split(' ')]
            atms = [pre+atm for pre, atm in zip(pres,atms)]
            n_atms = [float(atm.replace(atm.rstrip('1234567890.'), '')) for atm in chemical_formula.split(' ')]

            amu = [mat_dict[atm][2] for atm in atms]

            N = np.sum(n_atms)
            n = N*z/self.iws.sample().getOrientedLattice().volume()

            M = np.dot(n_atms,amu)

            if N > 0:
                rho = (n/N)/0.6022*M
                V = m/rho
                R = np.cbrt(0.75/np.pi*m/rho) # cm
            else:
                rho, V, R = 0, 0, 0

            return m, M, n, N, rho, V, R 

        else:

            return 0, 0, 0, 0, 0, 0, 0 

    def __spherical_absorption(self):

        filename = os.path.join(os.path.dirname(__file__), 'absorption_sphere.csv')

        data = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=np.arange(1,92))

        muR = np.loadtxt(filename, skiprows=1, delimiter=',', usecols=(0))
        theta = np.loadtxt(filename, delimiter=',', max_rows=1, usecols=np.arange(1,92))

        f = scipy.interpolate.interp2d(muR, 2*theta, data.T, kind='cubic')

        return f

    def __material_constants(self):

        filename = os.path.join(os.path.dirname(__file__), 'NIST_cross-sections.dat')

        isotope, x_tot, x_abs, z, amu, br, bi = np.genfromtxt(filename, unpack=True, missing_values='---', dtype=('|U6', float, float, float, float, float, float))

        mat_dict = {}

        for i, atm in enumerate(isotope):

            mat_dict[atm] = x_tot[i], x_abs[i], amu[i]

        return mat_dict

    def apply_spherical_correction(self, vanadium_mass=0, fname=None):

        if fname is not None:
            absorption_file = open(fname, 'w')

        f = self.__spherical_absorption()

        chemical_formula = self.chemical_formula

        if chemical_formula is not None:

            mat_dict = self.__material_constants()

            atms = [atm.replace('(','').replace(')','').rstrip('1234567890.') for atm in chemical_formula.split(' ')]
            pres = [re.findall('(?:\d+)', atm)[0] if '(' in atm else '' for atm in chemical_formula.split(' ')]
            atms = [pre+atm for pre, atm in zip(pres,atms)]
            n_atms = [float(atm.replace(atm.rstrip('1234567890.'), '')) for atm in chemical_formula.split(' ')]

            x_tot = [mat_dict[atm][0] for atm in atms]
            x_abs = [mat_dict[atm][1] for atm in atms]

            chemical_formula = '-'.join(chemical_formula.split(' '))

            m, M, n, N, rho, V, R = self.__equivalent_sphere()

            sigma_a = np.dot(n_atms, x_abs)/N
            sigma_s = np.dot(n_atms, x_tot)/N

            if fname is not None:

                absorption_file.write('{}\n'.format(chemical_formula))
                absorption_file.write('absoption cross section: {:.4f} barn\n'.format(sigma_a))
                absorption_file.write('scattering cross section: {:.4f} barn\n'.format(sigma_s))

                absorption_file.write('linear absorption coefficient: {:.4f} 1/cm\n'.format(n*sigma_a))
                absorption_file.write('linear scattering coefficient: {:.4f} 1/cm\n'.format(n*sigma_s))

                absorption_file.write('mass: {:.4f} g\n'.format(m))
                absorption_file.write('density: {:.4f} g/cm^3\n'.format(rho))

                absorption_file.write('volume: {:.4f} cm^3\n'.format(V))
                absorption_file.write('radius: {:.4f} cm\n'.format(R))

                absorption_file.write('total atoms: {:.4f}\n'.format(N))
                absorption_file.write('molar mass: {:.4f} g/mol\n'.format(M))
                absorption_file.write('number density: {:.4f} 1/A^3\n'.format(n))

            van = self.nws.sample().getMaterial()

            van_sigma_a = van.absorbXSection()
            van_sigma_s = van.totalScatterXSection()

            van_M = van.relativeMolecularMass()
            van_n = van.numberDensityEffective # A^-3
            van_N = van.totalAtoms 

            van_rho = (van_n/van_N)/0.6022*van_M
            van_V = vanadium_mass/van_rho

            van_R = np.cbrt(0.75/np.pi*van_V)

            van_mu_s = van_n*van_sigma_s
            van_mu_a = van_n*van_sigma_a

            Uiso = float(self.nws.sample().getCrystalStructure().getScatterers()[0].split(' ')[-1])

            if fname is not None:

                absorption_file.write('\nV\n')
                absorption_file.write('absoption cross section: {:.4f} barn\n'.format(van_sigma_a))
                absorption_file.write('scattering cross section: {:.4f} barn\n'.format(van_sigma_s))

                absorption_file.write('linear absorption coefficient: {:.4f} 1/cm\n'.format(van_n*van_sigma_a))
                absorption_file.write('linear scattering coefficient: {:.4f} 1/cm\n'.format(van_n*van_sigma_s))

                absorption_file.write('mass: {:.4f} g\n'.format(vanadium_mass))
                absorption_file.write('density: {:.4f} g/cm^3\n'.format(van_rho))

                absorption_file.write('volume: {:.4f} cm^3\n'.format(van_V))
                absorption_file.write('radius: {:.4f} cm\n'.format(van_R))

                absorption_file.write('total atoms: {:.4f}\n'.format(van_N))
                absorption_file.write('molar mass: {:.4f} g/mol\n'.format(van_M))
                absorption_file.write('number density: {:.4f} 1/A^3\n'.format(van_n))
                absorption_file.write('isotropic displacement parameter: {:.4f} A^2\n'.format(Uiso))

                absorption_file.close()

            for key in self.peak_dict.keys():

                peaks = self.peak_dict.get(key)

                for peak in peaks:

                    wls = peak.get_wavelengths()
                    two_thetas = peak.get_scattering_angles()

                    Astar, Astar_van, T, Tbar = [], [], [], []

                    for wl, two_theta in zip(wls, two_thetas):

                        mu = n*(sigma_s+sigma_a*wl/1.8) # barn / ang^3 = 1/cm
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

                        van_mu = van_n*(van_sigma_s+van_sigma_a*wl/1.8)
                        van_muR = van_mu*van_R

                        # print('linear absorption coefficient: {} 1/cm'.format(mu))

                        d = 0.5*wl/np.sin(two_theta)

                        correction = f(van_muR,np.rad2deg(two_theta))[0]*0+1
                        Astar_van.append(correction) # sq sf * np.exp(4*np.pi**2*Uiso/d**2)

                    peak.set_data_scale(Astar)
                    peak.set_norm_scale(Astar_van)

                    peak.set_transmission_coefficient(T)
                    peak.set_weighted_mean_path_length(Tbar)

            self.clear_peaks()
            self.repopulate_workspaces()

    def apply_ellipsoidal_correction(self, vanadium_mass=0, ratios=[1,1,1], polar=np.pi/2, azimuthal=np.pi/2, omega=0, fname=None):

        if fname is not None:
            absorption_file = open(fname, 'w')

        chemical_formula = self.chemical_formula

        if chemical_formula is not None:

            mat_dict = self.__material_constants()

            atms = [atm.replace('(','').replace(')','').rstrip('1234567890.') for atm in chemical_formula.split(' ')]
            pres = [re.findall('(?:\d+)', atm)[0] if '(' in atm else '' for atm in chemical_formula.split(' ')]
            atms = [pre+atm for pre, atm in zip(pres,atms)]
            n_atms = [float(atm.replace(atm.rstrip('1234567890.'), '')) for atm in chemical_formula.split(' ')]

            x_tot = [mat_dict[atm][0] for atm in atms]
            x_abs = [mat_dict[atm][1] for atm in atms]

            chemical_formula = '-'.join(chemical_formula.split(' '))

            m, M, n, N, rho, V, R = self.__equivalent_sphere()

            sigma_a = np.dot(n_atms, x_abs)/N
            sigma_s = np.dot(n_atms, x_tot)/N

            ratios = np.array(ratios)/ratios[0]

            a1 = R/np.cbrt(ratios[1]*ratios[2])
            a2 = ratios[1]*a1
            a3 = ratios[2]*a1

            if fname is not None:

                absorption_file.write('{}\n'.format(chemical_formula))
                absorption_file.write('absoption cross section: {:.4f} barn\n'.format(sigma_a))
                absorption_file.write('scattering cross section: {:.4f} barn\n'.format(sigma_s))

                absorption_file.write('linear absorption coefficient: {:.4f} 1/cm\n'.format(n*sigma_a))
                absorption_file.write('linear scattering coefficient: {:.4f} 1/cm\n'.format(n*sigma_s))

                absorption_file.write('mass: {:.4f} g\n'.format(m))
                absorption_file.write('density: {:.4f} g/cm^3\n'.format(rho))

                absorption_file.write('volume: {:.4f} cm^3\n'.format(V))
                absorption_file.write('radius: {:.4f} cm\n'.format(R))

                absorption_file.write('a1: {:.4f} cm\n'.format(a1))
                absorption_file.write('a2: {:.4f} cm\n'.format(a2))
                absorption_file.write('a3: {:.4f} cm\n'.format(a3))

                absorption_file.write('total atoms: {:.4f}\n'.format(N))
                absorption_file.write('molar mass: {:.4f} g/mol\n'.format(M))
                absorption_file.write('number density: {:.4f} 1/A^3\n'.format(n))

            van = self.nws.sample().getMaterial()

            van_sigma_a = van.absorbXSection()
            van_sigma_s = van.totalScatterXSection()

            van_M = van.relativeMolecularMass()
            van_n = van.numberDensityEffective # A^-3
            van_N = van.totalAtoms 

            van_rho = (van_n/van_N)/0.6022*van_M
            van_V = vanadium_mass/van_rho

            van_R = np.cbrt(0.75/np.pi*van_V)

            van_mu_s = van_n*van_sigma_s
            van_mu_a = van_n*van_sigma_a

            Uiso = float(self.nws.sample().getCrystalStructure().getScatterers()[0].split(' ')[-1])

            if fname is not None:

                absorption_file.write('\nV\n')
                absorption_file.write('absoption cross section: {:.4f} barn\n'.format(van_sigma_a))
                absorption_file.write('scattering cross section: {:.4f} barn\n'.format(van_sigma_s))

                absorption_file.write('linear absorption coefficient: {:.4f} 1/cm\n'.format(van_n*van_sigma_a))
                absorption_file.write('linear scattering coefficient: {:.4f} 1/cm\n'.format(van_n*van_sigma_s))

                absorption_file.write('mass: {:.4f} g\n'.format(vanadium_mass))
                absorption_file.write('density: {:.4f} g/cm^3\n'.format(van_rho))

                absorption_file.write('volume: {:.4f} cm^3\n'.format(van_V))
                absorption_file.write('radius: {:.4f} cm\n'.format(van_R))

                absorption_file.write('total atoms: {:.4f}\n'.format(van_N))
                absorption_file.write('molar mass: {:.4f} g/mol\n'.format(van_M))
                absorption_file.write('number density: {:.4f} 1/A^3\n'.format(van_n))
                absorption_file.write('isotropic displacement parameter: {:.4f} A^2\n'.format(Uiso))

                absorption_file.close()

            ux = np.cos(azimuthal)*np.sin(polar)
            uy = np.sin(azimuthal)*np.sin(polar)
            uz = np.cos(polar)

            U = np.array([[np.cos(omega)+ux**2*(1-np.cos(omega)), ux*uy*(1-np.cos(omega))-uz*np.sin(omega), ux*uz*(1-np.cos(omega))+uy*np.sin(omega)],
                          [uy*ux*(1-np.cos(omega))+uz*np.sin(omega), np.cos(omega)+uy**2*(1-np.cos(omega)), uy*uz*(1-np.cos(omega))-ux*np.sin(omega)],
                          [uz*ux*(1-np.cos(omega))-uy*np.sin(omega), uz*uy*(1-np.cos(omega))+ux*np.sin(omega), np.cos(omega)+uz**2*(1-np.cos(omega))]])

            for key in self.peak_dict.keys():

                peaks = self.peak_dict.get(key)

                for peak in peaks:

                    wls = peak.get_wavelengths()
                    two_thetas = peak.get_scattering_angles()
                    az_phis = peak.get_azimuthal_angles()
                    
                    Rs = np.array(peak.get_goniometers())

                    kx_hat = np.sin(two_thetas)*np.cos(az_phis)
                    ky_hat = np.sin(two_thetas)*np.sin(az_phis)
                    kz_hat = np.cos(two_thetas)-1

                    ix = np.zeros_like(kx_hat)
                    iy = np.zeros_like(ky_hat)
                    iz = np.ones_like(kz_hat)

                    fx = -(ix+kx_hat)
                    fy = -(iy+ky_hat)
                    fz = -(iz+kz_hat)

                    i1, i2, i3 = np.einsum('kji,jk->ik', Rs, np.einsum('ji,jk->ik', U, [ix, iy, iz])) 
                    f1, f2, f3 = np.einsum('kji,jk->ik', Rs, np.einsum('ji,jk->ik', U, [fx, fy, fz]))

                    mu = n*(sigma_s+sigma_a*wls/1.8)

                    T, Tbar = self.__ellipsoid_absorption(mu, a1, a2, a3, i1, i2, i3, f1, f2, f3)

                    Astar = 1/T

                    Astar_van = Astar*0+1

                    peak.set_data_scale(Astar)
                    peak.set_norm_scale(Astar_van)

                    peak.set_transmission_coefficient(T)
                    peak.set_weighted_mean_path_length(Tbar)

            self.clear_peaks()
            self.repopulate_workspaces()

    def __volume_integral(self, f, p1, alpha, R):

        return scipy.integrate.simpson(scipy.integrate.simpson(scipy.integrate.simpson(f*R.reshape(-1,1)**2*np.sin(p1.reshape(-1,1,1)), R, axis=2), p1, axis=1), alpha, axis=0)

    def __ellipsoid_absorption(self, mu, a1, a2, a3, i1, i2, i3, f1, f2, f3, N=12):

        I1, I2, I3 = i1/a1, i2/a2, i3/a3
        F1, F2, F3 = f1/a1, f2/a2, f3/a3

        I = np.sqrt(I1**2+I2**2+I3**2)
        F = np.sqrt(F1**2+F2**2+F3**2)

        phi = np.arccos(-(I1*F1+I2*F2+I3*F3)/I/F)

        R = np.linspace(0,1,11)
        p1 = np.linspace(0,np.pi,31)
        alpha = np.linspace(0,2*np.pi,61)

        p2 = np.zeros_like(R.reshape(-1,1))+np.arccos(np.cos(p1.reshape(-1,1,1))*np.cos(np.pi-phi)+np.sin(p1.reshape(-1,1,1))*np.sin(np.pi-phi)*np.cos(alpha.reshape(-1,1,1,1)))

        f1 = R.reshape(-1,1)*np.cos(p1.reshape(-1,1,1))+np.sqrt(1-R.reshape(-1,1)**2*np.sin(p1.reshape(-1,1,1))**2)+np.zeros_like(alpha.reshape(-1,1,1,1))
        f2 = R.reshape(-1,1)*np.cos(p2                )+np.sqrt(1-R.reshape(-1,1)**2*np.sin(p2                )**2)+np.zeros_like(alpha.reshape(-1,1,1,1))

        n = 1
        if np.size(phi) > 1:
            n = np.size(phi)
        else:
            n = np.size(mu)

        a = np.zeros((N+1,n))
        t = np.zeros((N+1,n))

        for j in range(N+1):
            f = 0
            for p in range(j+1):
                f += scipy.special.comb(j,p)/I**p/F**(j-p)*f1**p*f2**(j-p)
            a[j,:] = 3/4/np.pi/scipy.special.factorial(j)*mu**j*self.__volume_integral(f, p1, alpha, R)
            t[j,:] = a[j,:]*j/mu

        da = np.zeros((N+1,N+1,n))
        dt = np.zeros((N+1,N+1,n))

        da[0,:,:] = +np.cumsum(((-1)**np.arange(N+1)*a.T).T, axis=0)
        dt[0,:,:] = -np.cumsum(((-1)**np.arange(N+1)*t.T).T, axis=0)

        for j in range(1,N+1):
            da[j,:-j,:] = (da[j-1,1:(N+2-j),:]+da[j-1,:-j,:])/2
            dt[j,:-j,:] = (dt[j-1,1:(N+2-j),:]+dt[j-1,:-j,:])/2

        #A = da[2*N//3,N//3,:]
        #Tbar = dt[2*N//3,N//3,:]

        A = da[-1,0,:]
        Tbar = dt[-1,0,:]

        Tbar /= A

        return A, Tbar

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

    def apply_extinction_correction(self, r, g, s, mu, phi, a, b, c, e, model='secondary, gaussian', fname=None):

        ol = self.iws.sample().getOrientedLattice()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

        if fname is not None:
            extinction_file = open(fname, 'a')
            extinction_file.write('\n\n')

            hdr_hkl = ['#      h', '       k', '       l', '    d-sp',
                       '           I', '       Icorr', 
                       '          F2', '      F2corr']

            fmt_hkl = 8*'{:8}'+'\n'
            extinction_file.write(fmt_hkl.format(*hdr_hkl))

            fmt_hkl = 3*'{:8.3f}'+'{:8.4f}'+4*'{:12.2f}'+'\n'

        generator = ReflectionGenerator(self.cs)

        sg = SpaceGroupFactory.createSpaceGroup(self.hm)
        pg = PointGroupFactory.createPointGroupFromSpaceGroup(sg)

        f1, f2 = self.__spherical_extinction(model)

        m, M, n, N, rho, V, R = self.__equivalent_sphere()

        V = self.iws.sample().getOrientedLattice().volume() # Ang^3
        R *= 1e+8 # Ang

        keys = list(self.peak_dict.keys())

        sort = np.argsort([self.get_d(*key) for key in keys])[::-1]

        keys = [keys[i] for i in sort]

        structure = { }

        for key in keys:

            h, k, l, m, n, p = key

            if m**2+n**2+p**2 > 0:
                cent_key = -h, -k, -l, -m, -n, -p
                key_pairs = [key, cent_key]
            else:
                equi_keys = pg.getEquivalents([h,k,l])
                key_pairs = [(int(h),int(k),int(l),0,0,0) for h, k, l in equi_keys]

            if structure.get(key) is None:

                const1, const2 = [], []

                I, sig, tt, w, wl, wpl, u_hat, d_hat = [], [], [], [], [], [], [], []

                for pair in key_pairs:

                    peaks = self.peak_dict.get(pair)

                    if peaks is not None:

                        for peak in peaks:

                            intens = peak.get_merged_intensity()
                            sig_intens = peak.get_merged_intensity_error()

                            if intens > 3*sig_intens:

                                clusters = peak.get_peak_clusters()

                                two_thetas = peak.get_scattering_angles()
                                az_phis = peak.get_azimuthal_angles()

                                omegas = np.deg2rad(peak.get_omega_angles())
                                lamdas = peak.get_wavelengths()
                                wpls = peak.get_weighted_mean_path_length()

                                Rs = np.array(peak.get_goniometers())

                                c1, c2 = f1(two_thetas), f2(two_thetas)

                                for cluster in clusters:

                                    intens = peak.get_partial_merged_intensity(cluster)
                                    sig_intens = peak.get_partial_merged_intensity_error(cluster)

                                    pk_vol_fract = peak.get_partial_merged_peak_volume_fraction(cluster)
                                    bkg_vol_fract = peak.get_partial_merged_background_volume_fraction(cluster)

                                    #if pk_vol_fract > 0.5 and bkg_vol_fract > 0.25:

                                    if intens > 3*sig_intens:

                                        I.append(intens)
                                        sig.append(sig_intens)

                                        tt.append(two_thetas[cluster].mean())
                                        w.append(np.angle(np.sum(np.exp(1j*omegas[cluster]))))

                                        wl.append(lamdas[cluster].mean())
                                        wpl.append(wpls[cluster].mean())

                                        const1.append(c1[cluster].mean())
                                        const2.append(c2[cluster].mean())

                                        ki_norm = np.array([0, 0, 1])
                                        kf_norm = np.array([np.cos(az_phis[cluster])*np.sin(two_thetas[cluster]),
                                                            np.sin(az_phis[cluster])*np.sin(two_thetas[cluster]),
                                                            np.cos(two_thetas[cluster])]).T

                                        ui = np.array([np.dot(R.T, ki_norm)    for j, R in enumerate(Rs[cluster])]).T.mean(axis=1)
                                        uf = np.array([np.dot(R.T, kf_norm[j]) for j, R in enumerate(Rs[cluster])]).T.mean(axis=1)

                                        ui /= np.linalg.norm(ui)
                                        uf /= np.linalg.norm(uf)

                                        dn = np.cross(uf,ui)
                                        dn /= np.linalg.norm(dn)

                                        u_hat.append(uf)
                                        d_hat.append(dn)

                I = np.array(I)
                sig = np.array(sig)
                tt = np.array(tt)
                w = np.array(w)
                wl = np.array(wl)
                wpl = np.array(wpl)

                u_hat = np.array(u_hat)
                d_hat = np.array(d_hat)

                const1 = np.array(const1)
                const2 = np.array(const2)

                if I.size > 0:

                    F2 = np.mean(I/s)

                    x0 = (F2,)
                    bounds = (0,np.inf)
                    args = (r, g, s, mu, phi, a, b, c, e, const1, const2, tt, w, wl, wpl, u_hat, d_hat, R, V, model, I, sig)

                    sol = scipy.optimize.least_squares(self.__residual_structure_factor, x0=x0, bounds=bounds, args=args, loss='soft_l1')
                    F2 = sol.x[0]

                    for key_pair in key_pairs:
                        structure[key_pair] = F2

        for key in keys:

            peaks = self.peak_dict.get(key)

            h, k, l, m, n, p = key

            for peak in peaks:

                scale = peak.get_data_scale()
                I = peak.get_merged_intensity()

                if len(scale) > 0 and I > 0:

                    intens = peak.get_intensity()
                    sig_intens = peak.get_intensity_error()
                    two_theta = peak.get_scattering_angles()
                    az_phi = peak.get_azimuthal_angles()
                    omega = np.deg2rad(peak.get_omega_angles())
                    lamda = peak.get_wavelengths()
                    Tbar = peak.get_weighted_mean_path_length()
                    Rs = peak.get_goniometers()

                    ki_norm = np.array([0, 0, 1])
                    kf_norm = np.array([np.cos(az_phi)*np.sin(two_theta),
                                        np.sin(az_phi)*np.sin(two_theta),
                                        np.cos(two_theta)]).T

                    ui = np.array([np.dot(R.T, ki_norm)    for j, R in enumerate(Rs)])
                    uf = np.array([np.dot(R.T, kf_norm[j]) for j, R in enumerate(Rs)])

                    d = np.array([np.cross(u0,u)/np.linalg.norm(np.cross(u0,u)) for u0, u in zip(ui,uf)])

                    F2 = generator.getFsSquared([V3D(h,k,l)])[0]

                    c1, c2 = f1(two_theta), f2(two_theta)

                    F2corr = structure.get(key)

                    if F2corr is None: F2corr = np.inf

                    x = self.__extinction_x(r, g, F2corr, two_theta, lamda, Tbar, uf, d, R, V, 'primary')
                    y = self.__extinction_correction(r, g, F2corr, c1, c2, two_theta, lamda, Tbar, uf, d, R, V, model)

                    scale = self.beam_profile(omega, lamda, mu, phi, a, b, c, e)
                    y *= scale

                    y[~np.isfinite(y)] = np.inf
                    y[x > 30] = np.inf

                    inv_y = 1/y

                    peak.set_ext_scale(inv_y)

                    if fname is not None:

                        Icorr = peak.get_merged_intensity()

                        dh, dk, dl = m*np.array(mod_vec_1)+n*np.array(mod_vec_2)+p*np.array(mod_vec_3)

                        H, K, L = h+dh, k+dk, l+dl

                        d_spacing = ol.d(V3D(H,K,L))

                        line = H, K, L, d_spacing, I, Icorr, F2, F2corr

                        extinction_file.write(fmt_hkl.format(*line))

        self.clear_peaks()
        self.repopulate_workspaces()

    def __residual_structure_factor(self, x, *args):

        F2, = x

        r, g, s, mu, phi, a, b, c, e, c1, c2, two_theta, omega, lamda, Tbar, u_dir, d_dir, R, V, model, intens, sig_intens = args

        y = self.__extinction_correction(r, g, F2, c1, c2, two_theta, lamda, Tbar, u_dir, d_dir, R, V, model)
        scale = self.beam_profile(omega, lamda, mu, phi, a, b, c, e)

        return (y*s*F2*scale-intens)/sig_intens

    # ---

    def peak_families(self, top_fraction=0.1, min_pk_vol_fract=0.85, min_bkg_vol_fract=0.05):

        generator = ReflectionGenerator(self.cs)

        sg = SpaceGroupFactory.createSpaceGroup(self.hm)
        pg = PointGroupFactory.createPointGroupFromSpaceGroup(sg)

        I, E = [], []
        two_theta, omega, lamda, Tbar, u_dir, d_dir = [], [], [], [], [], []
        hkl, F2, d_spacing = [], [], []
        band = []

        ref_keys = set()

        for pn in range(self.iws.getNumberPeaks()):
            pk = self.iws.getPeak(pn)

            h, k, l = pk.getIntHKL()
            m, n, p = pk.getIntMNP()

            h, k, l, m, n, p = int(h), int(k), int(l), int(m), int(n), int(p)

            if m**2+n**2+p**2 == 0:

                key = (h,k,l,m,n,p)
                ref_keys.add(key)

        keys = list(ref_keys)

        for key in keys:

            i, err = [], []
            tt, w, wl, wpl = [], [], [], []

            u_hat, d_hat = [], []

            hkl_ind = []

            h, k, l, m, n, p = key

            equivalents = pg.getEquivalents(V3D(h,k,l))#[::-1]

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

                        clusters = peak.get_peak_clusters()

                        two_thetas = peak.get_scattering_angles()
                        az_phis = peak.get_azimuthal_angles()

                        omegas = np.deg2rad(peak.get_omega_angles())
                        lamdas = peak.get_wavelengths()
                        wpls = peak.get_weighted_mean_path_length()

                        Rs = np.array(peak.get_goniometers())

                        for cluster in clusters:

                            intens = peak.get_partial_merged_intensity(cluster)
                            sig_intens = peak.get_partial_merged_intensity_error(cluster)

                            pk_vol_fract = peak.get_partial_merged_peak_volume_fraction(cluster)
                            bkg_vol_fract = peak.get_partial_merged_background_volume_fraction(cluster)

                            #if pk_vol_fract > min_pk_vol_fract and bkg_vol_fract > min_bkg_vol_fract:

                            if intens > 3*sig_intens and pk_vol_fract > min_pk_vol_fract:

                                i.append(intens)
                                err.append(sig_intens)

                                tt.append(two_thetas[cluster].mean())

                                w.append(np.angle(np.sum(np.exp(1j*omegas[cluster]))))
                                wl.append(lamdas[cluster].mean())
                                wpl.append(wpls[cluster].mean())

                                ki_norm = np.array([0, 0, 1])
                                kf_norm = np.array([np.cos(az_phis[cluster])*np.sin(two_thetas[cluster]),
                                                    np.sin(az_phis[cluster])*np.sin(two_thetas[cluster]),
                                                    np.cos(two_thetas[cluster])]).T

                                ui = np.array([np.dot(R.T, ki_norm)    for j, R in enumerate(Rs[cluster])]).T.mean(axis=1)
                                uf = np.array([np.dot(R.T, kf_norm[j]) for j, R in enumerate(Rs[cluster])]).T.mean(axis=1)

                                ui /= np.linalg.norm(ui)
                                uf /= np.linalg.norm(uf)

                                dn = np.cross(ui,uf)
                                dn /= np.linalg.norm(dn)

                                u_hat.append(uf)
                                d_hat.append(dn)

                                hkl_ind.append([h,k,l])

            if len(wl) > 3:

                b = np.max(wl)-np.min(wl)

                sf = generator.getFsSquared([V3D(h,k,l)])[0]
                band.append(sf)

                I.append(np.array(i))
                E.append(np.array(err))

                two_theta.append(np.array(tt))
                omega.append(np.array(w))
                lamda.append(np.array(wl))
                Tbar.append(np.array(wpl))

                u_dir.append(np.array(u_hat))
                d_dir.append(np.array(d_hat))

                hkl.append(hkl_ind)
                F2.append(sf)
                d_spacing.append(d)

        no_fam = len(F2)

        min_no = int(no_fam*top_fraction)

        sort = np.argsort(band)[::-1][:min_no]

        I = [I[i] for i in sort]
        E = [E[i] for i in sort]

        two_theta = [two_theta[i] for i in sort]
        omega = [omega[i] for i in sort]
        lamda = [lamda[i] for i in sort]
        Tbar = [Tbar[i] for i in sort]

        u_dir = [u_dir[i] for i in sort]
        d_dir = [d_dir[i] for i in sort]

        hkl = [hkl[i] for i in sort]
        F2 = [F2[i] for i in sort]
        d_spacing = [d_spacing[i] for i in sort]

        return I, E, two_theta, omega, lamda, Tbar, hkl, F2, d_spacing, u_dir, d_dir

    def __extinction_factor(self, rs, gs, two_theta, lamda, Tbar, u, d, R, V, model):

        r = self.__anisotropic_model(u.T, rs)
        g = self.__anisotropic_model(d.T, gs)

        a = 1e-4 # Ang

        rho = r/lamda

        if model == 'primary':

            xi = 1.5*a**2/V**2*lamda**4*rho**2

        elif model == 'secondary, gaussian':

            xi = a**2/V**2*lamda**3*rho/np.sqrt(1+rho**2*np.sin(two_theta)**2/g**2)*(Tbar*1e8)

        elif model == 'secondary, lorentzian':

            xi = a**2/V**2*lamda**3*rho/(1+rho*np.sin(two_theta)/g)*(Tbar*1e8)

        elif 'type II' in model:

            xi = a**2/V**2*lamda**3*rho*(Tbar*1e8)

        elif 'type I' in model:

            xi = a**2/V**2*lamda**3*g/np.sin(two_theta)*(Tbar*1e8)

        return xi

    def __extinction_x(self, r, g, F2, two_theta, lamda, Tbar, u, d, R, V, model):

        xi = self.__extinction_factor(r, g, two_theta, lamda, Tbar, u, d, R, V, model)

        return xi*F2

    def __extinction_correction(self, r, g, F2, c1, c2, two_theta, lamda, Tbar, u, d, R, V, model):

        xp = self.__extinction_x(r, g, F2, two_theta, lamda, Tbar, u, d, R, V, 'primary')
        yp = 1/(1+c1*xp**c2)

        xs = self.__extinction_x(r, g, F2, two_theta, lamda, Tbar, u, d, R, V, model)
        ys = 1/(1+c1*(yp*xs)**c2)

        return yp*ys

    def __extinction_model(self, r, g, s, F2, c1, c2, two_theta, lamda, Tbar, u, d, R, V, model):

        y = self.__extinction_correction(r, g, F2, c1, c2, two_theta, lamda, Tbar, u, d, R, V, model)

        return s*F2*y
        
    def __anisotropic_model(self, vec, vals):
       
        *lamda, phi, theta, omega = vals

        Lamda = np.diag(lamda)
        Q = self.__U_matrix(phi, theta, omega)

        A = np.dot(np.dot(Q,Lamda),Q.T)

        return 1/np.sqrt(A[0,0]*vec[0]**2+A[1,1]*vec[1]**2+A[2,2]*vec[2]**2+2*(A[1,2]*vec[1]*vec[2]+A[0,2]*vec[0]*vec[2]+A[0,1]*vec[0]*vec[1]))

    def __extinction_residual(self, params, I, E, HKL, two_theta, omega, lamda, Tbar, u_dir, d_dir, R, V, f1, f2, model, vary_F2=False):

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

        r = [params['r_vals_{}'.format(j)] for j in range(6)]
        g = [params['g_vals_{}'.format(j)] for j in range(6)]

        s = params['s']

        mu, phi, c = params['mu'], params['phi'], params['c']
        a, b, e = params['a'], params['b'], params['e']

        diff_I = np.array([])
        err_I = np.array([])

        for j, (i, err, hkl, tt, w, wl, wpl, u, d) in enumerate(zip(I, E, HKL, two_theta, omega, lamda, Tbar, u_dir, d_dir)):

            c1, c2 = f1(tt), f2(tt)

            h, k, l = hkl[0]

            sf = generator.getFsSquared([V3D(h,k,l)])[0] if not vary_F2 else params['F2_{}'.format(j)]

            scale = self.beam_profile(w, wl, mu, phi, a, b, c, e)
            intens = self.__extinction_model(r, g, s, sf, c1, c2, tt, wl, wpl, u, d, R, V, model)

            dI = (i-intens*scale)

            diff_I = np.concatenate((diff_I,dI))
            err_I = np.concatenate((err_I,err))

        return diff_I/err_I

    def __wobble_init(self, theta, mu, k):

        return np.exp(k*np.cos(theta-mu))

    def __estimate_wobble(self, params, HKL, R, V, f1, f2, model):

        r, g, s = params['r'], params['g'], params['s']

        generator = ReflectionGenerator(self.cs)

        sg = SpaceGroupFactory.createSpaceGroup(self.hm)
        pg = PointGroupFactory.createPointGroupFromSpaceGroup(sg)

        values = []
        angles = []

        for hkl in HKL:

            equivalents = pg.getEquivalents(hkl)

            value = []
            angle = []

            for equivalent in equivalents:

                h, k, l = equivalent

                F2 = generator.getFsSquared([V3D(h,k,l)])[0]

                key = (int(h), int(k), int(l), 0, 0, 0)

                peaks = self.peak_dict.get(key)

                if peaks is not None:

                    for ind, peak in enumerate(peaks):

                        wl = peak.get_wavelengths()
                        wpl = peak.get_weighted_mean_path_length()
                        tt = peak.get_scattering_angles()
                        w = np.deg2rad(peak.get_omega_angles())

                        # scales = peak.get_data_scale().copy()

                        # peak.set_data_scale(scales*scale)

                        c1, c2 = f1(tt), f2(tt)

                        y = self.__extinction_correction(r, g, F2, c1, c2, tt, wl, wpl, R, V, model)

                        peak.set_ext_scale(1/y)

                        I = peak.get_intensity()
                        merge = peak.get_merged_intensity()

                        # peak.set_data_scale(scales)

                        peak.set_ext_scale(y*0+1)

                        mask = np.isfinite(I) & (I > 0)

                        if len(I[mask]) > 0 and merge > 0:

                            average = np.angle(np.sum(np.exp(1j*w[mask])))

                            value.append(merge)
                            angle.append(average)

            if len(value) >= 2:

                value = (np.array(value)/np.mean(value)).tolist()

                values += value
                angles += angle

        values = np.array(values)
        angles = np.array(angles)

        popt, pcov = scipy.optimize.curve_fit(self.__wobble_init, angles, values, (0,0.1), bounds=([-180,0],[180,np.inf]), loss='soft_l1', verbose=2)
        mu, k_const = popt

        phi = 0
        a = 0
        b = k_const/2
        c = k_const/2
        e = 0

        return mu, phi, c, a, b, e

    def __wobble_residual(self, params, HKL, R, V, f1, f2, model):

        generator = ReflectionGenerator(self.cs)

        sg = SpaceGroupFactory.createSpaceGroup(self.hm)
        pg = PointGroupFactory.createPointGroupFromSpaceGroup(sg)

        mu, phi, c = params['mu'], params['phi'], params['c']
        a, b, e = params['a'], params['b'], params['e']

        r, g, s = params['r'], params['g'], params['s']

        diff = []

        for hkl in HKL:

            z, sig = [], []

            equivalents = pg.getEquivalents(hkl)

            for equivalent in equivalents:

                h, k, l = equivalent

                F2 = generator.getFsSquared([V3D(h,k,l)])[0]

                key = (int(h), int(k), int(l), 0, 0, 0)

                peaks = self.peak_dict.get(key)

                if peaks is not None:

                    for peak in peaks:

                        sig0 = peak.get_merged_intensity_error()

                        if sig0 > 0:

                            wl = peak.get_wavelengths()
                            wpl = peak.get_weighted_mean_path_length()
                            tt = peak.get_scattering_angles()
                            w = np.deg2rad(peak.get_omega_angles())

                            scale = self.beam_profile(w, wl, mu, phi, a, b, c, e)

                            scales = peak.get_data_scale().copy()

                            peak.set_data_scale(scales/scale)

                            c1, c2 = f1(tt), f2(tt)

                            y = self.__extinction_correction(r, g, F2, c1, c2, tt, wl, wpl, R, V, model)

                            peak.set_ext_scale(1/y)

                            z0 = peak.get_merged_intensity()

                            peak.set_data_scale(scales)

                            peak.set_ext_scale(y*0+1)

                            z.append(z0)
                            sig.append(sig0)

            z, sig = np.array(z), np.array(sig)

            z0 = np.mean(z)
            sig0 = np.array(sig)

            diff += ((z-z0)/sig0).tolist()
            diff += ((1/z-1/z0)*sig0).tolist()

        return diff

    def extinction_curves(self, r, g, s, mu, phi, a, b, c, e, model):

        f1, f2 = self.__spherical_extinction(model)

        m, M, n, N, rho, V, R = self.__equivalent_sphere()

        V = self.iws.sample().getOrientedLattice().volume() # Ang^3
        R *= 1e+8 # Ang

        I, E, two_theta, omega, lamda, Tbar, hkl, F2, d_spacing, u_dir, d_dir = self.peak_families()

        X, Y = [], []

        for j, (i, err, sf, tt, w, wl, wpl, u, d) in enumerate(zip(I, E, F2, two_theta, omega, lamda, Tbar, u_dir, d_dir)):

            c1, c2 = f1(tt), f2(tt)

            scale = s*self.beam_profile(w, wl, mu, phi, a, b, c, e)

            x = self.__extinction_x(r, g, sf, tt, wl, wpl, u, d, R, V, model)
            y = self.__extinction_model(r, g, scale, sf, c1, c2, tt, wl, wpl, u, d, R, V, model)

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

    def beam_profile(self, omega, lamda, mu, phi, a, b, c, e):

        t = omega-mu

        x = np.cos(t)*np.cos(phi)-np.sqrt(1-e**2)*np.sin(t)*np.sin(phi)

        return np.exp(-(c*x-b)**2/(1+a*lamda)**2)

    def fit_extinction(self, model):

        f1, f2 = self.__spherical_extinction(model)

        m, M, n, N, rho, V, R = self.__equivalent_sphere()

        V = self.iws.sample().getOrientedLattice().volume() # Ang^3
        R *= 1e+8 # Ang

        I, E, two_theta, omega, lamda, Tbar, hkl, F2, d_spacing, u_dir, d_dir = self.peak_families()

        params = Parameters()

        params.add('s', value=1, min=0)

        for j in [0]:
            params.add('r_vals_{}'.format(j), value=1e+16, min=1e-16, max=1e+16, vary=False)
            params.add('g_vals_{}'.format(j), value=1e+16, min=1e-16, max=1e+16, vary=False)
        for j in [1,2]:
            params.add('r_vals_{}'.format(j), expr='r_vals_0')
            params.add('g_vals_{}'.format(j), expr='g_vals_0')
        for j in [3,5]:
            params.add('r_vals_{}'.format(j), value=0, min=-np.pi/2, max=np.pi/2, vary=False)
            params.add('g_vals_{}'.format(j), value=0, min=-np.pi/2, max=np.pi/2, vary=False)
        for j in [4]:
            params.add('r_vals_{}'.format(j), value=np.pi/2, min=np.pi/4, max=3*np.pi/4, vary=False)
            params.add('g_vals_{}'.format(j), value=np.pi/2, min=np.pi/4, max=3*np.pi/4, vary=False)

        params.add('U', min=0, max=1, value=1e-3, vary=True)

        for j in range(1):
            if 'type II' in model:
                params['r_vals_{}'.format(j)].set(vary=True)
                params['g_vals_{}'.format(j)].set(vary=False)
            elif 'type I' in model:
                params['r_vals_{}'.format(j)].set(vary=True)
                params['g_vals_{}'.format(j)].set(vary=True)
            elif 'secondary' in model:
                params['r_vals_{}'.format(j)].set(vary=True)
                params['g_vals_{}'.format(j)].set(vary=True)
            else:
                params['r_vals_{}'.format(j)].set(vary=True)
                params['g_vals_{}'.format(j)].set(vary=False)

        for j in range(1):
            params['r_vals_{}'.format(j)].set(value=1e+4)
            params['g_vals_{}'.format(j)].set(value=1e+4)

        # params['r'].set(value=0, vary=False)
        # params['g'].set(vary=True)
        # params['s'].set(vary=True)
        # params['U'].set(value=0, vary=False)

        params.add('mu', min=-np.pi, max=np.pi, value=0, vary=False)
        params.add('phi', min=-np.pi, max=np.pi, value=0, vary=False)

        params.add('a', min=0, max=np.inf, value=0, vary=False)
        params.add('b', min=0, max=np.inf, value=0, vary=False)
        params.add('c', min=0, max=np.inf, value=0, vary=False)
        params.add('e', min=0, max=1, value=0, vary=False)

        generator = ReflectionGenerator(self.cs)

        for j, sf in enumerate(F2):

            params.add('F2_{}'.format(j), min=0, max=np.inf, value=sf, vary=False)

        out = Minimizer(self.__extinction_residual, params, reduce_fcn='negentropy', fcn_args=(I, E, hkl, two_theta, omega, lamda, Tbar, u_dir, d_dir, R, V, f1, f2, model, False))
        result = out.minimize(method='least_squares')

        report_fit(result)

        r_vals = [result.params['r_vals_{}'.format(j)].value for j in range(6)]
        g_vals = [result.params['g_vals_{}'.format(j)].value for j in range(6)]

        s = result.params['s'].value
        U = result.params['U'].value

        mu = result.params['mu'].value
        phi = result.params['phi'].value

        a = result.params['a'].value
        b = result.params['b'].value
        c = result.params['c'].value
        e = result.params['e'].value

        for j in range(6):
            params['r_vals_{}'.format(j)].set(value=r_vals[j], expr=None)
            params['g_vals_{}'.format(j)].set(value=g_vals[j], expr=None)

        for j in range(1,3):
            params['r_vals_{}'.format(j)].set(min=1e-16, max=1e+16)
            params['g_vals_{}'.format(j)].set(min=1e-16, max=1e+16)

        params['s'].set(value=s)
        params['U'].set(value=U)

        # mu, phi, c, a, b, e = self.__estimate_wobble(params, hkl, R, V, f1, f2, model)
        # 
        # params['mu'].set(vary=True, value=mu)
        # params['phi'].set(vary=True, value=phi)
        # 
        # params['a'].set(vary=True, value=a)
        # params['b'].set(vary=True, value=b)
        # params['c'].set(vary=True, value=c)
        # params['e'].set(vary=True, value=e)
        # 
        # params['r'].set(vary=False)
        # params['g'].set(vary=False)
        # params['s'].set(vary=False)
        # params['U'].set(vary=False)

        for j in range(6):
            if 'type II' in model:
                params['r_vals_{}'.format(j)].set(vary=True)
                params['g_vals_{}'.format(j)].set(vary=False)
            elif 'type I' in model:
                params['r_vals_{}'.format(j)].set(vary=True)
                params['g_vals_{}'.format(j)].set(vary=True)
            elif 'secondary' in model:
                params['r_vals_{}'.format(j)].set(vary=True)
                params['g_vals_{}'.format(j)].set(vary=True)
            else:
                params['r_vals_{}'.format(j)].set(vary=True)
                params['g_vals_{}'.format(j)].set(vary=False)

        params['s'].set(vary=True)
        params['U'].set(vary=True)

        for j, sf in enumerate(F2):

            params['F2_{}'.format(j)].set(vary=False)

        out = Minimizer(self.__extinction_residual, params, reduce_fcn='negentropy', fcn_args=(I, E, hkl, two_theta, omega, lamda, Tbar, u_dir, d_dir, R, V, f1, f2, model, False))
        result = out.minimize(method='least_squares')

        report_fit(result)

        r_vals = [result.params['r_vals_{}'.format(j)].value for j in range(6)]
        g_vals = [result.params['g_vals_{}'.format(j)].value for j in range(6)]

        for j in range(6):
            params['r_vals_{}'.format(j)].set(value=r_vals[j], vary=False)
            params['g_vals_{}'.format(j)].set(value=g_vals[j], vary=False)

        # params['U'].set(vary=True)
        # params['s'].set(vary=True)
        # 
        # for j, _ in enumerate(hkl):
        # 
        #     params['F2_{}'.format(j)].set(vary=False)
        # 
        # out = Minimizer(self.__extinction_residual, params, reduce_fcn='negentropy', fcn_args=(I, E, hkl, two_theta, omega, lamda, Tbar, u_dir, d_dir, R, V, f1, f2, model, False))
        # result = out.minimize(method='least_squares')
        # 
        # report_fit(result)

        s = result.params['s'].value
        U = result.params['U'].value

        params['U'].set(value=U, vary=False)
        params['s'].set(value=s, vary=False)

        params['mu'].set(vary=True)
        params['phi'].set(vary=True)

        params['a'].set(value=1, vary=True)
        params['b'].set(value=0, vary=True)
        params['c'].set(value=1, vary=True)
        params['e'].set(vary=True)

        #out = Minimizer(self.__wobble_residual, params, reduce_fcn='negentropy', fcn_args=(hkl, R, V, f1, f2, model))
        # result = out.minimize(method='least_squares')

        for j, sf in enumerate(F2):

            sf = result.params['F2_{}'.format(j)].value
            params['F2_{}'.format(j)].set(value=sf, vary=False)

        out = Minimizer(self.__extinction_residual, params, reduce_fcn='negentropy', fcn_args=(I, E, hkl, two_theta, omega, lamda, Tbar, u_dir, d_dir, R, V, f1, f2, model, False))
        result = out.minimize(method='least_squares')

        report_fit(result)

        r_vals = [result.params['r_vals_{}'.format(j)].value for j in range(6)]
        g_vals = [result.params['g_vals_{}'.format(j)].value for j in range(6)]

        s = result.params['s'].value
        U = result.params['U'].value

        mu = result.params['mu'].value
        phi = result.params['phi'].value

        a = result.params['a'].value
        b = result.params['b'].value
        c = result.params['c'].value
        e = result.params['e'].value

        return r_vals, g_vals, s, U, mu, phi, a, b, c, e, result.redchi

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

        if n_pk > 0:

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

class PeakFitPrune:

    def __init__(self, filename, mod_vec1=[0,0,0], mod_vec2=[0,0,0], mod_vec3=[0,0,0], max_order=0):

        self.filename = filename

        if (max_order == 0 or '_nuc' in filename) and ('_sat' not in filename):
            self.sat = False
        else:
            self.sat = True

        delim = (4,4,4) if self.sat else () 
        delim += (4,4,4,8,8,4,8,8,9,9,9,9,9,9,6,7,7,4,9,8,7,7)

        data = np.genfromtxt(filename, delimiter=delim, skip_footer=1)
        self.data = data.reshape(-1,len(delim))
        
        self.mod_vec1 = np.array(mod_vec1)
        self.mod_vec2 = np.array(mod_vec2)
        self.mod_vec3 = np.array(mod_vec3)

    def func(self, x, a, b, c):

        return a*np.exp(-b*x)+c

    def fit_peaks(self):

        dictionary = {}

        for line in self.data:

            if self.sat:
                h, k, l, m, n, p, I, sig, seq, wl, *info, d, row, col = line
            else:
                h, k, l, I, sig, seq, wl, *info, d, row, col = line
                m = n = p = 0

            hkl = np.array([h,k,l])+m*self.mod_vec1+n*self.mod_vec2+p*self.mod_vec3

            HKL = np.column_stack([hkl,-hkl])

            sort = np.lexsort(HKL, axis=0)

            key = tuple(HKL[:,sort[0]].tolist())

            if dictionary.get(key) is None:
                dictionary[key] = [I], [sig], [wl], [d]
            else:
                I_list, sig_list, wl_list, d_list = dictionary.get(key)
                I_list.append(I)
                sig_list.append(sig)
                wl_list.append(wl)
                d_list.append(d)
                dictionary[key] = I_list, sig_list, wl_list, d_list

        self.fitted_band = {}

        fname, ext = os.path.splitext(self.filename)

        with PdfPages(fname+'_prune'+'.pdf') as pdf:

            for key in dictionary.keys():

                I_list, sig_list, wl_list, d_list = dictionary.get(key)

                x, y, e = np.array(wl_list), np.array(I_list), np.array(sig_list)+0.001

                sort = np.argsort(x)

                x, y, e = x[sort], y[sort], e[sort]

                if len(x) >= 4:

                    popt, pcov = scipy.optimize.curve_fit(self.func, x-x.min(), y, sigma=e, p0=[y.max()-y.min(),0,y.min()], method='trf', bounds=([0,0,0],[np.inf,np.inf,np.inf]), loss='soft_l1')

                    y_hat = self.func(x-x.min(), *popt)-y

                    mu = np.mean(y_hat) #np.sum(I/sig**2)/np.sum(1/sig**2)
                    sigma = np.std(y_hat) #1/np.sqrt(np.sum(1/sig**2))

                    self.fitted_band[key] = popt, x.min(), mu, sigma

                    fig, ax = plt.subplots(1,1)
                    ax.errorbar(x, y, yerr=e, fmt='o', color='C0')
                    ax.minorticks_on()
                    ax.plot(x, self.func(x-x.min(), *popt), linestyle='-', color='C1')
                    ax.errorbar(x, y_hat, yerr=e, fmt='o', color='C2')
                    ax.plot(x, x*0, linestyle='--', color='C3' )
                    ax.set_xlabel('Wavelength [ang.]')
                    ax.set_ylabel('Intensity [arb. unit]')
                    ax.set_title('({:4.2f} {:4.2f} {:4.2f}) d = {:8.2f} ang.'.format(*[*key,np.mean(d_list).round(4)]))
                    pdf.savefig()
                    plt.close()
                        
    def write_intensity(self):

        fname, ext = os.path.splitext(self.filename)

        cols = 6 if self.sat else 3

        hkl_fmt = cols*'{:4.0f}'+2*'{:8.2f}'+'{:4.0f}'+2*'{:8.5f}'+6*'{:9.5f}'+\
                '{:6.0f}{:7.0f}{:7.4f}{:4.0f}{:9.5f}{:8.4f}'+2*'{:7.2f}'+'\n'

        with open(fname+'_prune'+ext, 'w') as f:

            for line in self.data:

                if self.sat:
                    h, k, l, m, n, p, I, sig, seq, wl, *info = line
                else:
                    h, k, l, I, sig, seq, wl, *info = line
                    m = n = p = 0

                hkl = np.array([h,k,l])+m*self.mod_vec1+n*self.mod_vec2+p*self.mod_vec3

                HKL = np.column_stack([hkl,-hkl])

                sort = np.lexsort(HKL, axis=0)

                key = tuple(HKL[:,sort[0]].tolist())

                if self.fitted_band.get(key) is not None:

                    popt, wl0, mu, sigma = self.fitted_band[key]

                    y_hat = self.func(wl-wl0, *popt)-I

                    z = (y_hat-mu)/sigma # z score

                    if np.abs(z) < 3:

                        if self.sat:
                            f.write(hkl_fmt.format(h,k,l,m,n,p,I,sig,seq,wl,*info))
                        else:
                            f.write(hkl_fmt.format(h,k,l,I,sig,seq,wl,*info))

            f.write(hkl_fmt.format(*[0]*len(hkl_fmt.split('}{'))))