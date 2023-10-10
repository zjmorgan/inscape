import numpy as np
import matplotlib.pyplot as plt
from mantid.simpleapi import *

from matplotlib.backends.backend_pdf import PdfPages

import os
import sys

from mantid.geometry import PointGroupFactory, SpaceGroupFactory

filename = sys.argv[1] #

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

import imp
import parameters

imp.reload(parameters)

dictionary = parameters.load_input_file(filename)

facility, instrument = parameters.set_instrument(dictionary['instrument'])
ipts = dictionary['ipts']

working_directory = '/{}/{}/IPTS-{}/shared/'.format(facility,instrument,ipts)
shared_directory = '/{}/{}/shared/'.format(facility,instrument)

if dictionary.get('tube-file') is not None:
    tube_calibration = os.path.join(shared_directory+'calibration', dictionary['tube-file'])
else:
    tube_calibration = None

if dictionary.get('detector-file') is not None:
    detector_calibration = os.path.join(shared_directory+'calibration', dictionary['detector-file'])
else:
    detector_calibration = None

a = dictionary.get('a')
b = dictionary.get('b')
c = dictionary.get('c')
alpha = dictionary.get('alpha')
beta = dictionary.get('beta')
gamma = dictionary.get('gamma')

T, F = True, False

sample    = [T, F, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T]
panels    = [F, F, F, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T]
moderator = [F, T, F, F, F, F, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T]
size      = [F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F]
time      = [F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F]

directory = os.path.dirname(os.path.abspath(filename))
outname = dictionary['name']

parameters.output_input_file(filename, directory, outname+'_cal')

peaks_workspace = dictionary['peaks-workspace']

outdir = os.path.join(directory, outname)
if not os.path.exists(outdir):
    os.mkdir(outdir)

if os.path.splitext(peaks_workspace)[1] == '.nxs':
    LoadNexus(OutputWorkspace='peaks', 
              Filename=os.path.join(directory,peaks_workspace))
else:
    LoadIsawPeaks(OutputWorkspace='peaks', 
                  Filename=os.path.join(directory,peaks_workspace))

# ConvertPeaksWorkspace(PeakWorkspace='peaks', 
#                       OutputWorkspace='peaks')

sr_directory = '/SNS/{}/shared/SCDCalibration/'.format(instrument)
sr_file = dictionary.get('superresolution-file') # 'CORELLI_Definition_2017-04-04_super_resolution.xml'

if sr_file is not None:
    LoadEmptyInstrument(FileName=sr_directory+sr_file, 
                        OutputWorkspace=instrument)
else:
    LoadEmptyInstrument(InstrumentName=instrument,
                        OutputWorkspace=instrument)

if detector_calibration is not None:
    if os.path.splitext(detector_calibration)[1] == '.xml':
        LoadParameterFile(Workspace=instrument, Filename=detector_calibration)
    else:
        LoadIsawDetCal(InputWorkspace=instrument, Filename=detector_calibration)

if mtd['peaks'].columnCount() == 14:
    ConvertPeaksWorkspace(PeakWorkspace='peaks', 
                          InstrumentWorkspace=instrument, 
                          OutputWorkspace='peaks')

CloneWorkspace(InputWorkspace='peaks', OutputWorkspace='ref')

ApplyInstrumentToPeaks(InputWorkspace='peaks', 
                       InstrumentWorkspace=instrument,
                       OutputWorkspace='peaks')

banks = mtd['peaks'].column(13)
for i in range(mtd['peaks'].getNumberPeaks()-1,-1,-1):
    mtd['peaks'].getPeak(i).setIntensity(0)
    mtd['peaks'].getPeak(i).setSigmaIntensity(0)
    if (banks[i] == ''):
        mtd['peaks'].removePeak(i)

N = mtd['peaks'].getNumberPeaks()
sr = set()

for i in range(N):
    run = mtd['peaks'].getPeak(i).getRunNumber()
    mtd['peaks'].getPeak(i).setIntensity(0)
    mtd['peaks'].getPeak(i).setSigmaIntensity(0)
    sr.add(int(run))

banks = list(set(banks))

runs = list(sr)
runs.sort()

for j in range(N):
    mtd['peaks'].getPeak(j).setGoniometerMatrix(np.eye(3))

if (np.allclose([a, b], c) and np.allclose([alpha, beta, gamma], 90)):
    cell_type = 'Cubic'
elif (np.allclose([a, b], c) and np.allclose([alpha, beta], gamma)):
    cell_type = 'Rhombohedral'
elif (np.isclose(a, b) and np.allclose([alpha, beta, gamma], 90)):
    cell_type = 'Tetragonal'
elif (np.isclose(a, b) and np.allclose([alpha, beta], 90) and np.isclose(gamma, 120)): 
    cell_type = 'Hexagonal' 
elif (np.allclose([alpha, beta, gamma], 90)): 
    cell_type = 'Orthorhombic'
elif np.allclose([alpha, gamma], 90):
    cell_type = 'Monoclinic'
elif np.allclose([alpha, beta], 90): 
    cell_type = 'Monoclinic'
else:
    cell_type = 'Triclinic'

sample_shift = []

with PdfPages(os.path.join(outdir, outname+'.pdf')) as pdf:

    for i, _ in enumerate(panels):

        FilterPeaks(InputWorkspace='peaks', 
                    FilterVariable='h^2+k^2+l^2', 
                    FilterValue=0, 
                    Operator='!=',
                    OutputWorkspace='peaks')

        FilterPeaks(InputWorkspace='peaks', 
                    FilterVariable='QMod', 
                    FilterValue=0, 
                    Operator='>',
                    OutputWorkspace='peaks')

        FilterPeaks(InputWorkspace='peaks',
                    BankName='',
                    Criterion='!=',
                    OutputWorkspace='peaks')

        N = mtd['peaks'].getNumberPeaks()
        for j in range(N):
            mtd['peaks'].getPeak(j).setGoniometerMatrix(np.eye(3))

        CreatePeaksWorkspace(InstrumentWorkspace='peaks',
                             NumberOfPeaks=0,
                             OutputWorkspace='peaks_tmp')

        for run in runs:

            FilterPeaks(InputWorkspace='peaks', 
                        FilterVariable='RunNumber', 
                        FilterValue=run, 
                        Operator='=', 
                        OutputWorkspace='tmp')
            
            if mtd['tmp'].getNumberPeaks() > 10:
            
                CalculateUMatrix(PeaksWorkspace='tmp',
                                 a=a,
                                 b=b,
                                 c=c,
                                 alpha=alpha,
                                 beta=beta,
                                 gamma=gamma)

                u = mtd['tmp'].sample().getOrientedLattice().getU()
                for j in range(mtd['tmp'].getNumberPeaks()):
                    mtd['tmp'].getPeak(j).setGoniometerMatrix(u)

                CombinePeaksWorkspaces(LHSWorkspace='tmp', 
                                       RHSWorkspace='peaks_tmp', 
                                       OutputWorkspace='peaks_tmp')

        CloneWorkspace(InputWorkspace='peaks_tmp', OutputWorkspace='peaks')

        SetUB(Workspace='peaks', 
              a=a, 
              b=b, 
              c=c, 
              alpha=alpha, 
              beta=beta, 
              gamma=gamma, 
              u='0,0,1', 
              v='1,0,0')

        FilterPeaks(InputWorkspace='peaks', 
                    FilterVariable='h^2+k^2+l^2', 
                    FilterValue=0, 
                    Operator='!=', 
                    OutputWorkspace='peaks')

        SCDCalibratePanels(PeakWorkspace='peaks',
                           RecalculateUB=False,
                           a=a,
                           b=b,
                           c=c,
                           alpha=alpha,
                           beta=beta,
                           gamma=gamma,
                           OutputWorkspace='calibration_table',
                           DetCalFilename=os.path.join(outdir,outname+'.DetCal'),
                           CSVFilename=os.path.join(outdir,outname+'.csv'),
                           XmlFilename=os.path.join(outdir,outname+'.xml'),
                           CalibrateT0=time[i],
                           SearchRadiusT0=10,
                           CalibrateL1=moderator[i],
                           SearchRadiusL1=0.2,
                           CalibrateBanks=panels[i],
                           SearchRadiusTransBank=0.5,
                           SearchRadiusRotXBank=5,
                           SearchRadiusRotYBank=5,
                           SearchRadiusRotZBank=5,
                           VerboseOutput=True,
                           SearchRadiusSamplePos=0.2,
                           TuneSamplePosition=sample[i],
                           CalibrateSize=size[i],
                           SearchRadiusSize=0.1,
                           FixAspectRatio=True)

        LoadParameterFile(Workspace=instrument, 
                          Filename=os.path.join(outdir,outname+'.xml'))

        sample_pos = mtd[instrument].getInstrument().getComponentByName('sample-position').getPos()
        sample_shift.append(sample_pos)

        for bank in banks:
            MoveInstrumentComponent(Workspace=instrument, 
                                    ComponentName=bank if instrument != 'CORELLI' else bank+'/sixteenpack', 
                                    X=-sample_pos[0], Y=-sample_pos[1], Z=-sample_pos[2], 
                                    RelativePosition=True)

        MoveInstrumentComponent(Workspace=instrument, 
                                ComponentName='sample-position', 
                                X=-sample_pos[0], Y=-sample_pos[1], Z=-sample_pos[2], 
                                RelativePosition=True)

        MoveInstrumentComponent(Workspace=instrument, 
                                ComponentName='moderator', 
                                X=0, Y=0, Z=-sample_pos[2], 
                                RelativePosition=True)

        ApplyInstrumentToPeaks(InputWorkspace='peaks', 
                               InstrumentWorkspace=instrument,
                               OutputWorkspace='peaks')

        SCDCalibratePanels(PeakWorkspace='peaks',
                           RecalculateUB=False,
                           OutputWorkspace='calibration_table',
                           DetCalFilename=os.path.join(outdir,outname+'.DetCal'),
                           CSVFilename=os.path.join(outdir,outname+'.csv'),
                           XmlFilename=os.path.join(outdir,outname+'.xml'),
                           CalibrateT0=False,
                           SearchRadiusT0=10,
                           CalibrateL1=False,
                           SearchRadiusL1=0.2,
                           CalibrateBanks=False,
                           SearchRadiusTransBank=0.5,
                           SearchradiusRotXBank=0,
                           SearchradiusRotYBank=0,
                           SearchradiusRotZBank=0,
                           VerboseOutput=True,
                           SearchRadiusSamplePos=0.2,
                           TuneSamplePosition=False) 

        LoadParameterFile(Workspace=instrument, 
                          Filename=os.path.join(outdir,outname+'.xml'))

        ApplyInstrumentToPeaks(InputWorkspace='peaks', 
                               InstrumentWorkspace=instrument,
                               OutputWorkspace='peaks')

        bn = []
        lat_a, lat_b, lat_c, lat_alpha, lat_beta, lat_gamma = [], [], [], [], [], []
        err_lat_a, err_lat_b, err_lat_c, err_lat_alpha, err_lat_beta, err_lat_gamma = [], [], [], [], [], []

        for bank in banks:
            FilterPeaks('peaks', Criterion='=', BankName=bank, OutputWorkspace='tmp')
            FilterPeaks('tmp', FilterVariable='h^2+k^2+l^2', 
                        FilterValue=0, Operator='>', OutputWorkspace='tmp')
            if mtd['tmp'].getNumberPeaks() > 20:
                FindUBUsingIndexedPeaks('tmp', Tolerance=0.15)
                #FindUBUsingLatticeParameters('tmp', a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
                #OptimizeLatticeForCellType('tmp', CellType=cell_type, Apply=True)
                lat_a.append(mtd['tmp'].sample().getOrientedLattice().a())
                lat_b.append(mtd['tmp'].sample().getOrientedLattice().b())
                lat_c.append(mtd['tmp'].sample().getOrientedLattice().c())
                lat_alpha.append(mtd['tmp'].sample().getOrientedLattice().alpha())
                lat_beta.append(mtd['tmp'].sample().getOrientedLattice().beta())
                lat_gamma.append(mtd['tmp'].sample().getOrientedLattice().gamma())
                # ---
                err_lat_a.append(mtd['tmp'].sample().getOrientedLattice().errora())
                err_lat_b.append(mtd['tmp'].sample().getOrientedLattice().errorb())
                err_lat_c.append(mtd['tmp'].sample().getOrientedLattice().errorc())
                err_lat_alpha.append(mtd['tmp'].sample().getOrientedLattice().erroralpha())
                err_lat_beta.append(mtd['tmp'].sample().getOrientedLattice().errorbeta())
                err_lat_gamma.append(mtd['tmp'].sample().getOrientedLattice().errorgamma())
                bn.append(int(bank.strip('bank')))

        fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')

        ax[0,0].errorbar(bn, lat_a, err_lat_a, fmt='o', color='C0')
        ax[0,1].errorbar(bn, lat_b, err_lat_b, fmt='o', color='C1')
        ax[0,2].errorbar(bn, lat_c, err_lat_c, fmt='o', color='C2')

        ax[1,0].errorbar(bn, lat_alpha, err_lat_alpha, fmt='s', color='C0')
        ax[1,1].errorbar(bn, lat_beta, err_lat_beta, fmt='s', color='C1')
        ax[1,2].errorbar(bn, lat_gamma, err_lat_gamma, fmt='s', color='C2')

        ax[0,0].axhline(a, linestyle='--', color='k', zorder=0)
        ax[0,1].axhline(b, linestyle='--', color='k', zorder=0)
        ax[0,2].axhline(c, linestyle='--', color='k', zorder=0)

        ax[1,0].axhline(alpha, linestyle='--', color='k', zorder=0)
        ax[1,1].axhline(beta, linestyle='--', color='k', zorder=0)
        ax[1,2].axhline(gamma, linestyle='--', color='k', zorder=0)

        ax[0,0].set_ylabel('a [\u212B]')
        ax[0,1].set_ylabel('b [\u212B]')
        ax[0,2].set_ylabel('c [\u212B]')

        ax[1,0].set_ylabel('\u03B1 [deg.]')
        ax[1,1].set_ylabel('\u03B2 [deg.]')
        ax[1,2].set_ylabel('\u03B3 [deg.]')

        ax[1,0].set_xlabel('Bank number')
        ax[1,1].set_xlabel('Bank number')
        ax[1,2].set_xlabel('Bank number')
        
        if i == 0:
            ylim_00 = ax[0,0].get_ylim()
            ylim_01 = ax[0,1].get_ylim()
            ylim_02 = ax[0,2].get_ylim()
            ylim_10 = ax[1,0].get_ylim()
            ylim_11 = ax[1,1].get_ylim()
            ylim_12 = ax[1,2].get_ylim()
        else:
            ax[0,0].set_ylim(ylim_00)
            ax[0,1].set_ylim(ylim_01)
            ax[0,2].set_ylim(ylim_02)
            ax[1,0].set_ylim(ylim_10)
            ax[1,1].set_ylim(ylim_11)
            ax[1,2].set_ylim(ylim_12)

        ax[0,0].minorticks_on()
        ax[0,1].minorticks_on()
        ax[0,2].minorticks_on()

        ax[1,0].minorticks_on()
        ax[1,1].minorticks_on()
        ax[1,2].minorticks_on()

        pdf.savefig()
        plt.close()

SaveNexus(InputWorkspace='peaks', 
          Filename=os.path.join(outdir,outname+'_peaks_workspace.nxs'))

LoadParameterFile(Workspace=instrument, 
                  Filename=os.path.join(outdir,outname+'.xml'))

ApplyInstrumentToPeaks(InputWorkspace='ref', 
                       InstrumentWorkspace=instrument,
                       OutputWorkspace='ref')

f = open(os.path.join(outdir, outname+'_sample.txt'), 'w')
f.write(' dx [micron] dy [micron] dz [micron]\n')

for sample_pos in sample_shift:
    x, y, z = sample_pos
    f.write('{:12.2f}{:12.2f}{:12.2f}\n'.format(x*1e6,y*1e6,z*1e6))

f.close()

bn = []
lat_a, lat_b, lat_c, lat_alpha, lat_beta, lat_gamma = [], [], [], [], [], []
err_lat_a, err_lat_b, err_lat_c, err_lat_alpha, err_lat_beta, err_lat_gamma = [], [], [], [], [], []

for bank in banks:
    FilterPeaks('ref', Criterion='=', BankName=bank, OutputWorkspace='tmp')
    FilterPeaks('tmp', FilterVariable='h^2+k^2+l^2', 
                FilterValue=0, Operator='>', OutputWorkspace='tmp')
    if mtd['tmp'].getNumberPeaks() > 20:
        try:
            FindUBUsingIndexedPeaks('tmp', Tolerance=0.15)
        except:
            FindUBUsingLatticeParameters('tmp', a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        OptimizeLatticeForCellType('tmp', CellType=cell_type, Apply=True)
        lat_a.append(mtd['tmp'].sample().getOrientedLattice().a())
        lat_b.append(mtd['tmp'].sample().getOrientedLattice().b())
        lat_c.append(mtd['tmp'].sample().getOrientedLattice().c())
        lat_alpha.append(mtd['tmp'].sample().getOrientedLattice().alpha())
        lat_beta.append(mtd['tmp'].sample().getOrientedLattice().beta())
        lat_gamma.append(mtd['tmp'].sample().getOrientedLattice().gamma())
        # ---
        err_lat_a.append(mtd['tmp'].sample().getOrientedLattice().errora())
        err_lat_b.append(mtd['tmp'].sample().getOrientedLattice().errorb())
        err_lat_c.append(mtd['tmp'].sample().getOrientedLattice().errorc())
        err_lat_alpha.append(mtd['tmp'].sample().getOrientedLattice().erroralpha())
        err_lat_beta.append(mtd['tmp'].sample().getOrientedLattice().errorbeta())
        err_lat_gamma.append(mtd['tmp'].sample().getOrientedLattice().errorgamma())
        bn.append(int(bank.strip('bank')))

fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')

ax[0,0].errorbar(bn, lat_a, err_lat_a, fmt='o', color='C0')
ax[0,1].errorbar(bn, lat_b, err_lat_b, fmt='o', color='C1')
ax[0,2].errorbar(bn, lat_c, err_lat_c, fmt='o', color='C2')

ax[1,0].errorbar(bn, lat_alpha, err_lat_alpha, fmt='s', color='C0')
ax[1,1].errorbar(bn, lat_beta, err_lat_beta, fmt='s', color='C1')
ax[1,2].errorbar(bn, lat_gamma, err_lat_gamma, fmt='s', color='C2')

ax[0,0].axhline(a, linestyle='--', color='k', zorder=0)
ax[0,1].axhline(b, linestyle='--', color='k', zorder=0)
ax[0,2].axhline(c, linestyle='--', color='k', zorder=0)

ax[1,0].axhline(alpha, linestyle='--', color='k', zorder=0)
ax[1,1].axhline(beta, linestyle='--', color='k', zorder=0)
ax[1,2].axhline(gamma, linestyle='--', color='k', zorder=0)

ax[0,0].set_ylabel('a [\u212B]')
ax[0,1].set_ylabel('b [\u212B]')
ax[0,2].set_ylabel('c [\u212B]')

ax[1,0].set_ylabel('\u03B1 [deg.]')
ax[1,1].set_ylabel('\u03B2 [deg.]')
ax[1,2].set_ylabel('\u03B3 [deg.]')

ax[1,0].set_xlabel('Bank number')
ax[1,1].set_xlabel('Bank number')
ax[1,2].set_xlabel('Bank number')

ax[0,0].minorticks_on()
ax[0,1].minorticks_on()
ax[0,2].minorticks_on()

ax[1,0].minorticks_on()
ax[1,1].minorticks_on()
ax[1,2].minorticks_on()

fig.savefig(os.path.join(directory, outname+'.pdf'))
fig.show()