import numpy as np
import matplotlib.pyplot as plt
from mantid.simpleapi import *

import os
import sys

from mantid.geometry import PointGroupFactory, SpaceGroupFactory

filename = sys.argv[1]

if n_proc > os.cpu_count():
    n_proc = os.cpu_count()

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

parameters.output_input_file(filename, directory, outname+'_cal')

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

# calibration procedure --------------------------------------------------------
sample    = [T] #[T, F, T, T, T, T, T, T, T, F, F, F]
panels    = [T] #[F, F, F, T, T, T, T, T, T, F, T, F]
moderator = [T] #[F, T, F, F, F, F, T, T, T, T, T, T]
size      = [T] #[F, F, F, F, F, F, F, F, F, T, F, T]
time      = [T] #[F, F, F, F, F, F, F, F, F, F, F, F]

working_directory = '/{}/{}/IPTS-{}/shared/'.format(facility,instrument,ipts)
shared_directory = '/{}/{}/shared/'.format(facility,instrument)

directory = os.path.dirname(os.path.abspath(filename))
outname = dictionary['name']

peaks_workspace = dictionary['peaks-workspace']

outdir = os.path.join(directory, outname)
if not os.path.exists(outdir):
    os.mkdir(outdir)

# ------------------------------------------------------------------------------

if os.path.splitext(peaks_workspace)[1] == '.nxs':
    LoadNexus(OutputWorkspace='peaks', 
              Filename=os.path.join(directory,peaks_workspace))
else:
    LoadIsawPeaks(OutputWorkspace='peaks', 
                  Filename=os.path.join(directory,peaks_workspace))

banks = mtd['peaks'].column(13)
for i in range(mtd['peaks'].getNumberPeaks()-1,-1,-1):
    if (banks[i] == ''):
        mtd['peaks'].removePeak(i)

LoadEmptyInstrument(InstrumentName=instrument, OutputWorkspace=instrument)

sr_directory = '/SNS/{}/shared/SCDCalibration/'.format(instrument)
sr_file = dictionary.get('superresolution-file') # 'CORELLI_Definition_2017-04-04_super_resolution.xml'

if sr_file is not None:
    LoadEmptyInstrument(FileName=sr_directory+sr_file, 
                        OutputWorkspace=detector_calibration+'_super_resolution')

if detector_calibration is not None:
    if os.path.splitext(detector_calibration)[1] == '.xml':
        LoadParameterFile(Workspace=instrument, Filename=detector_calibration)
    else:
        LoadIsawDetCal(InputWorkspace=instrument, Filename=detector_calibration)

    if sr_file is not None:
        if os.path.splitext(detector_calibration)[1] == '.xml':
            LoadParameterFile(Workspace=instrument+'_super_resolution', Filename=detector_calibration)
        else:
            LoadIsawDetCal(InputWorkspace=instrument+'_super_resolution', Filename=detector_calibration)
        peaks = ApplyInstrumentToPeaks('peaks', instrument+'_super_resolution')
    else:
        peaks = ApplyInstrumentToPeaks('peaks', instrument)

N = mtd['peaks'].getNumberPeaks()
sr = set()

for i in range(N):
    run = mtd['peaks'].getPeak(i).getRunNumber()
    sr.add(int(run))

runs = list(sr)
runs.sort()

gon = mtd['peaks'].run().getGoniometer()

R = []

for run in runs:
    FilterPeaks(InputWorkspace='peaks', 
                FilterVariable='RunNumber', 
                FilterValue=run, 
                Operator='=', 
                OutputWorkspace='tmp')
    gon.setR(mtd['tmp'].getPeak(0).getGoniometerMatrix())
    omega, chi, phi = gon.getEulerAngles('YZY')
    R.append([omega, chi, phi])

R = np.array(R)
    
for j in range(N):
    mtd['peaks'].getPeak(j).setGoniometerMatrix(np.eye(3))

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
                OutputWorkspace='pws')

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

    if sr_file is not None:
        LoadParameterFile(Workspace=instrument+'_super_resolution', 
                          Filename=os.path.join(directory,outdir+'.xml'))
        peaks = ApplyInstrumentToPeaks('peaks', instrument+'_super_resolution')
    else:
        peaks = ApplyInstrumentToPeaks('peaks', instrument)

sample_pos = mtd['peaks'].getPeak(0).getSamplePos()

banks =  list(set(mtd['peaks'].column(13)))
banks.sort()

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
                        
peaks = ApplyInstrumentToPeaks('peaks', instrument)
       
SCDCalibratePanels(PeakWorkspace='peaks',
                   RecalculateUB=False,
                   OutputWorkspace='testCaliTable',
                   DetCalFilename=os.path.join(outdir,outname+'_centered.DetCal'),
                   CSVFilename=os.path.join(outdir,outname+'_centered.csv'),
                   XmlFilename=os.path.join(outdir,outname+'_centered.xml'),
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

SaveNexus(InputWorkspace='peaks', 
          Filename=os.path.join(outdir,outname+'_peaks_workspace.nxs'))

bn = []
lat_a, lat_b, lat_c, lat_alpha, lat_beta, lat_gamma = [], [], [], [], [], []
err_lat_a, err_lat_b, err_lat_c, err_lat_alpha, err_lat_beta, err_lat_gamma = [], [], [], [], [], []

for bank in banks:
    name = 'bank'+str(bank)
    FilterPeaks('pws', Criterion='=', BankName=name, OutputWorkspace='tmp')
    FilterPeaks('tmp', FilterVariable='h^2+k^2+l^2', 
                FilterValue=0, Operator='>', OutputWorkspace='tmp')
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
    bn.append(bank)

fig, ax = plt.subplots(2, 3, sharey=True)

ax[0,0].errorbar(bn, lat_a, err_lat_a, fmt='o')
ax[0,1].errorbar(bn, lat_b, err_lat_b, fmt='o')
ax[0,2].errorbar(bn, lat_c, err_lat_c, fmt='o')

ax[1,0].errorbar(bn, lat_alpha, err_lat_alpha, fmt='s')
ax[1,1].errorbar(bn, lat_beta, err_lat_beta, fmt='s')
ax[1,2].errorbar(bn, lat_gama, err_lat_gamma, fmt='s')

ax[0,0].axhline(a, '--', zorder=100)
ax[0,1].axhline(b, '--', zorder=100)
ax[0,2].axhline(c, '--', zorder=100)

ax[1,0].axhline(alpha, '--', zorder=100)
ax[1,1].axhline(beta, '--', zorder=100)
ax[1,2].axhline(gamma, '--', zorder=100)

ax[0,0].set_ylabel('$a$ [$\AA$]')
ax[0,1].set_ylabel('$b$ [$\AA$]')
ax[0,2].set_ylabel('$c$ [$\AA$]')

ax[1,0].set_ylabel('$\alpha$ [$^\circ$]')
ax[1,1].set_ylabel('$\beta$ [$^\circ$]')
ax[1,2].set_ylabel('$\gamma$ [$^\circ$]')

ax[1,0].set_xlabel('Bank number')
ax[1,1].set_xlabel('Bank number')
ax[1,2].set_xlabel('Bank number')

fig.savefig(os.path.join(outdir,outname+'.pdf'))
fig.show()