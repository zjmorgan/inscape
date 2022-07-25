# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import os

instrument = 'CORELLI'
banks_to_mask = '1-6,29-30,62-67,91,87-88,25-26'
tubes_to_mask = '1,16'
pixels_to_mask = '1-12,245-256'

bank_tube_to_mask = ['45,11', '49,1', '52,16']
bank_tube_pixel_to_mask = ['58,13-16,80-130', '59,1-4,80-130']

ipts = 23019
run_no = [269692,269752]
run_no = [269754,269814]
run_no = [270781,270841]

k_min, k_max = 2.5, 10

n_bins = 10

file_directory = '/SNS/{}/IPTS-{}/nexus/'
file_name = '{}_{}.nxs.h5'

gon_axis = 'BL9:Mot:Sample:Axis3.RBV'

if type(run_no) is list:
    files_to_load = ','.join([os.path.join(file_directory.format(instrument,ipts), file_name.format(instrument,run)) for run in range(run_no[0],run_no[1]+1)])
else:
    files_to_load = os.path.join(file_directory.format(instrument,ipts), file_name.format(instrument,run_no))

Load(Filename=files_to_load, OutputWorkspace='van')

NormaliseByCurrent(InputWorkspace='van', OutputWorkspace='van')

MaskBTP(Workspace='van', Bank=banks_to_mask)
MaskBTP(Workspace='van', Tube=tubes_to_mask)
MaskBTP(Workspace='van', Pixel=pixels_to_mask)

for pair in bank_tube_to_mask:
    bank_to_mask, tube_to_mask = pair.split(',')
    MaskBTP(Workspace='van', Bank=bank_to_mask, Tube=tube_to_mask)

for triplet in bank_tube_pixel_to_mask:
    bank_to_mask, tube_to_mask, pixel_to_mask = triplet.split(',')
    MaskBTP(Workspace='van', Bank=bank_to_mask, Tube=tube_to_mask, Pixel=pixel_to_mask)

LoadEmptyInstrument(InstrumentName=instrument, OutputWorkspace=instrument)

CreateGroupingWorkspace(InputWorkspace=instrument,
                        GroupDetectorsBy='bank',
                        OutputWorkspace='group')

GroupDetectors(InputWorkspace='van',
           ExcludeGroupNumbers=banks_to_mask,
           CopyGroupingFromWorkspace='group',
           OutputWorkspace='van')

ConvertUnits(InputWorkspace='van', OutputWorkspace='van', Target='Momentum')
CropWorkspace(InputWorkspace='van', OutputWorkspace='van', XMin=k_min, XMax=k_max)

Rebin(InputWorkspace='van', OutputWorkspace='van', Params='{},{},{}'.format(k_min,(k_max-k_min)/n_bins,k_max))

if instrument == 'CORELLI':
    SetGoniometer(Workspace='van', Axis0='{},0,1,0,1'.format(gon_axis))
else:
    SetGoniometer(Workspace='van', Goniometers='Universal')

Y = []
varphi = []

for van in mtd['van']:
    data = mtd[str(van)]
    omega, chi, phi = data.run().getGoniometer().getEulerAngles('YZY')
    varphi.append(omega)
    for i in range(data.getNumberHistograms()):
        y = data.readY(i)
        Y.append(y)
        
Y = np.array(Y).reshape(mtd['van'].size(),-1,10)
varphi = np.array(varphi)

varphi = np.mod(varphi, 360)

fig, ax = plt.subplots(1, 1, num=i)
for j in range(n_bins):
    ax.plot(varphi[1:-1], np.sum(Y[1:-1,:,j], axis=1)/np.sum(Y[1:-1,:,j], axis=1).mean(), linestyle='-', marker='.', label='{:2.2f}'.format(k_min+j*(k_max-k_min)/(n_bins-1)))
ax.legend()
ax.set_title('Sensitivity of momentum [ang.] on detector counts')
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_xlabel(r'Goniometer angle [deg.]') #
ax.set_ylabel(r'Normalized counts')
fig.show()

for i in range(91):

    fig, ax = plt.subplots(1, 1, num=i)
    for j in range(n_bins):
        #ax.plot(varphi, np.sum(Y[:,:,j], axis=1)/np.sum(Y[:,:,j], axis=1).mean(), linestyle='-', marker='.', label='{:2.2f}'.format(k_min+j*(k_max-k_min)/(n_bins-1)))
        ax.plot(varphi, Y[:,i,j]/Y[:,i,j].mean(), linestyle='-', marker='.', label='{:2.2f}'.format(k_min+j*(k_max-k_min)/(n_bins-1)))
    ax.legend()
    ax.set_title('Sensitivity of momentum [ang.] on detector counts')
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlabel(r'Goniometer angle [deg.]') #
    ax.set_ylabel(r'Normalized counts')
    fig.show()

fig, ax = plt.subplots(1, 1, num=0)
bank = np.arange(91)
for j in range(n_bins):
    data = Y[:,:,j]/Y[:,:,j].mean()
    ax.plot(bank, np.std(data, axis=0), linestyle='-', marker='.', label='{:2.2f}'.format(k_min+j*(k_max-k_min)/(n_bins-1)))
ax.legend()
ax.set_title('Sensitivity of momentum [ang.] on detector counts')
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_xlabel(r'Spectrum no') #
ax.set_ylabel(r'Normalized counts')
fig.show()

plt.close('all')