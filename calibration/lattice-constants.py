# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import sys 
import os 

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/home/zgf/.git/inscape/integration/')

# directories ------------------------------------------------------------------
iptsfolder = '/SNS/CORELLI/IPTS-23019/'

# lattice constants ------------------------------------------------------------
a = 11.9157
b = 11.9157
c = 11.9157
alpha = 90
beta = 90
gamma = 90
     
# banks ------------------------------------------------------------------------
exclude = []

banks = [ 7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
         24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
         43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
         60, 61, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
         83, 84, 85, 86, 87, 88, 89, 90]

rowC = [63, 91]
rowB = [30, 62]
rowA = [1, 29]

rowC = [63, 73]
rowB = [30, 50]
rowA = [1, 20]

# calibration files ------------------------------------------------------------
peaks_workspace = 'garnet_2022a_refined_peaks_workspace.nxs'
#peaks_workspace = 'garnet_2022a_peaks_workspace.nxs'

calibration_name = 'garnet_2022a_refined_centered'

ub_file = 'garnet_2022a.mat'

outname = 'compare_lattice_constants' 

cell_type = 'Cubic'

sr_directory = '/SNS/CORELLI/shared/SCDCalibration/'
sr_file = 'CORELLI_Definition_2017-04-04_superresolution.xml'

ClearCache(AlgorithmCache=True, 
           InstrumentCache=True,
           DownloadedInstrumentFileCache=True,
           GeometryFileCache=True,
           WorkspaceCache=True,
           UsageServiceCache=True)

LoadEmptyInstrument(FileName=os.path.join(sr_directory,sr_file), 
                    OutputWorkspace='corelli_eng')

LoadNexus(OutputWorkspace='pws', 
          Filename=os.path.join(directory,peaks_workspace))
          
pws = ApplyInstrumentToPeaks('pws', 'corelli_eng')

latA, bnA = [], []
for bank in range(rowA[0],rowA[1]+1):
    if bank in banks and not bank in exclude:
        name = 'bank'+str(bank)
        FilterPeaks('pws', Criterion='=', BankName=name, OutputWorkspace='tmp')
        FilterPeaks('tmp', FilterVariable='h^2+k^2+l^2', 
                    FilterValue=0, Operator='>', OutputWorkspace='tmp')
        FindUBUsingIndexedPeaks('tmp', Tolerance=0.15)
        FindUBUsingLatticeParameters('tmp', a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        #OptimizeLatticeForCellType('tmp', CellType=cell_type, Apply=True)
        latA.append([mtd['tmp'].sample().getOrientedLattice().a(),
                     mtd['tmp'].sample().getOrientedLattice().b(),
                     mtd['tmp'].sample().getOrientedLattice().c(),
                     mtd['tmp'].sample().getOrientedLattice().alpha(),
                     mtd['tmp'].sample().getOrientedLattice().beta(),
                     mtd['tmp'].sample().getOrientedLattice().gamma()])
        bnA.append(bank)

latA = np.array(latA)
bnA = np.array(bnA)

latB, bnB = [], []
for bank in range(rowB[0],rowB[1]+1):
    if bank in banks and not bank in exclude:
        name = 'bank'+str(bank)
        FilterPeaks('pws', Criterion='=', BankName=name, OutputWorkspace='tmp')
        FilterPeaks('tmp', FilterVariable='h^2+k^2+l^2', 
                    FilterValue=0, Operator='>', OutputWorkspace='tmp')
        FindUBUsingIndexedPeaks('tmp', Tolerance=0.15)
        FindUBUsingLatticeParameters('tmp', a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        #OptimizeLatticeForCellType('tmp', CellType=cell_type, Apply=True)
        latB.append([mtd['tmp'].sample().getOrientedLattice().a(),
                     mtd['tmp'].sample().getOrientedLattice().b(),
                     mtd['tmp'].sample().getOrientedLattice().c(),
                     mtd['tmp'].sample().getOrientedLattice().alpha(),
                     mtd['tmp'].sample().getOrientedLattice().beta(),
                     mtd['tmp'].sample().getOrientedLattice().gamma()])
        bnB.append(bank)

latB = np.array(latB)
bnB = np.array(bnB)

latC, bnC = [], []
for bank in range(rowC[0],rowC[1]+1):
    if bank in banks and not bank in exclude:
        name = 'bank'+str(bank)
        FilterPeaks('pws', Criterion='=', BankName=name, OutputWorkspace='tmp')
        FilterPeaks('tmp', FilterVariable='h^2+k^2+l^2', 
                    FilterValue=0, Operator='>', OutputWorkspace='tmp')
        FindUBUsingIndexedPeaks('tmp', Tolerance=0.15)
        FindUBUsingLatticeParameters('tmp', a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        #OptimizeLatticeForCellType('tmp', CellType=cell_type, Apply=True)
        latC.append([mtd['tmp'].sample().getOrientedLattice().a(),
                     mtd['tmp'].sample().getOrientedLattice().b(),
                     mtd['tmp'].sample().getOrientedLattice().c(),
                     mtd['tmp'].sample().getOrientedLattice().alpha(),
                     mtd['tmp'].sample().getOrientedLattice().beta(),
                     mtd['tmp'].sample().getOrientedLattice().gamma()])
        bnC.append(bank)

latC = np.array(latC)
bnC = np.array(bnC)

aA = latA[:,0]
bA = latA[:,1]
cA = latA[:,2]
alphaA = latA[:,3]
betaA = latA[:,4]
gammaA = latA[:,5]

aB = latB[:,0]
bB = latB[:,1]
cB = latB[:,2]
alphaB = latB[:,3]
betaB = latB[:,4]
gammaB = latB[:,5]

aC = latC[:,0]
bC = latC[:,1]
cC = latC[:,2]
alphaC = latC[:,3]
betaC = latC[:,4]
gammaC = latC[:,5]

# ---

ClearCache(AlgorithmCache=True, 
           InstrumentCache=True,
           DownloadedInstrumentFileCache=True,
           GeometryFileCache=True,
           WorkspaceCache=True,
           UsageServiceCache=True)

LoadEmptyInstrument(FileName=os.path.join(sr_directory,sr_file), 
                    OutputWorkspace='corelli_cal')

LoadParameterFile(Workspace='corelli_cal', 
                  Filename=os.path.join(directory,calibration_name+'.xml'))

LoadNexus(OutputWorkspace='pws', 
          Filename=os.path.join(directory,peaks_workspace))

pws = ApplyInstrumentToPeaks('pws', 'corelli_cal')

latA, bnA = [], []
for bank in range(rowA[0],rowA[1]+1):
    if bank in banks and not bank in exclude:
        name = 'bank'+str(bank)
        FilterPeaks('pws', Criterion='=', BankName=name, OutputWorkspace='tmp')
        FilterPeaks('tmp', FilterVariable='h^2+k^2+l^2', 
                    FilterValue=0, Operator='>', OutputWorkspace='tmp')
        FindUBUsingIndexedPeaks('tmp', Tolerance=0.15)
        FindUBUsingLatticeParameters('tmp', a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        #OptimizeLatticeForCellType('tmp', CellType=cell_type, Apply=True)
        latA.append([mtd['tmp'].sample().getOrientedLattice().a(),
                     mtd['tmp'].sample().getOrientedLattice().b(),
                     mtd['tmp'].sample().getOrientedLattice().c(),
                     mtd['tmp'].sample().getOrientedLattice().alpha(),
                     mtd['tmp'].sample().getOrientedLattice().beta(),
                     mtd['tmp'].sample().getOrientedLattice().gamma()])
        bnA.append(bank)
       
latA = np.array(latA)
bnA = np.array(bnA)

latB, bnB = [], []
for bank in range(rowB[0],rowB[1]+1):
    if bank in banks and not bank in exclude:
        name = 'bank'+str(bank)
        FilterPeaks('pws', Criterion='=', BankName=name, OutputWorkspace='tmp')
        FilterPeaks('tmp', FilterVariable='h^2+k^2+l^2', 
                    FilterValue=0, Operator='>', OutputWorkspace='tmp')
        FindUBUsingIndexedPeaks('tmp', Tolerance=0.15)
        FindUBUsingLatticeParameters('tmp', a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        #OptimizeLatticeForCellType('tmp', CellType=cell_type, Apply=True)
        latB.append([mtd['tmp'].sample().getOrientedLattice().a(),
                     mtd['tmp'].sample().getOrientedLattice().b(),
                     mtd['tmp'].sample().getOrientedLattice().c(),
                     mtd['tmp'].sample().getOrientedLattice().alpha(),
                     mtd['tmp'].sample().getOrientedLattice().beta(),
                     mtd['tmp'].sample().getOrientedLattice().gamma()])
        bnB.append(bank)

latB = np.array(latB)
bnB = np.array(bnB)

latC, bnC = [], []
for bank in range(rowC[0],rowC[1]+1):
    if bank in banks and not bank in exclude:
        name = 'bank'+str(bank)
        FilterPeaks('pws', Criterion='=', BankName=name, OutputWorkspace='tmp')
        FilterPeaks('tmp', FilterVariable='h^2+k^2+l^2', 
                    FilterValue=0, Operator='>', OutputWorkspace='tmp')
        FindUBUsingIndexedPeaks('tmp', Tolerance=0.15)
        FindUBUsingLatticeParameters('tmp', a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        #OptimizeLatticeForCellType('tmp', CellType=cell_type, Apply=True)
        latC.append([mtd['tmp'].sample().getOrientedLattice().a(),
                     mtd['tmp'].sample().getOrientedLattice().b(),
                     mtd['tmp'].sample().getOrientedLattice().c(),
                     mtd['tmp'].sample().getOrientedLattice().alpha(),
                     mtd['tmp'].sample().getOrientedLattice().beta(),
                     mtd['tmp'].sample().getOrientedLattice().gamma()])
        bnC.append(bank)

latC = np.array(latC)
bnC = np.array(bnC)

aA_cal = latA[:,0]
bA_cal = latA[:,1]
cA_cal = latA[:,2]
alphaA_cal = latA[:,3]
betaA_cal = latA[:,4]
gammaA_cal = latA[:,5]

aB_cal = latB[:,0]
bB_cal = latB[:,1]
cB_cal = latB[:,2]
alphaB_cal = latB[:,3]
betaB_cal = latB[:,4]
gammaB_cal = latB[:,5]

aC_cal = latC[:,0]
bC_cal = latC[:,1]
cC_cal = latC[:,2]
alphaC_cal = latC[:,3]
betaC_cal = latC[:,4]
gammaC_cal = latC[:,5]

# ---

fig, ax = plt.subplots(3,3,sharey=True)

ax[0,0].plot(bnA, aA, 'o', label='Engineering')
ax[0,0].plot(bnA, aA_cal, 's', label='Calibrated')
ax[0,0].plot(bnA, aA*0+a, '--', zorder=100)
ax[0,0].set_ylabel('$a$ [$\AA$]')
ax[0,0].set_title('Row A')
ax[0,0].legend()

ax[1,0].plot(bnA, bA, 'o', label='Engineering')
ax[1,0].plot(bnA, bA_cal, 's', label='Calibrated')
ax[1,0].plot(bnA, bA*0+b, '--', zorder=100)
ax[1,0].set_ylabel('$b$ [$\AA$]')

ax[2,0].plot(bnA, cA, 'o', label='Engineering')
ax[2,0].plot(bnA, cA_cal, 's', label='Calibrated')
ax[2,0].plot(bnA, cA*0+c, '--', zorder=100)
ax[2,0].set_xlabel('Bank number')
ax[2,0].set_ylabel('$c$ [$\AA$]')

# ---

ax[0,1].plot(bnB, aB, 'o', label='Engineering')
ax[0,1].plot(bnB, aB_cal, 's', label='Calibrated')
ax[0,1].plot(bnB, aB*0+a, '--', zorder=100)
ax[0,1].set_title('Row B')

ax[1,1].plot(bnB, bB, 'o', label='Engineering')
ax[1,1].plot(bnB, bB_cal, 's', label='Calibrated')
ax[1,1].plot(bnB, bB*0+b, '--', zorder=100)

ax[2,1].plot(bnB, cB, 'o', label='Engineering')
ax[2,1].plot(bnB, cB_cal, 's', label='Calibrated')
ax[2,1].plot(bnB, cB*0+c, '--', zorder=100)
ax[2,1].set_xlabel('Bank number')

# ---

ax[0,2].plot(bnC, aC, 'o', label='Engineering')
ax[0,2].plot(bnC, aC_cal, 's', label='Calibrated')
ax[0,2].plot(bnC, aC*0+a, '--', zorder=100)
ax[0,2].set_title('Row C')

ax[1,2].plot(bnC, bC, 'o', label='Engineering')
ax[1,2].plot(bnC, bC_cal, 's', label='Calibrated')
ax[1,2].plot(bnC, bC*0+b, '--', zorder=100)

ax[2,2].plot(bnC, cC, 'o', label='Engineering')
ax[2,2].plot(bnC, cC_cal, 's', label='Calibrated')
ax[2,2].plot(bnC, cC*0+c, '--', zorder=100)
ax[2,2].set_xlabel('Bank number')

fig.savefig(os.path.join(directory,outname+'.png'))
fig.show()

# ---

fig, ax = plt.subplots(3,3,sharey=True)

ax[0,0].plot(bnA, alphaA, 'o', label='Engineering')
ax[0,0].plot(bnA, alphaA_cal, 's', label='Calibrated')
ax[0,0].plot(bnA, alphaA*0+alpha, '--', zorder=100)
ax[0,0].set_ylabel('$a$ [$\AA$]')
ax[0,0].set_title('Row A')
ax[0,0].legend()

ax[1,0].plot(bnA, betaA, 'o', label='Engineering')
ax[1,0].plot(bnA, betaA_cal, 's', label='Calibrated')
ax[1,0].plot(bnA, betaA*0+beta, '--', zorder=100)
ax[1,0].set_ylabel('$b$ [$\AA$]')

ax[2,0].plot(bnA, gammaA, 'o', label='Engineering')
ax[2,0].plot(bnA, gammaA_cal, 's', label='Calibrated')
ax[2,0].plot(bnA, gammaA*0+gamma, '--', zorder=100)
ax[2,0].set_xlabel('Bank number')
ax[2,0].set_ylabel('$c$ [$\AA$]')

# ---

ax[0,1].plot(bnB, alphaB, 'o', label='Engineering')
ax[0,1].plot(bnB, alphaB_cal, 's', label='Calibrated')
ax[0,1].plot(bnB, alphaB*0+alpha, '--', zorder=100)
ax[0,1].set_title('Row B')

ax[1,1].plot(bnB, betaB, 'o', label='Engineering')
ax[1,1].plot(bnB, betaB_cal, 's', label='Calibrated')
ax[1,1].plot(bnB, betaB*0+beta, '--', zorder=100)

ax[2,1].plot(bnB, gammaB, 'o', label='Engineering')
ax[2,1].plot(bnB, gammaB_cal, 's', label='Calibrated')
ax[2,1].plot(bnB, gammaB*0+gamma, '--', zorder=100)
ax[2,1].set_xlabel('Bank number')

# ---

ax[0,2].plot(bnC, alphaC, 'o', label='Engineering')
ax[0,2].plot(bnC, alphaC_cal, 's', label='Calibrated')
ax[0,2].plot(bnC, alphaC*0+alpha, '--', zorder=100)
ax[0,2].set_title('Row C')

ax[1,2].plot(bnC, betaC, 'o', label='Engineering')
ax[1,2].plot(bnC, betaC_cal, 's', label='Calibrated')
ax[1,2].plot(bnC, betaC*0+beta, '--', zorder=100)

ax[2,2].plot(bnC, gammaC, 'o', label='Engineering')
ax[2,2].plot(bnC, gammaC_cal, 's', label='Calibrated')
ax[2,2].plot(bnC, gammaC*0+gamma, '--', zorder=100)
ax[2,2].set_xlabel('Bank number')

fig.savefig(os.path.join(directory,outname+'_angles.png'))
fig.show()