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

# calibration files ------------------------------------------------------------
peaks_workspace = 'garnet_2022a_refined_peaks_workspace.nxs'
ub_file = 'garnet_2022a.mat'
calibration_name = 'garnet_2022a_refined_centered'

cell_type = 'Cubic'

LoadNexus(OutputWorkspace='pws', 
          Filename=os.path.join(directory,peaks_workspace))

sr_directory = '/SNS/CORELLI/shared/SCDCalibration/'
sr_file = 'CORELLI_Definition_2017-04-04_superresolution.xml'

LoadEmptyInstrument(InstrumentName='CORELLI', OutputWorkspace='corelli')

LoadEmptyInstrument(FileName=os.path.join(sr_directory,sr_file), 
                    OutputWorkspace='corelli_superresolution')

LoadParameterFile(Workspace='corelli', 
                  Filename=os.path.join(directory,calibration_name+'.xml'))

LoadParameterFile(Workspace='corelli_superresolution', 
                  Filename=os.path.join(directory,calibration_name+'.xml'))

pws = ApplyInstrumentToPeaks('pws', 'corelli_superresolution')

banks = [ 7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
         24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
         43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
         60, 61, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
         83, 84, 85, 86, 87, 88, 89, 90]

rowC = [63, 91]
rowB = [30, 62]
rowA = [1, 29]

latA, bnA = [], []
for bank in range(rowA[0],rowA[1]+1):
    if bank in banks and not bank in exclude:
        name = 'bank'+str(bank)
        FilterPeaks('pws', Criterion='=', BankName=name, OutputWorkspace='tmp')
        FilterPeaks('tmp', FilterVariable='h^2+k^2+l^2', 
                    FilterValue=0, Operator='>', OutputWorkspace='tmp')
        FindUBUsingIndexedPeaks('tmp', Tolerance=0.15)
        OptimizeLatticeForCellType('tmp', CellType=cell_type, Apply=True)
        latA.append([mtd['tmp'].sample().getOrientedLattice().a(),
                     mtd['tmp'].sample().getOrientedLattice().b(),
                     mtd['tmp'].sample().getOrientedLattice().c()])
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
        OptimizeLatticeForCellType('tmp', CellType=cell_type, Apply=True)
        latB.append([mtd['tmp'].sample().getOrientedLattice().a(),
                     mtd['tmp'].sample().getOrientedLattice().b(),
                     mtd['tmp'].sample().getOrientedLattice().c()])
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
        OptimizeLatticeForCellType('tmp', CellType=cell_type, Apply=True)
        latC.append([mtd['tmp'].sample().getOrientedLattice().a(),
                     mtd['tmp'].sample().getOrientedLattice().b(),
                     mtd['tmp'].sample().getOrientedLattice().c()])
        bnC.append(bank)
       
latC = np.array(latC)
bnC = np.array(bnC)

aA = latA[:,0]
bA = latA[:,1]    
cA = latA[:,2]

aB = latB[:,0]
bB = latB[:,1]    
cB = latB[:,2]

aC = latC[:,0]
bC = latC[:,1]    
cC = latC[:,2]

fig, ax = plt.subplots(3,3,sharey=True)

ax[0,0].plot(bnA, aA, 'o')
ax[0,0].plot(bnA, aA*0+a, '--')
ax[0,0].set_ylabel('$a$ [$\AA$]')
ax[0,0].set_title('Row A')

ax[1,0].plot(bnA, bA, 'o')
ax[1,0].plot(bnA, bA*0+b, '--')
ax[1,0].set_ylabel('$b$ [$\AA$]')

ax[2,0].plot(bnA, cA, 'o')
ax[2,0].plot(bnA, cA*0+c, '--')
ax[2,0].set_xlabel('Bank number')
ax[2,0].set_ylabel('$c$ [$\AA$]')

# ---

ax[0,1].plot(bnB, aB, 'o')
ax[0,1].plot(bnB, aB*0+a, '--')
ax[0,1].set_title('Row B')

ax[1,1].plot(bnB, bB, 'o')
ax[1,1].plot(bnB, bB*0+b, '--')

ax[2,1].plot(bnB, cB, 'o')
ax[2,1].plot(bnB, cB*0+c, '--')
ax[2,1].set_xlabel('Bank number')

# ---

ax[0,2].plot(bnC, aC, 'o')
ax[0,2].plot(bnC, aC*0+a, '--')
ax[0,2].set_title('Row C')

ax[1,2].plot(bnC, bC, 'o')
ax[1,2].plot(bnC, bC*0+b, '--')

ax[2,2].plot(bnC, cC, 'o')
ax[2,2].plot(bnC, cC*0+c, '--')
ax[2,2].set_xlabel('Bank number')

fig.savefig(os.path.join(directory,'compare_const_sample.png'))
fig.show()