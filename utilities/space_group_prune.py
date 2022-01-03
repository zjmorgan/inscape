# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from mantid.geometry import SpaceGroupFactory
from mantid.kernel import V3D

sampleT = '005K_lscale5'
filename = '/SNS/CORELLI/IPTS-26829/shared/scripts/Mn3Si2Te6_CCR_'+sampleT+'_corrected_filtered.hkl'

data = np.loadtxt(filename)

sg = SpaceGroupFactory.createSpaceGroup('P -3 1 c')

miss, total = 0, 0

lines = []

print('Space group #{} ({})'.format(sg.getNumber(),sg.getHMSymbol()))
print('Reflections:')

for line in data:
    
    h, k, l, I, sig, d = line
    
    hkl = V3D(h,k,l)
    
    if sg.isAllowedReflection(hkl):
        lines.append(line)
    else:
        print('({},{},{})'.format(int(h),int(k),int(l)))
        miss += 1
    total += 1
        
print('{}/{} reflections not allowed in space group'.format(miss,total))

hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:8.4f}\n'

new_filename = filename.split('.')
new_filename = ''.join(new_filename[:-1]+['_pruned.']+new_filename[-1:])

print(new_filename)

with open(new_filename, 'w') as f:
    for line in lines:
        f.write(hkl_format.format(*line))
