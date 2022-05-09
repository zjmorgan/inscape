# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from mantid.geometry import PointGroupFactory, SpaceGroupFactory
from mantid.kernel import V3D

filename = '/SNS/CORELLI/IPTS-26829/shared/scripts/Mn3Si2Te6_SS_100K_0T_corrected_filtered.hkl'

data = np.loadtxt(filename)

sg = SpaceGroupFactory.createSpaceGroup('P -3 1 c')
pg = PointGroupFactory.createPointGroupFromSpaceGroup(sg)

miss, total = 0, 0

lines = []

print('Space group #{} ({})'.format(sg.getNumber(),sg.getHMSymbol()))
print('Reflections:')

dictionary = {}

for line in data:
    
    h, k, l, I, sig, d = line
    
    hkl = V3D(h,k,l)
    
    if sg.isAllowedReflection(hkl):
        lines.append(line)
        
        equivalents = pg.getEquivalents(hkl)
        
        key = tuple(equivalents)
                
        if dictionary.get(key) is None:
            dictionary[key] = [len(equivalents),d,[(h,k,l)],[I]]
        else:
            item = dictionary[key]
            
            intens_list = item[3]
            intens_list.append(I)
            item[3] = intens_list  
            
            peak_list = item[2]
            peak_list.append((h,k,l))
            item[2] = peak_list        
            
            dictionary[key] = item
    else:
        print('({},{},{}) d = {:2.4f} \u212B'.format(int(h),int(k),int(l),d))
        miss += 1
    total += 1
    
print(dictionary)

data = []

for key in dictionary.keys():
    item = dictionary[key]
    
    peak_list = np.array(item[2])
    intens_list = np.array(item[3])
    
    median = np.median(intens_list)
    Q1, Q3 = np.percentile(intens_list, [25,75])
    IQR = Q3-Q1
    
    high = np.argwhere(Q3+1.5*IQR < intens_list)
    low = np.argwhere(Q1-1.5*IQR > intens_list)
    
    if len(high) > 0:
        print('Intensity too high outlier', peak_list[high])
    if len(low) > 0:
        print('Intensity too low outlier', peak_list[low])
        
    i = np.argmax(intens_list)
    
    h, k, l = peak_list[i]
    I = intens_list[i]
    d = item[1]
    
    line = h, k, l, I, sig, d
    
    data.append(line)

print('{}/{} reflections not allowed in space group'.format(miss,total))

hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:8.4f}\n'

new_filename = filename.split('.')
new_filename = ''.join(new_filename[:-1]+['_pruned_max.']+new_filename[-1:])

print(new_filename)

with open(new_filename, 'w') as f:
    for line in data:
        f.write(hkl_format.format(*line))