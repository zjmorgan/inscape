# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from mantid.geometry import PointGroupFactory, SpaceGroupFactory
from mantid.kernel import V3D

import os 
directory = os.path.dirname(os.path.realpath(__file__))

CCR_file = directory+'/Mn3Si2Te6_CCR_100K_correction.hkl'
SS_file = directory+'/Mn3Si2Te6_SS_100K_0T_correction.hkl'

sg = SpaceGroupFactory.createSpaceGroup('P -3 1 c')
pg = PointGroupFactory.createPointGroupFromSpaceGroup(sg)

new_filename = directory+'/Mn3Si2Te6_SS_100K_0T_scaled.hkl'

CCR_data = np.loadtxt(CCR_file)
SS_data = np.loadtxt(SS_file)

scale = {}
reference = {}
dictionary = {}

for line in CCR_data:

    h, k, l, I, sig, d = line

    key = (h,k,-l)

    dictionary[key] = d, [I], [sig]

    hkl = V3D(h,k,l)
    
    equivalents = pg.getEquivalents(hkl)
    
    key = tuple(equivalents)
    
    if reference.get(key) is not None:
        intens_list = reference[key]
        intens_list.append(I)
    else:
        reference[key] = [I]

for line in SS_data:

    h, k, l, I, sig, d = line

    key = (h,k,l)

    if dictionary.get(key) is not None:
        item = dictionary[key]
        intens_list = item[1]
        intens_list.append(I)
        err_list = item[2]
        err_list.append(sig)
    else:
        dictionary[key] = d, [I], [sig]

# x, y = [], []
# 
# for key in dictionary.keys():
#     
#     item = dictionary[key]
# 
#     h, k, l = key
#     d = item[0]
# 
#     intens_list = item[1]
# 
#     if len(intens_list) > 1:
#         x.append(intens_list[0])
#         y.append(intens_list[1])
# 
# x, y = np.array(x), np.array(y)
# 
# p = np.polyfit(x, y, 1)
# 
# print(p)
# 
# data = []
# 
# for line in SS_data:
#     
#     h, k, l, I, sig, d = line
# 
#     I /= p[0]
#     sig /= np.sqrt(p[0])
# 
#     line = h, k, l, I, sig, d
#     data.append(line)
#     
# hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:8.4f}\n'
# 
# with open(new_filename, 'w') as f:
#     for line in data:
#         f.write(hkl_format.format(*line))
        
for key in reference.keys():
    
    I = np.mean(reference[key])
    
    reference[key] = I
    
print(reference)
    
for line in SS_data:
    
    h, k, l, I, sig, d = line
    
    hkl = V3D(h,k,l)
    
    equivalents = pg.getEquivalents(hkl)
    
    key = tuple(equivalents)
    
    if reference.get(key) is not None:
        scale[(h,k,l)] = I/reference[key]
        
temps = [5, 5, 5]
fields = [0, 1, 5]

hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:8.4f}\n'

SS_files = [directory+'/Mn3Si2Te6_SS_{:03}K_{}T_subtraction.hkl'.format(temp,field) for temp, field in zip(temps,fields)]

for SS_file in SS_files:
    
    SS_data = np.loadtxt(SS_file)
    
    new_filename = SS_file.replace('_subtraction','_scaled')
    
    with open(new_filename, 'w') as f:
    
        for line in SS_data:
        
            h, k, l, I, sig, d = line
            
            hkl = V3D(h,k,l)
            
            equivalents = pg.getEquivalents(hkl)
            
            key = tuple(equivalents)
            
            if scale.get((h,k,l)) is not None:
                
                p = scale[(h,k,l)]
                
                I /= p
                sig /= np.sqrt(p)
                
                line = h, k, l, I, sig, d
        
                f.write(hkl_format.format(*line))    
                
SS_file = directory+'/Mn3Si2Te6_SS_100K_0T_correction.hkl'

SS_data = np.loadtxt(SS_file)

new_filename = SS_file.replace('_correction','_scaled')

with open(new_filename, 'w') as f:

    for line in SS_data:
    
        h, k, l, I, sig, d = line
        
        hkl = V3D(h,k,l)
        
        equivalents = pg.getEquivalents(hkl)
        
        key = tuple(equivalents)
        
        if scale.get((h,k,l)) is not None:
            
            p = scale[(h,k,l)]
            
            print(I, p, I/p)
                        
            I /= p
            sig /= np.sqrt(p)
                        
            line = h, k, l, I, sig, d
    
            f.write(hkl_format.format(*line))  