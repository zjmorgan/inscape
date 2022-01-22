# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from mantid.geometry import PointGroupFactory, SpaceGroupFactory
from mantid.kernel import V3D

import os 
directory = os.path.dirname(os.path.realpath(__file__))

magnetic_file = directory+'/Mn3Si2Te6_SS_005K_1T_correction.hkl'
nuclear_file = directory+'/Mn3Si2Te6_SS_100K_0T_correction.hkl'

new_filename = directory+'/Mn3Si2Te6_SS_005K_1T_subtraction.hkl'

magnetic_data = np.loadtxt(magnetic_file)
nuclear_data = np.loadtxt(nuclear_file)

dictionary = {}

for line in magnetic_data:
    
    h, k, l, I, sig, d = line
    
    key = (h,k,l)
    
    dictionary[key] = d, [I], [sig]
    
for line in nuclear_data:
    
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

data = []

for key in dictionary.keys():
    
    item = dictionary[key]
    
    h, k, l = key
    d = item[0]
    
    intens_list = item[1]
    err_list = item[2]
    
    if len(intens_list) > 1:
        I = intens_list[0]-intens_list[1]
        sig = np.sqrt(err_list[0]**2+err_list[1]**2)
        
        line = h, k, l, I, sig, d
        data.append(line)

print(len(data))

hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:8.4f}\n'

with open(new_filename, 'w') as f:
    for line in data:
        f.write(hkl_format.format(*line))