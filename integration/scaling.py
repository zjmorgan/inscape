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

new_filename = directory+'/Mn3Si2Te6_SS_100K_0T_scaled.hkl'

CCR_data = np.loadtxt(CCR_file)
SS_data = np.loadtxt(SS_file)

dictionary = {}

for line in CCR_data:

    h, k, l, I, sig, d = line

    key = (h,k,-l)

    dictionary[key] = d, [I], [sig]

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

x, y = [], []

for key in dictionary.keys():
    
    item = dictionary[key]

    h, k, l = key
    d = item[0]

    intens_list = item[1]

    if len(intens_list) > 1:
        x.append(intens_list[0])
        y.append(intens_list[1])

x, y = np.array(x), np.array(y)

p = np.polyfit(x, y, 1)

print(p)

data = []

for line in SS_data:
    
    h, k, l, I, sig, d = line

    I /= p[0]
    sig /= np.sqrt(p[0])

    line = h, k, l, I, sig, d
    data.append(line)

hkl_format = '{:4.0f}{:4.0f}{:4.0f}{:8.2f}{:8.2f}{:8.4f}\n'

with open(new_filename, 'w') as f:
    for line in data:
        f.write(hkl_format.format(*line))