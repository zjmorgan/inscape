# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

file_1 = '/SNS/CORELLI/IPTS-28994/shared/scripts/ErFeO3_CCR_006K_240runs_van_bkg_absorb_cor_integration.hkl'
file_2 = '/SNS/CORELLI/IPTS-28994/shared/scripts/ErFeO3_CCR_125K_240runs_van_bkg_integration.hkl'

data_1 = np.loadtxt(file_1)
data_2 = np.loadtxt(file_2)

dictionary = {}

for line in data_1:
    
    h, k, l, I, sig, d = line
    
    key = (h,k,l)

    if dictionary.get(key) is None:
        dictionary[key] = [[I],[sig],d]

for line in data_2:
    
    h, k, l, I, sig, d = line
    
    key = (h,k,l)

    if dictionary.get(key) is None:
        dictionary[key] = [[I],[sig],d]
    else:
        items = dictionary[key]
        intens_list, err_list, d = items
        intens_list.append(I)
        err_list.append(sig)
        
        dictionary[key] = [intens_list,err_list,d]
        
x, y1, y2 = [], [], []
        
for key in dictionary.keys():

    items = dictionary[key]
    intens_list, err_list, d = items
    
    if len(intens_list) == 2:
       
       x.append(d)
       y1.append(intens_list[0])
       y2.append(intens_list[1])
       
x, y1, y2 = np.array(x), np.array(y1), np.array(y2)

mask = y1/y2 > 4
y1[mask] = np.nan

fig, ax = plt.subplots(1, 1, num='intensity-comparison')
ax.plot(x, y1/y2, '-o', label='With over without absorption correction')
ax.legend()
ax.set_xlabel('d [\u212B]')
ax.set_ylabel('Ratio')
fig.show()   