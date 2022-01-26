# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import sys 
import os 

directory = '/SNS/CORELLI/IPTS-26829/shared/scripts'
sys.path.append('/home/zgf/.git/inscape/integration/')
sys.path.append('/home/zgf/.git/inscape/integration/')

import imp
import peak

imp.reload(peak)

from peak import PeakDictionary, PeakEnvelope

peak_dictionary_1 = PeakDictionary(7.0555, 7.0555, 14.1447, 90, 90, 120)
peak_dictionary_1.load(directory+'/Mn3Si2Te6_electric_S2_006K_00mA_2nd_127runs_integration.pkl')

peak_dictionary_2 = PeakDictionary(7.0555, 7.0555, 14.1447, 90, 90, 120)
peak_dictionary_2.load(directory+'/Mn3Si2Te6_electric_S2_006K_08mA_2nd_127runs_integration.pkl')

ellipsoids = { }

for key in peak_dictionary_1.peak_dict.keys():
    
    peaks = peak_dictionary_1.peak_dict[key]
    
    items = []
    for peak in peaks:
        items.append(peak.get_A())
    ellipsoids[key] = items

for key in peak_dictionary_2.peak_dict.keys():
    
    peaks = peak_dictionary_2.peak_dict[key]
    
    items = []
    for peak in peaks:
        items.append(peak.get_A())
        
    if ellipsoids.get(key) is None:
        ellipsoids[key] = items
    elif len(items) == len(ellipsoids[key]):
        items = [l1 if np.linalg.det(l1) > np.linalg.det(l2) else l2 for l1, l2 in zip(items, ellipsoids[key])]
    else:
        del ellipsoids[key]

for key in peak_dictionary_2.peak_dict.keys():
    
    peaks = peak_dictionary_2.peak_dict[key]
    items = ellipsoids.get(key)
    
    for item, peak in zip(items, peaks):
        peak.set_A(item)
        
peak_dictionary_2.save(directory+'/Mn3Si2Te6_electric_S2_2nd_cooling_reference_ellipsoids.pkl')

peak_dictionary_2(0,0,2)