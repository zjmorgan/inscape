# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import sys 
import os 

directory = '/home/zgf/Documents/data/Mn3Si2Te6/'
sys.path.append('/home/zgf/.git/inscape/integration/')
#sys.path.append('/SNS/users/zgf/.git/inscape/integration')

import imp
import peak
import copy

imp.reload(peak)

from peak import PeakDictionary

n_proc = 12

peak_dict = { }

outname = 'Mn3Si2Te6_1T_005K'

peak_dictionary = PeakDictionary(7.0555, 7.0555, 14.1447, 90, 90, 120)

for i in range(n_proc):
    tmp_peak_dict = peak_dictionary.load_dictionary(os.path.join(directory,outname+'_p{}.pkl'.format(i)))

    if i == 0:
        peak_dict = copy.deepcopy(tmp_peak_dict)

    for key in list(tmp_peak_dict.keys()):
        peaks, tmp_peaks = peak_dict[key], tmp_peak_dict[key]
        
        new_peaks = []
        for peak, tmp_peak in zip(peaks, tmp_peaks):
            if tmp_peak.get_merged_intensity() > 0:
                new_peaks.append(tmp_peak)
            else:
                new_peaks.append(peak)
        peak_dict[key] = new_peaks

peak_dictionary.peak_dict = peak_dict

peak_dictionary._PeakDictionary__repopulate_workspaces()
peak_dictionary.save(os.path.join(directory,outname+'.pkl'))
peak_dictionary.save_hkl(os.path.join(directory,outname+'.hkl'))

for i in range(n_proc):
    os.remove(os.path.join(directory,outname+'_p{}.hkl'.format(i)))
    os.remove(os.path.join(directory,outname+'_p{}.pkl'.format(i)))