# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

directory = '/home/zgf/Documents/data/Mn3Si2Te6'
sys.path.append('/home/zgf/.git/inscape/integration/')

import peak

import imp
imp.reload(peak)

from peak import PeakDictionary

outname = 'Mn3Si2Te6_1T_005K.pkl'

peak_dictionary = PeakDictionary(7.0555, 7.0555, 14.1447, 90, 90, 120)
peak_dictionary.load(os.path.join(directory, outname))

peak_dictionary(0,1,0)