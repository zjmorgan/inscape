# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

shared = '/SNS/CORELLI/IPTS-28994/shared/'
directory = shared+'ErFeO3_300K_CCR_202208_V0725_2p5_8'
sys.path.append('/SNS/software/scd/dev/inscape_dev/integration/')

import peak

import imp
imp.reload(peak)

from peak import PeakDictionary

outname = 'ErFeO3_300K_28mg_full_202208.pkl'

peak_dictionary = PeakDictionary()
peak_dictionary.load_cif(os.path.join(shared, 'ErFeO3.cif'))
peak_dictionary.load(os.path.join(directory, outname))

peak_dictionary(1,3,0)