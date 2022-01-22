# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

def f(filename):
    
    ws = os.path.basename(filename)

    LoadNexus(Filename=filename, OutputWorkspace=ws)

    ConvertPeaksWorkspace(PeakWorkspace=mtd[ws], 
                          InstrumentWorkspace=mtd['super_resolution'],
                          OutputWorkspace=mtd[ws])
                                            
    SaveNexus(InputWorkspace=mtd[ws], Filename=filename)