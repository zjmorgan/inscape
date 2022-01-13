# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
     
import sys 
import os 

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/home/zgf/.git/inscape/integration/')

import merge

import imp
imp.reload(merge)

# directories ------------------------------------------------------------------
iptsfolder = '/SNS/CORELLI/IPTS-26829/'
scriptdir = iptsfolder+'shared/scripts/'

# calibration calibration -----------------------------------------------------
detector_calibration = '/SNS/CORELLI/IPTS-23019/shared/germanium_2021b/germanium_2021B_corrected.xml'
tube_calibration = '/SNS/CORELLI/shared/calibration/tube/calibration_corelli_20200109.nxs.h5'

# spectrum file ----------------------------------------------------------------
counts_file = '/SNS/CORELLI/shared/Vanadium/2021B_1005_SlimSAM/sa_SS_195098-195105_w_bkg_sub.nxs'
spectrum_file = '/SNS/CORELLI/shared/Vanadium/2021B_1005_SlimSAM/flux_SS_195098-195105_by_bank_w_bkg_sub.nxs'

ipts = 26829

# Mn3Si2Te6,SS, 0T, 100K, 2021/10
start1 = 221033
stop1 =  221077 # 221122

# # Mn3Si2Te6, SS, 5T, 005K, 2021/10
# start1 = 221129
# stop1 =  221173 # 221308
# # 
# # Mn3Si2Te6, SS, 0T, 005K, 2021/10
start1 = 221309
stop1 =  221488
# #
# # Mn3Si2Te6, SS, 1T, 005K, 2021/10
# start1 = 221684
# stop1 =  221728 # 221773
# 
# UB matrix --------------------------------------------------------------------
ub_file = "/SNS/CORELLI/IPTS-26829/shared/scripts/Mn3Si2Te6_005K_4p75T_SS.mat"

# peak prediction parameters ---------------------------------------------------
reflection_condition = 'Primitive'

# output name ------------------------------------------------------------------

outname = 'Mn3Si2Te6_SS_005K_0T_integration_fixed'

#outname = 'Mn3Si2Te6_SS_005K_5T_integration'
#outname = 'Mn3Si2Te6_SS_005K_0T_integration'
#outname = 'Mn3Si2Te6_SS_005K_1T_integration'

pdf_output = directory+'/peak-envelopes_{}.pdf'.format(outname)

runs = np.arange(start1, stop1+1)
         
merge.pre_integration(ipts, runs, ub_file, spectrum_file, counts_file, 
                      tube_calibration, detector_calibration, reflection_condition)
    
new_peak_dictionary = merge.PeakDictionary(7.0555, 7.0555, 14.1447, 90, 90, 120)
new_peak_dictionary.load(directory+'/Mn3Si2Te6_SS_005K_5T_integration.pkl')

peak_dictionary = merge.PeakDictionary(7.0555, 7.0555, 14.1447, 90, 90, 120)

for r in range(start1,stop1+1):
        
    ows = 'COR_'+str(r)
    opk = ows+'_pks'
    
    peak_dictionary.add_peaks(opk)

peak_envelope = merge.PeakEnvelope(pdf_output)
peak_envelope.show_plots(False)

peaks = peak_dictionary.to_be_integrated()
new_peaks = peak_dictionary.to_be_integrated()

# keys = [(-1,-1,-1)]
# keys = [(-1,-1,1)]
# keys = [(-2,1,1)]
# keys = [(0,3,1)]
# keys = [(2,2,1)]
# keys = [(4,0,1)]

#keys = [(5,-4,2),(-5,4,-2)] #
#for i, key in enumerate(keys[:]):

for i, key in enumerate(list(peaks.keys())[:]):
    print('Integrating peak : {}'.format(key))

    runs, numbers = peaks[key]

    h, k, l = key

    d = peak_dictionary.get_d(h, k, l)
    
    peak_envelope.clear_plots()

    peak_fit, peak_bkg_ratio, peak_score2d = 0, 0, 0
                                                  
    peak = new_peak_dictionary.peak_dict.get(key)
                    
    if peak is not None:
    
        A = peak.get_A()
        Q0 = peak.get_Q()
        D, W = np.linalg.eig(A)
        D = np.diag(D)

        radii = 1/np.sqrt(np.diagonal(D)) 
        
        print('Peak-radii: {}'.format(radii))
                            
        if np.isclose(np.abs(np.linalg.det(W)),1) and (radii < 0.15).all():

            data = merge.norm_integrator(peak_envelope, runs, Q0, D, W)
                                    
            peak_dictionary.integrated_result(key, Q0, A, peak_fit, peak_bkg_ratio, peak_score2d, data)

    h, k, l = key

    peak_dictionary(h, k, l)

    if i % 15 == 0:
        peak_dictionary.save(directory+'/{}.pkl'.format(outname))
    peak_dictionary.save_hkl(directory+'/{}.hkl'.format(outname))

peak_dictionary.save(directory+'/{}.pkl'.format(outname))
peak_dictionary.save_hkl(directory+'/{}.hkl'.format(outname))
peak_envelope.create_pdf()