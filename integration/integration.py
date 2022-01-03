# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
     
import sys 
import os 

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/SNS/CORELLI/shared/Integration')

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
counts_file = '/SNS/CORELLI/IPTS-23019/shared/germanium_2021b/sa_CCR_195098-195105_w_bkg_sub_cal.nxs'
spectrum_file = '/SNS/CORELLI/IPTS-23019/shared/germanium_2021b/flux_CCR_195098-195105_by_bank_w_bkg_sub_w_cal.nxs'

ipts = 26829
 
# Mn3Si2Te6, 005 K, 2021/07
#start1 = 198009
#stop1 = 198188
#sampleT = '005K'

# Mn3Si2Te6, 100 K, 2021/07
start1 = 198190
stop1 = 198369
sampleT = '100K'

# UB matrix --------------------------------------------------------------------
ub_file = '/SNS/CORELLI/IPTS-26829/shared/scripts/Mn3Si2Te6_4th_'+sampleT+'.mat'

# peak prediction parameters ---------------------------------------------------
reflection_condition = 'Primitive'

# output name ------------------------------------------------------------------

outname = 'Mn3Si2Te6_CCR_'+sampleT+'lscale5_integration'

pdf_output = directory+'/peak-envelopes_{}.pdf'.format(outname)

runs = np.arange(start1, stop1)
         
merge.pre_integration(ipts, runs, ub_file, spectrum_file, counts_file, 
                      tube_calibration, detector_calibration, reflection_condition)
    
peak_dictionary = merge.PeakDictionary(7.0555, 7.0555, 14.1447, 90, 90, 120)

for r in range(start1,stop1+1):
        
    ows = 'COR_'+str(r)
    opk = ows+'_pks'
    
    peak_dictionary.add_peaks(opk)

peak_envelope = merge.PeakEnvelope(pdf_output)
peak_envelope.show_plots(False)

peaks = peak_dictionary.to_be_integrated()

# keys = [(-1,-1,-1)]
# keys = [(-1,-1,1)]
# keys = [(-2,1,1)]
# keys = [(0,3,1)]
# keys = [(2,2,1)]
# keys = [(4,0,1)]

#keys = [(1,1,-1)]
#for i, key in enumerate(keys[:]):
      
for i, key in enumerate(list(peaks.keys())[:]):
    print('Integrating peak : {}'.format(key))
    
    runs, numbers = peaks[key]
    
    h, k, l = key
    
    d = peak_dictionary.get_d(h, k, l)
    
    peak_envelope.clear_plots()
    
    Q, Qx, Qy, Qz, weights, Q0 = merge.box_integrator(runs, numbers, binsize=0.005, radius=0.15)

    center, variance, peak_fit, peak_bkg_ratio, sig_noise_ratio, peak_total_data_ratio = merge.Q_profile(peak_envelope, key, Q, weights, 
                                                                                                         Q0, radius=0.15, bins=31)
        
    print('Peak-fit Q: {}'.format(peak_fit))
    print('Peak background ratio Q: {}'.format(peak_bkg_ratio))
    print('Signal-noise ratio Q: {}'.format(sig_noise_ratio))
    print('Peak-total to subtrated-data ratio Q: {}'.format(peak_total_data_ratio))
    
    if (sig_noise_ratio > 3 and 3*np.sqrt(variance) < 0.1 and np.abs(np.linalg.norm(Q0)-center) < 0.1):

        n, u, v = merge.projection_axes(Q0)

        center2d, covariance2d, peak_score2d, sig_noise_ratio2d = merge.projected_profile(peak_envelope, d, Q, Qx, Qy, Qz, weights,
                                                                                         Q0, u, v, center, variance, radius=0.1,
                                                                                         bins=21, bins2d=21)
        
        print('Peak-score 2d: {}'.format(peak_score2d))
        print('Signal-noise ratio 2d: {}'.format(sig_noise_ratio2d))
        
        if peak_score2d > 2 and not np.isinf(peak_score2d) and not np.isnan(peak_score2d) and np.linalg.norm(center2d) < 0.1 and sig_noise_ratio2d > 3:
            
            Qc, A, W, D = merge.ellipsoid(peak_envelope, Q0, 
                                          center, variance, 
                                          center2d, covariance2d, 
                                          n, u, v, xsigma=4, lscale=5, plot='first')
                            
            center, variance, peak_fit, peak_bkg_ratio, sig_noise_ratio, peak_total_data_ratio = merge.extracted_Q_profile(peak_envelope, key, Q, Qx, Qy, Qz, weights, 
                                                                                                                           Q0, u, v, center, variance, center2d, covariance2d, bins=21)
                                                                                                    
            print('Peak-fit Q second pass: {}'.format(peak_fit))
            print('Peak background ratio Q second pass: {}'.format(peak_bkg_ratio))
            print('Signal-noise ratio Q second pass: {}'.format(sig_noise_ratio))
            print('Peak-total to subtrated-data ratio Q: {}'.format(peak_total_data_ratio))
            
            if (sig_noise_ratio > 3 and 3*np.sqrt(variance) < 0.1 and np.abs(np.linalg.norm(Qc)-center) < 0.1 and peak_total_data_ratio < 2):

                if not np.isnan(covariance2d).any():
                    
                    Q0, A, W, D = merge.ellipsoid(peak_envelope, Q0, 
                                                  center, variance, 
                                                  center2d, covariance2d, 
                                                  n, u, v, xsigma=4, lscale=5, plot=None)  

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
