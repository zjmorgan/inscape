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
iptsfolder = '/SNS/CORELLI/IPTS-23019/'
scriptdir = iptsfolder+'shared/scripts/'

# calibration calibration -----------------------------------------------------
detector_calibration = '/SNS/CORELLI/shared/Calibration/2022A/calibration.xml'
tube_calibration = '/SNS/CORELLI/shared/calibration/tube/calibration_corelli_20200109.nxs.h5'

# spectrum file ----------------------------------------------------------------
counts_file = '/SNS/CORELLI/IPTS-23019/shared/Van5mmx7mm_300K/sa_CCR_221825-221837_w_cal.nxs'
spectrum_file = '/SNS/CORELLI/IPTS-23019/shared/Van5mmx7mm_300K/flux_CCR_221825-221837_by_bank_w_cal.nxs'

ipts = 23019

start = 221998
stop = 222172

# UB matrix --------------------------------------------------------------------
ub_file = os.path.join(directory,'garnet_2022a.mat')

# peak prediction parameters ---------------------------------------------------
reflection_condition = 'Body centred'

# output name ------------------------------------------------------------------
outname = 'garnet_calibration_refined'

pdf_output = os.path.join(directory,'peak-envelopes_{}.pdf'.format(outname))

runs = np.arange(start, stop+1)

merge.pre_integration(ipts, runs, ub_file, spectrum_file, counts_file, 
                      tube_calibration, detector_calibration, reflection_condition)
    
peak_dictionary = merge.PeakDictionary(11.92, 11.92, 11.92, 90, 90, 90)
for r in runs:
        
    ows = 'COR_'+str(r)
    opk = ows+'_pks'
    
    peak_dictionary.add_peaks(opk)

peak_envelope = merge.PeakEnvelope(pdf_output)
peak_envelope.show_plots(False)

peaks = peak_dictionary.to_be_integrated()
      
for i, key in enumerate(list(peaks.keys())[:]):
    print('Integrating peak : {}'.format(key))
    
    runs, numbers = peaks[key]
    
    h, k, l, m, n, p = key
    
    d = peak_dictionary.get_d(h, k, l)
    
    peak_envelope.clear_plots()
    
    for r, n in zip(runs, numbers):
    
        Q, Qx, Qy, Qz, weights, Q0 = merge.box_integrator([r], [n], binsize=0.005, radius=0.15)

        center, variance, peak_fit, peak_bkg_ratio, sig_noise_ratio, peak_total_data_ratio = merge.Q_profile(peak_envelope, key, Q, weights, 
                                                                                                             Q0, radius=0.15, bins=31)
        
        print('Peak-fit Q: {}'.format(peak_fit))
        print('Peak background ratio Q: {}'.format(peak_bkg_ratio))
        print('Signal-noise ratio Q: {}'.format(sig_noise_ratio))
        print('Peak-total to subtrated-data ratio Q: {}'.format(peak_total_data_ratio))
        
        print(weights)
        
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
        
                        peak_dictionary.calibrated_result(key, r, Q0)
            
    h, k, l, m, n, p = key
    
    peak_dictionary(h, k, l)

peak_dictionary.save(directory+'/{}.pkl'.format(outname))
peak_dictionary.save_calibration(directory+'/{}.nxs'.format(outname))
peak_envelope.create_pdf()
