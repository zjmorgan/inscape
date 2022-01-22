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

instrument = 'CORELLI'
ipts = 28994

# calibration calibration -----------------------------------------------------
detector_calibration = '/SNS/CORELLI/shared/Calibration/2022A/calibration.xml'
tube_calibration = '/SNS/CORELLI/shared/calibration/tube/calibration_corelli_20200109.nxs.h5'

# spectrum file ----------------------------------------------------------------
#counts_file = '/SNS/CORELLI/shared/Vanadium/2022A_0106_CCR/sa_CCR_221825-221837_w_cal.nxs'
#spectrum_file = '/SNS/CORELLI/shared/Vanadium/2022A_0106_CCR/flux_CCR_221825-221837_by_bank_w_cal.nxs'

counts_file = '/SNS/CORELLI/shared/Vanadium/2022A_0106_CCR/sa_CCR_221825-221837_w_bkg_sub_w_cal.nxs'
spectrum_file = '/SNS/CORELLI/shared/Vanadium/2022A_0106_CCR/flux_CCR_221825-221837_by_bank_w_bkg_sub_w_cal.nxs'

# UB matrix --------------------------------------------------------------------
ub_file = "/SNS/CORELLI/IPTS-28994/shared/scripts/ErFeO3_cooling.mat"

# peak prediction parameters ---------------------------------------------------
reflection_condition = 'Primitive'

# cooling
temp = 'cooling'
start = 225007
stop = 225112 # down to 115 K

# 6K
temp = '006K_10runs_van_bkg_'
start = 225150
#stop = start+3
stop = start+90
#stop = 225389+1

#start = 225270
#stop = start+120

# 125K
# temp = '125K_240runs_van_bkg'
# start = 225489
# stop = 225728+1

runs = np.arange(start, stop)
#runs = np.arange(start, start+121)
         
# output name ------------------------------------------------------------------
outname = 'ErFeO3_CCR_'+temp+'_integration'
pdf_output = directory+'/peak-envelopes_{}.pdf'.format(outname)

peak_dictionary = merge.PeakDictionary(5.265, 5.5761, 7.5747, 90, 90, 90)
peak_dictionary.set_scale_constant(1e+4)

merge.pre_integration(instrument, ipts, runs, ub_file, spectrum_file, counts_file, 
                      tube_calibration, detector_calibration, reflection_condition)

for r in runs:
        
    ows = '{}_{}'.format(instrument,r)
    opk = ows+'_pks'
    
    peak_dictionary.add_peaks(opk)

peak_envelope = merge.PeakEnvelope(pdf_output)
peak_envelope.show_plots(False)

peak_dictionary.split_peaks(5)
peaks = peak_dictionary.to_be_integrated()

for i, key in enumerate(list(peaks.keys())[:]):
    print('Integrating peak : {}'.format(key))
    
    redudancies = peaks[key]
    
    h, k, l, m, n, p = key
    
    d = peak_dictionary.get_d(h, k, l)
    
    for redudancy in redudancies:
    
        runs, numbers = redudancy
                
        peak_envelope.clear_plots()
        
        remove = False
                
        Q, Qx, Qy, Qz, weights, Q0 = merge.box_integrator(instrument, runs, numbers, binsize=0.005, radius=0.15)

        center, variance, peak_fit, peak_bkg_ratio, sig_noise_ratio, peak_total_data_ratio = merge.Q_profile(peak_envelope, key, Q, weights, 
                                                                                                             Q0, radius=0.15, bins=31)
            
        print('Peak-fit Q: {}'.format(peak_fit))
        print('Peak background ratio Q: {}'.format(peak_bkg_ratio))
        print('Signal-noise ratio Q: {}'.format(sig_noise_ratio))
        print('Peak-total to subtrated-data ratio Q: {}'.format(peak_total_data_ratio))
        
        if (sig_noise_ratio > 3 and 3*np.sqrt(variance) < 0.1 and np.abs(np.linalg.norm(Q0)-center) < 0.1):

            remove = True

        n, u, v = merge.projection_axes(Q0)

        center2d, covariance2d, peak_score2d, sig_noise_ratio2d = merge.projected_profile(peak_envelope, d, Q, Qx, Qy, Qz, weights,
                                                                                         Q0, u, v, center, variance, radius=0.1,
                                                                                         bins=21, bins2d=21)
        
        print('Peak-score 2d: {}'.format(peak_score2d))
        print('Signal-noise ratio 2d: {}'.format(sig_noise_ratio2d))
        
        if peak_score2d > 2 and not np.isinf(peak_score2d) and not np.isnan(peak_score2d) and np.linalg.norm(center2d) < 0.15 and sig_noise_ratio2d > 3:
            
            remove = True
            
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
        
        if (sig_noise_ratio > 3 and 3*np.sqrt(variance) < 0.1 and np.abs(np.linalg.norm(Qc)-center) < 0.1 and peak_total_data_ratio < 3):

            remove = True

        if not np.isnan(covariance2d).any():
            
            Q0, A, W, D = merge.ellipsoid(peak_envelope, Q0, 
                                          center, variance, 
                                          center2d, covariance2d, 
                                          n, u, v, xsigma=4, lscale=5, plot=None)  

            radii = 1/np.sqrt(np.diagonal(D)) 

            print('Peak-radii: {}'.format(radii))

            if np.isclose(np.abs(np.linalg.det(W)),1) and (radii < 0.3).all():

                data = merge.norm_integrator(peak_envelope, instrument, runs, Q0, D, W)

                peak_dictionary.integrated_result(key, Q0, A, peak_fit, peak_bkg_ratio, peak_score2d, data)

            else:

                remove = True

        if remove:

            peak_dictionary.partial_result(key, Q0, A, peak_fit, peak_bkg_ratio, peak_score2d)

peak_dictionary.save(directory+'/{}.pkl'.format(outname))
peak_dictionary.save_hkl(directory+'/{}.hkl'.format(outname))
peak_envelope.create_pdf()

peak_dictionary(1, 1, 2)