# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
     
import sys, os, imp

sys.path.append('/home/zgf/.git/inscape/integration')

import merge, peak, parameters

imp.reload(merge)
imp.reload(peak)
imp.reload(parameters)

from peak import PeakDictionary, PeakEnvelope

CreatePeaksWorkspace(NumberOfPeaks=0, OutputWorkspace='sample', OutputType='LeanElasticPeak')

filename = '/home/zgf/.git/inscape/integration/hb3a.txt' #sys.argv[1]

dictionary = parameters.load_input_file(filename)

a = dictionary['a']
b = dictionary['b']
c = dictionary['c']
alpha = dictionary['alpha']
beta = dictionary['beta']
gamma = dictionary['gamma']

reflection_condition = dictionary['reflection-condition']
group = dictionary['group']

if dictionary['chemical-formula'] is not None:
    chemical_formula = ' '.join(dictionary['chemical-formula'])
else:
    chemical_formula = dictionary['chemical-formula']

z_parameter = dictionary['z-parameter']
sample_radius = dictionary['sample-radius']

facility, instrument = merge.set_instrument(dictionary['instrument'])
ipts = dictionary['ipts']

working_directory = '/{}/{}/IPTS-{}/shared/'.format(facility,instrument,ipts)
shared_directory = '/{}/{}/shared/'.format(facility,instrument)

run_nos = dictionary['runs'] if type(dictionary['runs']) is list else [dictionary['runs']]

runs = []
for r in run_nos:
    if type(r) is list:
        runs += r
    else:
        runs += [r]

experiment = dictionary['experiment']

if dictionary['ub-file'] is not None:
    ub_file = os.path.join(working_directory, dictionary['ub-file'])

split_angle = dictionary['split-angle']

directory = os.path.dirname(os.path.abspath(filename))
outname = dictionary['name']

if dictionary['flux-file'] is not None:
    spectrum_file = os.path.join(shared_directory+'Vanadium', dictionary['flux-file'])
else:
    spectrum_file = None

if dictionary['vanadium-file'] is not None:
    counts_file = os.path.join(shared_directory+'Vanadium', dictionary['vanadium-file'])
else:
    counts_file = None

if dictionary['tube-file'] is not None:
    tube_calibration = os.path.join(shared_directory+'calibration', dictionary['tube-file'])
else:
    tube_calibration = None
    
if dictionary['detector-file'] is not None:
    detector_calibration = os.path.join(shared_directory+'calibration', dictionary['detector-file'])
else:
    detector_calibration = None
    
mod_vector_1 = dictionary['modulation-vector-1']
mod_vector_2 = dictionary['modulation-vector-2']
mod_vector_3 = dictionary['modulation-vector-3']
max_order = dictionary['max-order']
cross_terms = dictionary['cross-terms']

if not all([a,b,c,alpha,beta,gamma]):
    LoadIsawUB(InputWorkspace='sample', Filename=ub_file)
else:
    SetUB(Workspace='sample', a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

volume = mtd['sample'].sample().getOrientedLattice().volume()

ref_dict = dictionary.get('peak-dictionary')

merge.pre_integration(facility, instrument, ipts, runs, ub_file, spectrum_file, counts_file, 
                      tube_calibration, detector_calibration, reflection_condition,
                      mod_vector_1, mod_vector_2, mod_vector_3, max_order, cross_terms,
                      radius=sample_radius, chemical_formula=chemical_formula, volume=volume, z=z_parameter, exp=experiment)

if ref_dict is not None:
    ref_peak_dictionary = PeakDictionary(a, b, c, alpha, beta, gamma)
    ref_peak_dictionary.load(os.path.join(directory, ref_dict))

peak_dictionary = PeakDictionary(a, b, c, alpha, beta, gamma)
peak_dictionary.set_satellite_info(mod_vector_1, mod_vector_2, mod_vector_3, max_order)
peak_dictionary.set_scale_constant(1e+3)

for r in runs:
        
    if facility == 'HFIR':
        ows = '{}_{}_{}'.format(instrument,experiment,r)
    else:
        ows = '{}_{}'.format(instrument,r)
    opk = ows+'_pks'
    
    peak_dictionary.add_peaks(opk)

peak_envelope = PeakEnvelope(directory+'/{}.pdf'.format(outname))
peak_envelope.show_plots(False)

peak_dictionary.split_peaks(split_angle)
peaks = peak_dictionary.to_be_integrated()

for i, key in enumerate(list(peaks.keys())[:]):
    print('Integrating peak : {}'.format(key))
    
    redudancies = peaks[key]
    
    fixed = False
    if ref_dict is not None:
        ref_peaks = ref_peak_dictionary.peak_dict.get(key)
        if ref_peaks is not None:
            if len(ref_peaks) == len(redudancies):
                fixed = True
                
    h, k, l, m, n, p = key
    
    d = peak_dictionary.get_d(h, k, l, m, n, p)
    
    for j, redudancy in enumerate(redudancies):
    
        runs, numbers = redudancy
        
        peak_envelope.clear_plots()
        
        if fixed:
            
            ref_peak = ref_peaks[j]
            Q0 = ref_peak.get_Q()
            A = ref_peak.get_A()
            D, W = np.linalg.eig(A)
            D = np.diag(D)

            radii = 1/np.sqrt(np.diagonal(D)) 

            peak_fit, peak_bkg_ratio, peak_score2d = 0, 0, 0

            if np.isclose(np.abs(np.linalg.det(W)),1) and (radii < 0.3).all() and (radii > 0).all():

                data = merge.norm_integrator(peak_envelope, instrument, runs, Q0, D, W, fit=False)

                peak_dictionary.integrated_result(key, Q0, A, peak_fit, peak_bkg_ratio, peak_score2d, data, j)
        
        else:
        
            remove = False
                    
            Q, Qx, Qy, Qz, weights, Q0 = merge.box_integrator(instrument, runs, numbers, binsize=0.005, radius=0.15, exp=experiment)

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

            if (peak_score2d > 2 and not np.isinf(peak_score2d) and not np.isnan(peak_score2d) and np.linalg.norm(center2d) < 0.15 and sig_noise_ratio2d > 3):

                remove = True

            Qc, A, W, D = merge.ellipsoid(Q0, center, variance, center2d, covariance2d, 
                                          n, u, v, xsigma=4, lscale=5)
                                          
            peak_envelope.plot_projection_ellipse(*peak.draw_ellispoid(center2d, covariance2d, lscale=5))

            center, variance, peak_fit, peak_bkg_ratio, sig_noise_ratio, peak_total_data_ratio = merge.extracted_Q_profile(peak_envelope, key, Q, Qx, Qy, Qz, weights, 
                                                                                                                           Q0, u, v, center, variance, center2d, covariance2d, bins=21)

            print('Peak-fit Q second pass: {}'.format(peak_fit))
            print('Peak background ratio Q second pass: {}'.format(peak_bkg_ratio))
            print('Signal-noise ratio Q second pass: {}'.format(sig_noise_ratio))
            print('Peak-total to subtrated-data ratio Q: {}'.format(peak_total_data_ratio))
            
            if (sig_noise_ratio > 3 and 3*np.sqrt(variance) < 0.1 and np.abs(np.linalg.norm(Qc)-center) < 0.1 and peak_total_data_ratio < 3):

                remove = True

            if not np.isnan(covariance2d).any():

                Q0, A, W, D = merge.ellipsoid(Q0, center, variance, center2d, covariance2d, 
                                              n, u, v, xsigma=4, lscale=5)

                radii = 1/np.sqrt(np.diagonal(D)) 

                print('Peak-radii: {}'.format(radii))

                if np.isclose(np.abs(np.linalg.det(W)),1) and (radii < 0.3).all() and (radii > 0).all():

                    data = merge.norm_integrator(peak_envelope, facility, instrument, runs, Q0, D, W, exp=experiment, fit=False)

                    peak_dictionary.integrated_result(key, Q0, A, peak_fit, peak_bkg_ratio, peak_score2d, data, j)

                    peak_envelope.write_figure()

                else:

                    remove = True

            if remove:

                peak_dictionary.partial_result(key, Q0, A, peak_fit, peak_bkg_ratio, peak_score2d, j)

    if i % 15 == 0:

        peak_dictionary.save(directory+'/{}.pkl'.format(outname))
        peak_dictionary.save_hkl(directory+'/{}.hkl'.format(outname))        

peak_dictionary.save(directory+'/{}.pkl'.format(outname))
peak_dictionary.save_hkl(directory+'/{}.hkl'.format(outname))
peak_envelope.create_pdf()

peak_dictionary(1,0,1)