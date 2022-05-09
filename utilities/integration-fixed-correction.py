# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit

import sys 
import os 

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/home/zgf/.git/inscape/integration/')

# calibration calibration -----------------------------------------------------
detector_calibration = '/SNS/CORELLI/IPTS-23019/shared/germanium_2021b/germanium_2021B_corrected.xml'
tube_calibration = '/SNS/CORELLI/shared/calibration/tube/calibration_corelli_20200109.nxs.h5'

counts_file = '/SNS/CORELLI/IPTS-23019/shared/germanium_2021b/sa_CCR_195098-195105_w_bkg_sub_cal.nxs'
if not mtd.doesExist('sa'):
    LoadNexus(Filename=counts_file, OutputWorkspace='sa')
    
ws = mtd['sa']
        
LoadNexus(Filename=tube_calibration, OutputWorkspace='tube_table')
ApplyCalibration(Workspace=ws, CalibrationTable='tube_table')
LoadParameterFile(Workspace=ws, Filename=detector_calibration)

inst = mtd['sa'].getInstrument()

import merge

import imp
imp.reload(merge)

outname = 'Mn3Si2Te6_SS_100K_0T'

peak_dictionary = merge.PeakDictionary(7.0555, 7.0555, 14.1447, 90, 90, 120)
peak_dictionary.load(directory+'/Mn3Si2Te6_SS_005K_5T_integration.pkl')

new_peak_dictionary = merge.PeakDictionary(7.0555, 7.0555, 14.1447, 90, 90, 120)
new_peak_dictionary.load(directory+'/{}_integration_fixed.pkl'.format(outname))

bank_corr = np.loadtxt(os.path.join('/SNS/CORELLI/shared/Integration', 'bank-corrections.txt'), delimiter=',', 
                       dtype={ 'names': ('bank', 'm', 'b'), 'formats': (int,float,float) }, skiprows=1)

coefficients = { }

for item in bank_corr:
    bank, m, b = item
    coefficients[bank] = [m,b]

banks = [ 7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
         24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
         43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
         60, 61, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
         83, 84, 85, 86, 87, 88, 89, 90]

rowA = [1, 29]
rowB = [30, 62]
rowC = [63, 91]

rowA = np.arange(rowA[0],rowA[1]+1).tolist()
rowB = np.arange(rowB[0],rowB[1]+1).tolist()
rowC = np.arange(rowC[0],rowC[1]+1).tolist()

peaks = new_peak_dictionary.to_be_integrated()

for key in list(peaks.keys()):

    # print('Integrating peak : {}'.format(key))

    h, k, l = key

    peak = new_peak_dictionary.peak_dict.get(key)

    pk_by_bank = peak._PeakInformation__bank_num
    pk_by_row = peak._PeakInformation__row
    pk_by_col = peak._PeakInformation__col
    norm_scale = peak._PeakInformation__norm_scale.copy()

    intens = peak.get_merged_intensity()
    print(intens)

    if intens > 0 and not np.isinf(intens):

        for i in range(len(norm_scale)):
            bank = pk_by_bank[i]
            row = pk_by_row[i]
            col = pk_by_col[i]
            y = inst.getComponentByName('bank{}/sixteenpack/tube{}/pixel{}'.format(bank,col,row)).getPos().Y()
            coeffs = coefficients.get(bank)
            if coeffs is not None:
                if bank in rowC:
                    scale = coeffs[0]*np.abs(y)+coeffs[1]
                elif bank in rowB:
                    if y > 0:
                        scale = coeffs[0]*np.abs(y)+coeffs[1]
                    else:
                        scale = 1.0
                else:
                    scale = 1.0
                norm_scale[i] = scale

        peak._PeakInformation__norm_scale = norm_scale

    #intens = peak.get_merged_intensity()
    #print(intens)

def merge_pk_vol(pk_data, pk_norm):

    data = np.sum(pk_data, axis=0)
    norm = np.sum(pk_norm, axis=0)
    print(norm)

    #pk_vol = np.sum(~np.isnan(data/norm))/data.size
    pk_vol = np.sum(~(np.isnan(data/norm)))/data.size

    return pk_vol

for p in range(mtd['iws'].getNumberPeaks()):

    pk = mtd['iws'].getPeak(p)

    h, k, l = pk.getHKL()

    key = h,k,l

    peak = peak_dictionary.peak_dict.get(key)
    new_peak = new_peak_dictionary.peak_dict.get(key)

    if new_peak._PeakInformation__pk_data is not None:

        peak_vol = merge_pk_vol(new_peak._PeakInformation__pk_data, new_peak._PeakInformation__pk_norm)

        #print('[{:4.0f} {:4.0f} {:4.0f}], peak_vol:{:6.2f}'.format(h,k,l,peak_vol))

        if (peak_vol > 0.6):
            intens = peak.get_merged_intensity()
            sig_intens = peak.get_merged_intensity_error()
        else:
            intens = 0
            sig_intens = 0

        pk.setIntensity(intens)
        pk.setSigmaIntensity(sig_intens)

new_peak_dictionary._PeakDictionary__iws = mtd['iws']

new_peak_dictionary.save_hkl(directory+'/{}_correction_fixed.hkl'.format(outname), magnetic=True)
