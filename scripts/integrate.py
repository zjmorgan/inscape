# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

import sys, os, imp

sys.path.append('/home/zgf/.git/inscape/integration')

import multiprocessing

import merge, peak, parameters

imp.reload(merge)
imp.reload(peak)
imp.reload(parameters)

from peak import PeakDictionary

CreatePeaksWorkspace(NumberOfPeaks=0, OutputWorkspace='sample', OutputType='LeanElasticPeak')

filename, n_proc = sys.argv[1] sys.argv[2]

if n_proc > 16:
    n_proc = 16

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

run_labels = '_'.join([str(r[0])+'-'+str(r[-1]) if type(r) is list else str(r) for r in run_nos if any([item is list for item in run_nos])])

if run_labels == '':
    run_labels = str(run_nos[0])+'-'+str(run_nos[-1])

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

merge.load_normalization_calibration(facility, spectrum_file, counts_file,
                                     tube_calibration, detector_calibration)

if facility == 'HFIR':
    ows = '{}_{}'.format(instrument,experiment)+'_{}'
else:
    ows = '{}'.format(instrument)+'_{}'

opk = ows+'_pk'
omd = ows+'_md'

for r in runs:
    if mtd.doesExist(omd.format(r)):
        DeleteWorkspace(omd.format(r))

tmp = opk.format(run_labels)
filename = os.path.join(directory, tmp+'.nxs')
if os.path.exists(filename) and not mtd.doesExist(tmp):
    LoadNexus(Filename=filename, OutputWorkspace=tmp)
    for r in runs:
        FilterPeaks(InputWorkspace=tmp, 
                    FilterVariable='RunNumber',
                    FilterValue=r,
                    Operator='=',
                    OutputWorkspace=opk.format(r))

if not mtd.doesExist(tmp):

    with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:

        split_runs = [split.tolist() for split in np.array_split(runs, n_proc)]

        args = [directory, facility, instrument, ipts, ub_file, reflection_condition,
                mod_vector_1, mod_vector_2, mod_vector_3, max_order, cross_terms, experiment]

        join_args = [(split, *args) for split in split_runs]

        pool.starmap(merge.pre_integration, join_args)

for r in runs:
    if not mtd.doesExist(opk.format(r)):
        filename = os.path.join(directory, opk.format(r)+'.nxs')
        LoadNexus(Filename=filename, OutputWorkspace=opk.format(r))

if not mtd.doesExist(tmp):
    CreatePeaksWorkspace(InstrumentWorkspace='sa', NumberOfPeaks=0, OutputType='Peak', OutputWorkspace=tmp)
    for r in runs:
        CombinePeaksWorkspaces(LHSWorkspace=opk.format(r), RHSWorkspace=tmp, OutputWorkspace=tmp)
        filename = os.path.join(directory, opk.format(r)+'.nxs')
        os.remove(filename)
    filename = os.path.join(directory, tmp+'.nxs')
    FilterPeaks(InputWorkspace=tmp,
                FilterVariable='QMod',
                FilterValue=0,
                Operator='>',
                OutputWorkspace=tmp)
    SaveNexus(InputWorkspace=tmp, Filename=filename)

if ref_dict is not None:
    ref_peak_dictionary = PeakDictionary(a, b, c, alpha, beta, gamma)
    ref_peak_dictionary.load(os.path.join(directory, ref_dict))
else:
    ref_peak_dictionary = None

peak_dictionary = PeakDictionary(a, b, c, alpha, beta, gamma)
peak_dictionary.set_satellite_info(mod_vector_1, mod_vector_2, mod_vector_3, max_order)
peak_dictionary.set_scale_constant(1e+6)

for r in runs:
    peak_dictionary.add_peaks(opk.format(r))

peak_dictionary.split_peaks(split_angle)
peaks = peak_dictionary.to_be_integrated()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:

        keys = list(peaks.keys())
        split_keys = [split.tolist() for split in np.array_split(keys, n_proc)]

        args = [ref_peak_dictionary, ref_dict,
                directory, facility, instrument, ipts, runs,
                split_angle, a, b, c, alpha, beta, gamma,
                mod_vector_1, mod_vector_2, mod_vector_3, max_order,
                sample_radius, chemical_formula, volume, z_parameter, experiment]

        join_args = [(split, outname+'_p{}'.format(i), *args) for i, split in enumerate(split_keys)]

        pool.starmap(merge.integration_loop, join_args)