import logging
logging.disable(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')

import sys, os, re, imp, copy, shutil

# os.environ['OPENMP_NUM_THREADS'] = '1'

import multiprocessing

from mantid import config
config.setLogLevel(0, quiet=True)

from mantid.simpleapi import *
import numpy as np

import itertools

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

import merge, peak, parameters

imp.reload(merge)
imp.reload(peak)
imp.reload(parameters)

from peak import PeakDictionary, PeakStatistics
from PyPDF2 import PdfFileMerger

from mantid.kernel import V3D
from mantid.geometry import PointGroupFactory, SpaceGroupFactory

from scipy.spatial import KDTree

filename, n_proc = sys.argv[1], int(sys.argv[2])

if n_proc > os.cpu_count():
    n_proc = os.cpu_count()

scale_constant = 1e+4

if __name__ == '__main__':

    multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.freeze_support()

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    CreateSampleWorkspace(OutputWorkspace='sample')

    dictionary = parameters.load_input_file(filename)

    a = dictionary['a']
    b = dictionary['b']
    c = dictionary['c']
    alpha = dictionary['alpha']
    beta = dictionary['beta']
    gamma = dictionary['gamma']

    min_d = dictionary.get('minimum-d-spacing')

    if min_d is None:
        min_d = 0.7

    adaptive_scale = dictionary.get('adaptive-scale')
    scale_factor = dictionary.get('scale-factor')

    if scale_factor is None:
        scale_factor = 1

    reflection_condition = dictionary['reflection-condition']
    group = dictionary['group']

    if reflection_condition == 'P':
        reflection_condition = 'Primitive'
    elif reflection_condition == 'F':
        reflection_condition = 'All-face centred'
    elif reflection_condition == 'I':
        reflection_condition = 'Body centred'
    elif reflection_condition == 'A':
        reflection_condition = 'A-face centred'
    elif reflection_condition == 'B':
        reflection_condition = 'B-face centred'
    elif reflection_condition == 'C':
        reflection_condition = 'C-face centred'
    elif reflection_condition == 'R' or reflection_condition == 'Robv':
        reflection_condition = 'Rhombohedrally centred, obverse'
    elif reflection_condition == 'Rrev':
        reflection_condition = 'Rhombohedrally centred, reverse'
    elif reflection_condition == 'H':
         reflection_condition = 'Hexagonally centred, reverse'

    pgs = [pg.replace(' ', '') for pg in PointGroupFactory.getAllPointGroupSymbols()]
    sgs = [sg.replace(' ', '') for sg in SpaceGroupFactory.getAllSpaceGroupSymbols()]

    sg = None
    pg = None

    if type(group) is int:
        sg = SpaceGroupFactory.subscribedSpaceGroupSymbols(group)[0]
    elif group in pgs:
        pg = PointGroupFactory.createPointGroup(PointGroupFactory.getAllPointGroupSymbols()[pgs.index(group)]).getPointGroup().getHMSymbol()
    elif group in sgs:
        sg = SpaceGroupFactory.createSpaceGroup(SpaceGroupFactory.getAllSpaceGroupSymbols()[sgs.index(group)]).getHMSymbol()

    if dictionary.get('chemical-formula') is not None:
        chemical_formula = ''.join([' '+item if item.isalpha() else item for item in re.findall(r'[A-Za-z]+|\d+', dictionary['chemical-formula'])]).lstrip(' ')
    else:
        chemical_formula = None

    z_parameter = dictionary['z-parameter']
    sample_mass = dictionary['sample-mass']
    vanadium_mass = dictionary.get('vanadium-mass')

    if vanadium_mass is None:
        vanadium_mass = 0

    facility, instrument = parameters.set_instrument(dictionary['instrument'])
    ipts = dictionary['ipts']

    working_directory = '/{}/{}/IPTS-{}/shared/'.format(facility,instrument,ipts)
    shared_directory = '/{}/{}/shared/'.format(facility,instrument)

    run_nos = dictionary['runs'] if type(dictionary['runs']) is list else [dictionary['runs']]

    run_labels = '_'.join([str(r[0])+'-'+str(r[-1]) if type(r) is list else str(r) for r in run_nos if any([type(item) is list for item in run_nos])])

    if run_labels == '':
        run_labels = str(run_nos[0])+'-'+str(run_nos[-1])

    runs = []
    for r in run_nos:
        if type(r) is list:
            runs += r
        else:
            runs += [r]

    m_proc = len(runs) if n_proc > len(runs) else n_proc

    experiment = dictionary.get('experiment')

    if dictionary['ub-file'] is not None:
        ub_file = os.path.join(working_directory, dictionary['ub-file'])
        if '*' in ub_file:
            ub_file = [ub_file.replace('*', str(run)) for run in run_nos]
    else:
        ub_file = None

    split_angle = dictionary['split-angle']

    directory = os.path.dirname(os.path.abspath(filename))
    outname = dictionary['name']

    outdir = os.path.join(directory, outname)
    if not os.path.exists(outdir):
        #shutil.rmtree(outdir)
        os.mkdir(outdir)
    else:
        items = os.listdir(outdir)
        for item in items:
            if item.endswith('.png'):
                os.remove(os.path.join(outdir, item))
            elif item.endswith('.jpg'):
                os.remove(os.path.join(outdir, item))
            elif item.endswith('.txt'):
                os.remove(os.path.join(outdir, item))

    parameters.output_input_file(filename, directory, outname)

    if dictionary['flux-file'] is not None:
        spectrum_file = os.path.join(shared_directory+'Vanadium', dictionary['flux-file'])
    else:
        spectrum_file = None

    if dictionary['vanadium-file'] is not None:
        counts_file = os.path.join(shared_directory+'Vanadium', dictionary['vanadium-file'])
    else:
        counts_file = None

    if dictionary.get('tube-file') is not None:
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
        if ub_file is not None:
            if type(ub_file) is list:
                LoadIsawUB(InputWorkspace='sample', Filename=ub_file[0])
            else:
                LoadIsawUB(InputWorkspace='sample', Filename=ub_file)
            a = mtd['sample'].sample().getOrientedLattice().a()
            b = mtd['sample'].sample().getOrientedLattice().b()
            c = mtd['sample'].sample().getOrientedLattice().c()
            alpha = mtd['sample'].sample().getOrientedLattice().alpha()
            beta = mtd['sample'].sample().getOrientedLattice().beta()
            gamma = mtd['sample'].sample().getOrientedLattice().gamma()

    ref_dict = dictionary.get('peak-dictionary')

    merge.load_normalization_calibration(facility, instrument, spectrum_file, counts_file,
                                         tube_calibration, detector_calibration)

    if instrument == 'HB3A':
        ows = '{}_{}'.format(instrument,experiment)+'_{}'
    else:
        ows = '{}'.format(instrument)+'_{}'

    opk = ows+'_pk'
    omd = ows+'_md'

    for r in runs:
        if mtd.doesExist(omd.format(r)):
            DeleteWorkspace(omd.format(r))

    tmp = ows.format(run_labels)

    if os.path.exists(os.path.join(directory, tmp+'_pk.nxs')) and not mtd.doesExist(tmp):

        LoadNexus(Filename=os.path.join(directory, tmp+'_pk_lean.nxs'), OutputWorkspace=tmp+'_lean')
        LoadNexus(Filename=os.path.join(directory, tmp+'_pk.nxs'), OutputWorkspace=tmp)
        LoadIsawUB(InputWorkspace=tmp+'_lean', Filename=os.path.join(directory, tmp+'.mat'))
        LoadIsawUB(InputWorkspace=tmp, Filename=os.path.join(directory, tmp+'.mat'))

        for r in runs:
            FilterPeaks(InputWorkspace=tmp, 
                        FilterVariable='RunNumber',
                        FilterValue=r,
                        Operator='=',
                        OutputWorkspace=opk.format(r))
            FilterPeaks(InputWorkspace=tmp+'_lean', 
                        FilterVariable='RunNumber',
                        FilterValue=r,
                        Operator='=',
                        OutputWorkspace=opk.format(r)+'_lean')

    if not mtd.doesExist(tmp):

        split_runs = [split.tolist() for split in np.array_split(runs, m_proc)]

        args = [outdir, directory, facility, instrument, ipts, runs, ub_file, reflection_condition, min_d,
                spectrum_file, counts_file, tube_calibration, detector_calibration,
                mod_vector_1, mod_vector_2, mod_vector_3, max_order, cross_terms, experiment, tmp]

        join_args = [(split, outname+'_p{}'.format(i), *args) for i, split in enumerate(split_runs)]

        # merge.pre_integration(*join_args)

        with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:
            pool.starmap(merge.pre_integration, join_args)
            pool.close()
            pool.join()

    if not mtd.doesExist(tmp):   

        if mtd.doesExist('sa'):
            CreatePeaksWorkspace(InstrumentWorkspace='sa', NumberOfPeaks=0, OutputType='Peak', OutputWorkspace=tmp)
        else:
            CreatePeaksWorkspace(InstrumentWorkspace='van', NumberOfPeaks=0, OutputType='Peak', OutputWorkspace=tmp)

        CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace=tmp+'_lean')

        CreateEmptyTableWorkspace(OutputWorkspace='run_info')

        mtd['run_info'].addColumn('Int', 'RunNumber')
        mtd['run_info'].addColumn('Double', 'Scale')

        for i in range(m_proc):
            partname = outname+'_p{}'.format(i)

            LoadNexus(Filename=os.path.join(outdir, partname+'_log.nxs'), OutputWorkspace=partname+'_log')
            for j in range(mtd[partname+'_log'].rowCount()):
                items = mtd[partname+'_log'].row(j)
                mtd['run_info'].addRow(list(items.values()))
            DeleteWorkspace(partname+'_log')

            LoadNexus(Filename=os.path.join(outdir, partname+'_pk.nxs'), OutputWorkspace=partname+'_pk')
            LoadIsawUB(InputWorkspace=partname+'_pk', Filename=os.path.join(outdir, partname+'.mat'))
            CombinePeaksWorkspaces(LHSWorkspace=partname+'_pk', RHSWorkspace=tmp, OutputWorkspace=tmp)
            LoadIsawUB(InputWorkspace=tmp, Filename=os.path.join(outdir, partname+'.mat'))
            DeleteWorkspace(partname+'_pk')

            LoadNexus(Filename=os.path.join(outdir, partname+'_pk_lean.nxs'), OutputWorkspace=partname+'_pk_lean')
            LoadIsawUB(InputWorkspace=partname+'_pk_lean', Filename=os.path.join(outdir, partname+'.mat'))
            CombinePeaksWorkspaces(LHSWorkspace=partname+'_pk_lean', RHSWorkspace=tmp+'_lean', OutputWorkspace=tmp+'_lean')
            LoadIsawUB(InputWorkspace=tmp+'_lean', Filename=os.path.join(outdir, partname+'.mat'))
            DeleteWorkspace(partname+'_pk_lean')

            os.remove(os.path.join(outdir, partname+'_log.nxs'))
            os.remove(os.path.join(outdir, partname+'_pk_lean.nxs'))
            os.remove(os.path.join(outdir, partname+'_pk.nxs'))
            os.remove(os.path.join(outdir, partname+'.mat'))

        SaveNexus(InputWorkspace='run_info', Filename=os.path.join(directory, tmp+'_log.nxs'))
        SaveNexus(InputWorkspace=tmp+'_lean', Filename=os.path.join(directory, tmp+'_pk_lean.nxs'))
        SaveNexus(InputWorkspace=tmp, Filename=os.path.join(directory, tmp+'_pk.nxs'))
        SaveIsawUB(InputWorkspace=tmp, Filename=os.path.join(directory, tmp+'.mat'))

        for r in runs:
            FilterPeaks(InputWorkspace=tmp, 
                        FilterVariable='RunNumber',
                        FilterValue=r,
                        Operator='=',
                        OutputWorkspace=opk.format(r))
            FilterPeaks(InputWorkspace=tmp+'_lean', 
                        FilterVariable='RunNumber',
                        FilterValue=r,
                        Operator='=',
                        OutputWorkspace=opk.format(r)+'_lean')

    if mtd.doesExist('sa'):
        DeleteWorkspace('sa')

    if mtd.doesExist('flux'):
        DeleteWorkspace('flux')

    if mtd.doesExist('van'):
        DeleteWorkspace('van')

    if ref_dict is not None:
        ref_peak_dictionary = PeakDictionary(a, b, c, alpha, beta, gamma)
        ref_peak_dictionary.load(os.path.join(directory, ref_dict))
    else:
        ref_peak_dictionary = None

    peak_dictionary = PeakDictionary(a, b, c, alpha, beta, gamma)
    peak_dictionary.set_satellite_info(mod_vector_1, mod_vector_2, mod_vector_3, max_order)
    peak_dictionary.set_material_info(chemical_formula, z_parameter, sample_mass)
    peak_dictionary.set_scale_constant(scale_constant)

    for r in runs:

        if max_order > 0:

            ol = mtd[opk.format(r)].sample().getOrientedLattice()
            ol.setMaxOrder(max_order)

            ol.setModVec1(V3D(*mod_vector_1))
            ol.setModVec2(V3D(*mod_vector_2))
            ol.setModVec3(V3D(*mod_vector_3))

            UB = ol.getUB()

            mod_HKL = np.column_stack((mod_vector_1,mod_vector_2,mod_vector_3))
            mod_UB = np.dot(UB, mod_HKL)

            ol.setModUB(mod_UB)

            mod_1 = np.linalg.norm(mod_vector_1) > 0
            mod_2 = np.linalg.norm(mod_vector_2) > 0
            mod_3 = np.linalg.norm(mod_vector_3) > 0

            ind_1 = np.arange(-max_order*mod_1,max_order*mod_1+1).tolist()
            ind_2 = np.arange(-max_order*mod_2,max_order*mod_2+1).tolist()
            ind_3 = np.arange(-max_order*mod_3,max_order*mod_3+1).tolist()

            if cross_terms:
                iter_mnp = list(itertools.product(ind_1,ind_2,ind_3))
            else:
                iter_mnp = list(set(list(itertools.product(ind_1,[0],[0]))\
                                  + list(itertools.product([0],ind_2,[0]))\
                                  + list(itertools.product([0],[0],ind_3))))

            iter_mnp = [iter_mnp[s] for s in np.lexsort(np.array(iter_mnp).T, axis=0)]

            for pn in range(mtd[opk.format(r)].getNumberPeaks()):
                pk = mtd[opk.format(r)].getPeak(pn)
                hkl = pk.getHKL()
                for m, n, p in iter_mnp:
                    d_hkl = m*np.array(mod_vector_1)\
                          + n*np.array(mod_vector_2)\
                          + p*np.array(mod_vector_3)
                    HKL = np.round(hkl-d_hkl,4)
                    mnp = [m,n,p]
                    H, K, L = HKL
                    h, k, l = int(H), int(K), int(L)
                    if reflection_condition == 'Primitive':
                        allowed = True
                    elif reflection_condition == 'C-face centred':
                        allowed = (h + k) % 2 == 0
                    elif reflection_condition == 'A-face centred':
                        allowed = (k + l) % 2 == 0
                    elif reflection_condition == 'B-face centred':
                        allowed = (h + l) % 2 == 0
                    elif reflection_condition == 'Body centred':
                        allowed = (h + k + l) % 2 == 0
                    elif reflection_condition == 'Rhombohedrally centred, obverse':
                        allowed = (-h + k + l) % 3 == 0
                    elif reflection_condition == 'Rhombohedrally centred, reverse':
                        allowed = (h - k + l) % 3 == 0
                    elif reflection_condition == 'Hexagonally centred, reverse':
                        allowed = (h - k) % 3 == 0
                    if np.isclose(np.linalg.norm(np.mod(HKL,1)), 0) and allowed:
                        HKL = HKL.astype(int).tolist()
                        pk.setIntMNP(V3D(*mnp))
                        pk.setIntHKL(V3D(*HKL))

        peak_dictionary.add_peaks(opk.format(r))

        if mtd.doesExist(opk.format(r)):
            DeleteWorkspace(opk.format(r))
            DeleteWorkspace(opk.format(r)+'_lean')

    peak_dictionary.split_peaks(split_angle)
    peak_dict = peak_dictionary.to_be_integrated()

    # ClearCache(AlgorithmCache=True, InstrumentCache=True, UsageServiceCache=True)

    keys = list(peak_dict.keys())
    keys = [key for key in keys if peak_dictionary.get_d(*key) > min_d]

    split_keys = [split.tolist() for split in np.array_split(keys, n_proc)]

    filename = os.path.join(directory, tmp)

    int_list, peak_tree = None, None

    args = [ref_dict, peak_tree, int_list, filename,
            spectrum_file, counts_file, tube_calibration, detector_calibration,
            outdir, directory, facility, instrument, ipts, runs,
            split_angle, a, b, c, alpha, beta, gamma, reflection_condition,
            mod_vector_1, mod_vector_2, mod_vector_3, max_order, cross_terms,
            chemical_formula, z_parameter, sample_mass, experiment, tmp]

    join_args = [(split, outname+'_p{}'.format(i), *args) for i, split in enumerate(split_keys)]

    # merge.integration_loop(*join_args[0])

    with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:
        pool.starmap(merge.integration_loop, join_args)
        pool.close()
        pool.join()

    merger = PdfFileMerger()

    for i in range(n_proc):
        partfile = os.path.join(outdir, outname+'_p{}'.format(i)+'.pdf')
        if os.path.exists(partfile):
            merger.append(partfile)

    merger.write(os.path.join(directory, outname+'.pdf'))       
    merger.close()

    if os.path.exists(os.path.join(directory, outname+'.pdf')):
        for i in range(n_proc):
            partfile = os.path.join(outdir, outname+'_p{}'.format(i)+'.pdf')
            if os.path.exists(partfile):
                os.remove(partfile)

    merger = PdfFileMerger()

    for i in range(n_proc):
        partfile = os.path.join(outdir, 'rej_'+outname+'_p{}'.format(i)+'.pdf')
        if os.path.exists(partfile):
            merger.append(partfile)
                
    merger.write(os.path.join(outdir, 'rejected.pdf'))       
    merger.close()

    if os.path.exists(os.path.join(outdir, 'rejected.pdf')):
        for i in range(n_proc):
            partfile = os.path.join(outdir, 'rej_'+outname+'_p{}'.format(i)+'.pdf')
            if os.path.exists(partfile):
                os.remove(partfile)

    for i in range(n_proc):
        tmp_peak_dict = peak_dictionary.load_dictionary(os.path.join(outdir, outname+'_p{}.pkl'.format(i)))

        if i == 0:
            peak_dict = copy.deepcopy(tmp_peak_dict)

        for key in list(tmp_peak_dict.keys()):
            peaks, tmp_peaks = peak_dict[key], tmp_peak_dict[key]

            new_peaks = []
            for peak, tmp_peak in zip(peaks, tmp_peaks):
                if tmp_peak.get_merged_intensity() > 0:
                    new_peaks.append(tmp_peak)
                else:
                    new_peaks.append(peak)
            peak_dict[key] = new_peaks

    peak_dictionary.peak_dict = peak_dict

    peak_dictionary.clear_peaks()
    peak_dictionary.repopulate_workspaces()
    scale = peak_dictionary.save_hkl(os.path.join(directory, outname+'.int'), adaptive_scale=adaptive_scale, scale=scale_factor)
    peak_dictionary.save(os.path.join(directory, outname+'.pkl'))

    scale_file = open(os.path.join(outdir, 'scale.txt'), 'w')
    scale_file.write('{:10.4e}'.format(scale))
    scale_file.close()

    # ---

    keys = peak_dictionary.peak_dict.keys()

    int_list = []
    Q_points = []

    for key in list(keys):

        peaks = peak_dictionary.peak_dict.get(key)

        Q_point = []

        for peak in peaks:

            if peak.get_merged_intensity() > 0:
                Q0 = peak.get_Q()
                Q_point.append(Q0)

        if len(Q_point) > 0:

            Q_points.append(np.mean(Q_point, axis=0))
            int_list.append(key)

    weak = False

    if weak:

        Q_points = np.stack(Q_points)

        peak_tree = KDTree(Q_points)

        args = [peak_dict, peak_tree, int_list, filename,
                spectrum_file, counts_file, tube_calibration, detector_calibration,
                outdir, directory, facility, instrument, ipts, runs,
                split_angle, a, b, c, alpha, beta, gamma, reflection_condition,
                mod_vector_1, mod_vector_2, mod_vector_3, max_order, cross_terms,
                chemical_formula, z_parameter, sample_mass, experiment, tmp]

        join_args = [(split, outname+'_weak_p{}'.format(i), *args) for i, split in enumerate(split_keys)]

        with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:
            pool.starmap(merge.integration_loop, join_args)
            pool.close()
            pool.join()

        merger = PdfFileMerger()

        for i in range(n_proc):
            partfile = os.path.join(outdir, outname+'_weak_p{}'.format(i)+'.pdf')
            if os.path.exists(partfile):
                merger.append(partfile)

        merger.write(os.path.join(outdir, outname+'_weak.pdf'))       
        merger.close()

        if os.path.exists(os.path.join(outdir, outname+'_weak.pdf')):
            for i in range(n_proc):
                partfile = os.path.join(outdir, outname+'_weak_p{}'.format(i)+'.pdf')
                if os.path.exists(partfile):
                    os.remove(partfile)

        merger = PdfFileMerger()

        for i in range(n_proc):
            partfile = os.path.join(directory, 'rej_'+outname+'_weak_p{}'.format(i)+'.pdf')
            merger.append(partfile)

        merger.write(os.path.join(outdir, 'rejected_weak.pdf'))       
        merger.close()

        if os.path.exists(os.path.join(outdir, 'rejected_weak.pdf')):
            for i in range(n_proc):
                partfile = os.path.join(outdir, 'rej_'+outname+'_weak_p{}'.format(i)+'.pdf')
                if os.path.exists(partfile):
                    os.remove(partfile)

        merge_peak_dict = {}

        for i in range(n_proc):
            tmp_peak_dict = peak_dictionary.load_dictionary(os.path.join(outdir, outname+'_weak_p{}.pkl'.format(i)))

            for key in list(tmp_peak_dict.keys()):
                peaks, tmp_peaks = peak_dict[key], tmp_peak_dict[key]

                new_peaks = []
                for peak, tmp_peak in zip(peaks, tmp_peaks):
                    if tmp_peak.get_merged_intensity() > 0:
                        new_peaks.append(tmp_peak)
                    elif peak.get_merged_intensity() > 0:
                        new_peaks.append(peak)
                if len(new_peaks) > 0:
                    merge_peak_dict[key] = new_peaks

        peak_dictionary.peak_dict = merge_peak_dict

        peak_dictionary.clear_peaks()
        peak_dictionary.repopulate_workspaces()
        peak_dictionary.save_hkl(os.path.join(directory, outname+'.int'), adaptive_scale=False, scale=scale)
        peak_dictionary.save(os.path.join(directory, outname+'.pkl'))

    # ---

    peak_dictionary.save_reflections(os.path.join(directory, outname+'.hkl'), adaptive_scale=True)

    LoadIsawUB(InputWorkspace='cws', Filename=os.path.join(directory, tmp+'.mat'))

    peak_dictionary.save_calibration(os.path.join(directory, outname+'_cal.nxs'))
    peak_dictionary.recalculate_hkl()
    peak_dictionary.save_hkl(os.path.join(directory, outname+'.int'), adaptive_scale=False, scale=scale)

    if sg is not None:
        peak_statistics = PeakStatistics(os.path.join(directory, outname+'.int'), sg)
        peak_statistics.prune_outliers()
        peak_statistics.write_statisics()
        peak_statistics.write_intensity()

    absorption_file = os.path.join(outdir, 'absorption.txt')

    if chemical_formula is not None and z_parameter > 0 and sample_mass > 0:
        peak_dictionary.apply_spherical_correction(vanadium_mass, fname=absorption_file)
        peak_dictionary.recalculate_hkl()
        peak_dictionary.save_hkl(os.path.join(directory, outname+'_w_abs.int'), adaptive_scale=False, scale=scale)
        peak_dictionary.save_reflections(os.path.join(directory, outname+'_w_abs.hkl'), adaptive_scale=False, scale=scale)

        if sg is not None:
            peak_statistics = PeakStatistics(os.path.join(directory, outname+'_w_abs.int'), sg)
            peak_statistics.prune_outliers()
            peak_statistics.write_statisics()
            peak_statistics.write_intensity()

    peak_dictionary.save(os.path.join(directory, outname+'.pkl'))

    for i in range(n_proc):
        partfile = os.path.join(outdir, outname+'_p{}'.format(i)+'.hkl')
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(outdir, outname+'_p{}'.format(i)+'.pkl')
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(outdir, outname+'_weak_p{}'.format(i)+'.hkl')
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(outdir, outname+'_weak_p{}'.format(i)+'.pkl')
        if os.path.exists(partfile):
            os.remove(partfile)

    fmt_summary = 3*'{:8}'+'{:8}'+6*'{:8}'+'{:4}'+6*'{:8}'+'\n'
    fmt_stats = 3*'{:8}'+'{:8}'+9*'{:10}'+'\n'

    hdr_summary = ['#      h', '       k', '       l', '    d-sp', '  wl-min', '  wl-max', \
                   '  2t-min', '  2t-max', '  az-min', '  az-max', '   n', \
                   '  om-min', '  om-max', '  ch-min', '  ch-max', '  ph-min', '  ph-max']

    hdr_stats = ['#      h', '       k', '       l', '    d-sp', 
                 ' chi2-1d', ' pk/bkg-1d', ' I/sig-1d', 
                 ' chi2-2d', ' pk/bkg-2d', ' I/sig-2d',  
                 ' chi2-1d', ' pk/bkg-1d', ' I/sig-1d']

    peak_file = open(os.path.join(directory, outname+'_summary.txt'), 'w')
    excl_file = open(os.path.join(outdir, 'rejected_summary.txt'), 'w')

    peak_file.write(fmt_summary.format(*hdr_summary))
    excl_file.write(fmt_summary.format(*hdr_summary))

    for i in range(n_proc):
        partfile = os.path.join(outdir, outname+'_p{}'.format(i)+'_summary.txt')
        if os.path.exists(partfile):
            tmp_file = open(partfile, 'r')
            tmp_lines = tmp_file.readlines()
            for tmp_line in tmp_lines:
                peak_file.write(tmp_line)
            tmp_file.close()
            os.remove(partfile)

    for i in range(n_proc):
        partfile = os.path.join(outdir, 'rej_'+outname+'_p{}'.format(i)+'_summary.txt')
        if os.path.exists(partfile):
            tmp_file = open(partfile, 'r')
            tmp_lines = tmp_file.readlines()
            for tmp_line in tmp_lines:
                excl_file.write(tmp_line)
            tmp_file.close()
            os.remove(partfile)

    peak_file.close()
    excl_file.close()

    peak_file = open(os.path.join(directory, outname+'_stats.txt'), 'w')
    excl_file = open(os.path.join(outdir, 'rejected_stats.txt'), 'w')

    peak_file.write(fmt_stats.format(*hdr_stats))
    excl_file.write(fmt_stats.format(*hdr_stats))

    for i in range(n_proc):
        partfile = os.path.join(outdir, outname+'_p{}'.format(i)+'_stats.txt')
        if os.path.exists(partfile):
            tmp_file = open(partfile, 'r')
            tmp_lines = tmp_file.readlines()
            for tmp_line in tmp_lines:
                peak_file.write(tmp_line)
            tmp_file.close()
            os.remove(partfile)

    for i in range(n_proc):
        partfile = os.path.join(outdir, 'rej_'+outname+'_p{}'.format(i)+'_stats.txt')
        if os.path.exists(partfile):
            tmp_file = open(partfile, 'r')
            tmp_lines = tmp_file.readlines()
            for tmp_line in tmp_lines:
                excl_file.write(tmp_line)
            tmp_file.close()
            os.remove(partfile)

    peak_file.close()
    excl_file.close()

    # ---

    if weak:

        peak_file = open(os.path.join(directory, outname+'_weak_summary.txt'), 'w')
        excl_file = open(os.path.join(outdir, 'rejected_weak_summary.txt'), 'w')

        peak_file.write(fmt_summary.format(*hdr_summary))
        excl_file.write(fmt_summary.format(*hdr_summary))

        for i in range(n_proc):
            partfile = os.path.join(outdir, outname+'_weak_p{}'.format(i)+'_summary.txt')
            if os.path.exists(partfile):
                tmp_file = open(partfile, 'r')
                tmp_lines = tmp_file.readlines()
                for tmp_line in tmp_lines:
                    peak_file.write(tmp_line)
                tmp_file.close()
                os.remove(partfile)

        for i in range(n_proc):
            partfile = os.path.join(outdir, 'rej_'+outname+'_weak_p{}'.format(i)+'_summary.txt')
            if os.path.exists(partfile):
                tmp_file = open(partfile, 'r')
                tmp_lines = tmp_file.readlines()
                for tmp_line in tmp_lines:
                    excl_file.write(tmp_line)
                tmp_file.close()
                os.remove(partfile)

        peak_file.close()
        excl_file.close()

        peak_file = open(os.path.join(directory, outname+'_weak_stats.txt'), 'w')
        excl_file = open(os.path.join(outdir, 'rejected_weak_stats.txt'), 'w')

        peak_file.write(fmt_stats.format(*hdr_stats))
        excl_file.write(fmt_stats.format(*hdr_stats))

        for i in range(n_proc):
            partfile = os.path.join(outdir, outname+'_weak_p{}'.format(i)+'_stats.txt')
            if os.path.exists(partfile):
                tmp_file = open(partfile, 'r')
                tmp_lines = tmp_file.readlines()
                for tmp_line in tmp_lines:
                    peak_file.write(tmp_line)
                tmp_file.close()
                os.remove(partfile)

        for i in range(n_proc):
            partfile = os.path.join(outdir, 'rej_'+outname+'_weak_p{}'.format(i)+'_stats.txt')
            if os.path.exists(partfile):
                tmp_file = open(partfile, 'r')
                tmp_lines = tmp_file.readlines()
                for tmp_line in tmp_lines:
                    excl_file.write(tmp_line)
                tmp_file.close()
                os.remove(partfile)

        peak_file.close()
        excl_file.close()