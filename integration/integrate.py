import logging
#logging.disable(logging.CRITICAL)

import warnings
#warnings.filterwarnings('ignore')

import sys, os, re, imp, copy, shutil

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import multiprocess as multiprocessing

from mantid import config
#config.setLogLevel(0, quiet=True)

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

import matplotlib.pyplot as plt

from scipy.spatial import KDTree

filename, n_proc = sys.argv[1], int(sys.argv[2])

if n_proc > os.cpu_count():
    n_proc = os.cpu_count()

scale_constant = 1e+4

if __name__ == '__main__':

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    #ClearCache(AlgorithmCache=True, InstrumentCache=True, UsageServiceCache=True, DownloadedInstrumentFileCache=True, GeometryFileCache=True)

    CreateSampleWorkspace(OutputWorkspace='sample')

    dictionary = parameters.load_input_file(filename)

    a = dictionary['a']
    b = dictionary['b']
    c = dictionary['c']
    alpha = dictionary['alpha']
    beta = dictionary['beta']
    gamma = dictionary['gamma']

    min_d = dictionary.get('minimum-d-spacing')

    adaptive_scale = dictionary.get('adaptive-scale')
    scale_factor = dictionary.get('scale-factor')

    if scale_factor is None:
        scale_factor = 1

    reflection_condition = dictionary.get('reflection-condition')
    centering = dictionary.get('centering')

    group = dictionary.get('group')

    if np.any([key in ['P', 'Primitive'] for key in [centering, reflection_condition]]):
        reflection_condition = 'Primitive'    
        centering = 'P'
    elif np.any([key in ['F', 'All-face centred'] for key in [centering, reflection_condition]]):
        reflection_condition = 'All-face centred'
        centering = 'F'
    elif np.any([key in ['I', 'Body centred'] for key in [centering, reflection_condition]]):
        reflection_condition = 'Body centred'
        centering = 'I'
    elif np.any([key in ['A', 'A-face centred'] for key in [centering, reflection_condition]]):
        reflection_condition = 'A-face centred'
        centering = 'A'
    elif np.any([key in ['B', 'B-face centred'] for key in [centering, reflection_condition]]):
        reflection_condition = 'B-face centred'
        centering = 'B'
    elif np.any([key in ['C', 'C-face centred'] for key in [centering, reflection_condition]]):
        reflection_condition = 'C-face centred'
        centering = 'C'
    elif np.any([key in ['R', 'Robv', 'Rhombohedrally centred, obverse'] for key in [centering, reflection_condition]]):
        reflection_condition = 'Rhombohedrally centred, obverse'
        centering = 'R'
    elif np.any([key in ['Rrev', 'Rhombohedrally centred, reverse'] for key in [centering, reflection_condition]]):
        reflection_condition = 'Rhombohedrally centred, reverse'
        centering = 'R'
    elif np.any([key in ['H', 'Hexagonally centred, reverse'] for key in [centering, reflection_condition]]):
        reflection_condition = 'Hexagonally centred, reverse'
        centering = 'H'

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

    if dictionary.get('ub-file') is not None:
        ub_file = os.path.join(working_directory, dictionary['ub-file'])
        if '*' in ub_file:
            ub_file = [ub_file.replace('*', str(run)) for run in run_nos]
    else:
        ub_file = None

    split_angle = dictionary['split-angle']

    directory = os.path.dirname(os.path.abspath(filename))
    outname = dictionary['name']

    elastic = dictionary.get('elastic')
    timing_offset = dictionary.get('time-offset')

    if elastic:
        outname += '_cc'

    outdir = os.path.join(directory, outname)
    dbgdir = os.path.join(outdir, 'debug')
    if not os.path.exists(outdir):
        #shutil.rmtree(outdir)
        os.mkdir(outdir)
    if not os.path.exists(dbgdir):
        os.mkdir(dbgdir)

    items = os.listdir(dbgdir)
    for item in items:
        if item.endswith('.png') or item.endswith('.jpg') or item.endswith('.pdf') or item.endswith('.txt'):
            #if '_p' in item:
            #    if item.split('_p')[-1][0].isdigit():
            os.remove(os.path.join(dbgdir, item))

    parameters.output_input_file(filename, directory, outname+'_int')

    if dictionary.get('flux-file') is not None:
        spectrum_file = os.path.join(shared_directory+'Vanadium', dictionary['flux-file'])
    else:
        spectrum_file = None

    if dictionary.get('vanadium-file') is not None:
        counts_file = os.path.join(shared_directory+'Vanadium', dictionary['vanadium-file'])
    else:
        counts_file = None

    if dictionary.get('mask-file') is not None:
        mask_file = os.path.join(shared_directory+'Vanadium', dictionary['mask-file'])
    else:
        mask_file = None

    if dictionary.get('tube-file') is not None:
        tube_calibration = os.path.join(shared_directory+'calibration', dictionary['tube-file'])
    else:
        tube_calibration = None

    if dictionary.get('detector-file') is not None:
        detector_calibration = os.path.join(shared_directory+'calibration', dictionary['detector-file'])
    else:
        detector_calibration = None

    mod_vector_1 = dictionary['modulation-vector-1']
    mod_vector_2 = dictionary['modulation-vector-2']
    mod_vector_3 = dictionary['modulation-vector-3']
    max_order = dictionary['max-order']
    cross_terms = dictionary['cross-terms']

    min_d_sat = dictionary.get('minimum-modulation-d-spacing')
    if min_d_sat is None:
        min_d_sat = min_d

    if np.allclose(mod_vector_1, 0) and np.allclose(mod_vector_2, 0) and np.allclose(mod_vector_3, 0):
        max_order = 0

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
                                         tube_calibration, detector_calibration, mask_file)

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

    if os.path.exists(os.path.join(dbgdir, tmp+'_pk.nxs')) and not mtd.doesExist(tmp):

        LoadNexus(Filename=os.path.join(dbgdir, tmp+'_pk_lean.nxs'), OutputWorkspace=tmp+'_lean')
        LoadNexus(Filename=os.path.join(dbgdir, tmp+'_pk.nxs'), OutputWorkspace=tmp)
        LoadIsawUB(InputWorkspace=tmp+'_lean', Filename=os.path.join(dbgdir, tmp+'.mat'))
        LoadIsawUB(InputWorkspace=tmp, Filename=os.path.join(dbgdir, tmp+'.mat'))

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

    if os.path.exists(os.path.join(dbgdir, tmp+'_ellip.nxs')):

        LoadNexus(Filename=os.path.join(dbgdir, tmp+'_pk_ellip.nxs'), OutputWorkspace=tmp+'_ellip')
        LoadIsawUB(InputWorkspace=tmp+'_ellip', Filename=os.path.join(dbgdir, tmp+'.mat'))

    # if not mtd.doesExist(tmp):

    split_runs = [split.tolist() for split in np.array_split(runs, m_proc)]

    args = [outdir, dbgdir, directory, facility, instrument, ipts, runs, ub_file, reflection_condition, min_d,
            spectrum_file, counts_file, tube_calibration, detector_calibration, mask_file,
            mod_vector_1, mod_vector_2, mod_vector_3, max_order, cross_terms, experiment, tmp]

    join_args = [(split, i, outname+'_p{}'.format(i), *args) for i, split in enumerate(split_runs)]

    # merge.pre_integration(*join_args)

    config['MultiThreaded.MaxCores'] == 1
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

    print('Spawning threads for pre-integration')
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:
        pool.starmap(merge.pre_integration, join_args)
        pool.close()
        pool.join()
    print('Joining threads from pre-integration')

    config['MultiThreaded.MaxCores'] == 4
    os.environ.pop('OPENBLAS_NUM_THREADS', None)
    os.environ.pop('OMP_NUM_THREADS', None)

    # if not mtd.doesExist(tmp):   

    if mtd.doesExist('sa'):
        CreatePeaksWorkspace(InstrumentWorkspace='sa', NumberOfPeaks=0, OutputType='Peak', OutputWorkspace=tmp)
        CreatePeaksWorkspace(InstrumentWorkspace='sa', NumberOfPeaks=0, OutputType='Peak', OutputWorkspace=tmp+'_ellip')
    else:
        CreatePeaksWorkspace(InstrumentWorkspace='van', NumberOfPeaks=0, OutputType='Peak', OutputWorkspace=tmp)

    CreatePeaksWorkspace(NumberOfPeaks=0, OutputType='LeanElasticPeak', OutputWorkspace=tmp+'_lean')

    CreateEmptyTableWorkspace(OutputWorkspace='run_info')

    mtd['run_info'].addColumn('Int', 'RunNumber')
    mtd['run_info'].addColumn('Double', 'Scale')

    for i in range(m_proc):
        partname = outname+'_p{}'.format(i)

        LoadNexus(Filename=os.path.join(dbgdir, partname+'_log.nxs'), OutputWorkspace=partname+'_log')
        for j in range(mtd[partname+'_log'].rowCount()):
            items = mtd[partname+'_log'].row(j)
            mtd['run_info'].addRow(list(items.values()))
        DeleteWorkspace(partname+'_log')

        LoadNexus(Filename=os.path.join(dbgdir, partname+'_pk.nxs'), OutputWorkspace=partname+'_pk')
        if mtd[partname+'_pk'].getNumberPeaks() > 0:
            LoadIsawUB(InputWorkspace=partname+'_pk', Filename=os.path.join(dbgdir, partname+'.mat'))
            CombinePeaksWorkspaces(LHSWorkspace=partname+'_pk', RHSWorkspace=tmp, OutputWorkspace=tmp)
            LoadIsawUB(InputWorkspace=tmp, Filename=os.path.join(dbgdir, partname+'.mat'))
        DeleteWorkspace(partname+'_pk')

        LoadNexus(Filename=os.path.join(dbgdir, partname+'_pk_lean.nxs'), OutputWorkspace=partname+'_pk_lean')
        if mtd[partname+'_pk_lean'].getNumberPeaks() > 0:
            LoadIsawUB(InputWorkspace=partname+'_pk_lean', Filename=os.path.join(dbgdir, partname+'.mat'))
            CombinePeaksWorkspaces(LHSWorkspace=partname+'_pk_lean', RHSWorkspace=tmp+'_lean', OutputWorkspace=tmp+'_lean')
            LoadIsawUB(InputWorkspace=tmp+'_lean', Filename=os.path.join(dbgdir, partname+'.mat'))
        DeleteWorkspace(partname+'_pk_lean')

        if os.path.exists(os.path.join(dbgdir, partname+'_pk_ellip.nxs')):

            LoadNexus(Filename=os.path.join(dbgdir, partname+'_pk_ellip.nxs'), OutputWorkspace=partname+'_pk_ellip')
            if mtd[partname+'_pk_ellip'].getNumberPeaks() > 0:
                LoadIsawUB(InputWorkspace=partname+'_pk_ellip', Filename=os.path.join(dbgdir, partname+'.mat'))
                CombinePeaksWorkspaces(LHSWorkspace=partname+'_pk_ellip', RHSWorkspace=tmp+'_ellip', OutputWorkspace=tmp+'_ellip')
                LoadIsawUB(InputWorkspace=tmp+'_ellip', Filename=os.path.join(dbgdir, partname+'.mat'))
            DeleteWorkspace(partname+'_pk_ellip')

            os.remove(os.path.join(dbgdir, partname+'_pk_ellip.nxs'))

        os.remove(os.path.join(dbgdir, partname+'_log.nxs'))
        os.remove(os.path.join(dbgdir, partname+'_pk_lean.nxs'))
        os.remove(os.path.join(dbgdir, partname+'_pk.nxs'))
        os.remove(os.path.join(dbgdir, partname+'.mat'))
        
    SaveNexus(InputWorkspace='run_info', Filename=os.path.join(dbgdir, tmp+'_log.nxs'))
    SaveNexus(InputWorkspace=tmp+'_lean', Filename=os.path.join(dbgdir, tmp+'_pk_lean.nxs'))
    SaveNexus(InputWorkspace=tmp, Filename=os.path.join(dbgdir, tmp+'_pk.nxs'))
    SaveIsawUB(InputWorkspace=tmp, Filename=os.path.join(dbgdir, tmp+'.mat'))

    if mtd.doesExist(tmp+'_ellip'):
        SaveNexus(InputWorkspace=tmp+'_ellip', Filename=os.path.join(dbgdir, tmp+'_pk_ellip.nxs'))

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

    if ref_dict is not None:
        ref_peak_dictionary = PeakDictionary(a, b, c, alpha, beta, gamma)
        ref_peak_dictionary.load(os.path.join(working_directory, ref_dict))
        ref_dict = ref_peak_dictionary.peak_dict
    else:
        ref_peak_dictionary = None

    if mtd.doesExist(tmp+'_ellip'):

        Q, r = [], []

        for p in range(mtd[tmp+'_ellip'].getNumberPeaks()):
            pk = mtd[tmp+'_ellip'].getPeak(p)
            js = eval(pk.getPeakShape().toJSON())
            if pk.getIntensity() >= 1:
                Qmod = 2*np.pi/pk.getDSpacing()
                radius = np.cbrt([js['radius0']*js['radius1']*js['radius2']])
                if np.isfinite(Qmod) and np.isfinite(radius) and np.isreal(radius):
                    Q.append(Qmod)
                    r.append(radius)

        Q, r = np.array(Q), np.array(r)

        Q_max = np.nanmax(Q) if min_d is None else 2*np.pi/min_d

        Q_bin, Qh = np.linspace(np.nanmin(Q), Q_max, 20, retstep=True)
        r_bin = np.zeros_like(Q_bin)
        for i in range(Q_bin.size-1):
            mask = np.logical_and(Q > Q[i]-Qh/2, Q <= Q[i+1]+Qh/2)
            r_bin[i] = np.nanmean(r[mask])

        mask = np.logical_and(np.isfinite(r_bin), r_bin > 0)

        r_bin = r_bin[mask]
        Q_bin = Q_bin[mask]

        box_fit_size = np.linalg.lstsq(np.vstack([np.ones_like(Q_bin), Q_bin]).T, r_bin)[0]

        fig, ax = plt.subplots()
        ax.plot(Q_bin, r_bin, '.')
        ax.plot(Q_bin, box_fit_size[0]+box_fit_size[1]*Q_bin, '-')
        ax.set_xlabel('Q [\u212B\u207B\u00B9]')
        ax.set_ylabel('Radius [\u212B\u207B\u00B9]')
        fig.savefig(os.path.join(outdir, 'size.pdf'))

        with open(os.path.join(outdir, 'size.txt'), 'w') as f:
            f.write('nominal peak Q size        : {:12.4f}\n'.format(box_fit_size[0]))
            f.write('adaptive peak Q multiplier : {:12.4f}\n'.format(box_fit_size[1]))

    else:

        box_fit_size = 0.15, 0

    cif_file = dictionary.get('cif-file')

    peak_dictionary = PeakDictionary(a, b, c, alpha, beta, gamma)

    if cif_file is not None:
        peak_dictionary.load_cif(os.path.join(working_directory, cif_file))

    peak_dictionary.set_satellite_info(mod_vector_1, mod_vector_2, mod_vector_3, max_order)
    peak_dictionary.set_material_info(chemical_formula, z_parameter, sample_mass)
    peak_dictionary.set_scale_constant(scale_constant)

    cluster = dictionary.get('close-satellite-fitting')
    if cluster is None:
        cluster = False

    for r in runs:

        if min_d is not None:
            FilterPeaks(InputWorkspace=opk.format(r),
                        OutputWorkspace=opk.format(r),
                        FilterVariable='DSpacing',
                        FilterValue=min_d, 
                        Operator='>')
            FilterPeaks(InputWorkspace=opk.format(r)+'_lean',
                        OutputWorkspace=opk.format(r)+'_lean',
                        FilterVariable='DSpacing',
                        FilterValue=min_d,
                        Operator='>')

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
                    elif reflection_condition == 'All-face centred':
                        allowed = (h + l) % 2 == 0 and (k + l) % 2 == 0 and (h + k) % 2 == 0
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

        if mtd.doesExist('flux'):
            lamda_min = 2*np.pi/mtd['flux'].dataX(0).max()
            lamda_max = 2*np.pi/mtd['flux'].dataX(0).min()
        else:
            lamda_min = None
            lamda_max = None

        print('Adding run {}'.format(opk.format(r)), cluster, lamda_min, lamda_max)

        peak_dictionary.add_peaks(opk.format(r), cluster, lamda_min, lamda_max)

        if mtd.doesExist(opk.format(r)):
            DeleteWorkspace(opk.format(r))
            DeleteWorkspace(opk.format(r)+'_lean')

    if mtd.doesExist('sa'):
        DeleteWorkspace('sa')

    if mtd.doesExist('flux'):
        DeleteWorkspace('flux')

    if mtd.doesExist('van'):
        DeleteWorkspace('van')

    peak_dictionary.split_peaks(split_angle)
    peak_dict = peak_dictionary.to_be_integrated()

    keys = list(peak_dict.keys())

    if min_d is not None:
        keys = [key for key in keys if peak_dictionary.get_d(*key) > min_d]

    key_list, run_list, ind_list = [], [], []
    for key in keys:
        peaks = peak_dictionary.peak_dict.get(key)
        for j, peak in enumerate(peaks):
            run_list.append(peak.get_run_numbers()[0])
            key_list.append(key)
            ind_list.append(j)

    if np.isclose(split_angle, 0):
        sort = np.argsort(run_list)
    else:
        sort = np.arange(len(run_list))

    keys = [key_list[i] for i in sort]
    inds = [ind_list[i] for i in sort]

    split_keys = [split.tolist() for split in np.array_split(keys, n_proc)]
    split_inds = [split.tolist() for split in np.array_split(inds, n_proc)]

    filename = os.path.join(dbgdir, tmp)

    int_list, peak_tree = None, None

    args = [ref_dict, int_list, filename, box_fit_size,
            spectrum_file, counts_file, tube_calibration, detector_calibration, mask_file,
            outdir, dbgdir, directory, facility, instrument, ipts, runs,
            split_angle, min_d, min_d_sat, a, b, c, alpha, beta, gamma, reflection_condition,
            mod_vector_1, mod_vector_2, mod_vector_3, max_order, cross_terms,
            chemical_formula, z_parameter, sample_mass, elastic, timing_offset, experiment, tmp, cluster]

    join_args = [(split_key, split_ind, i, outname+'_p{}'.format(i), *args) for i, (split_key, split_ind) in enumerate(zip(split_keys,split_inds))]

    # merge.integration_loop(*join_args[0])

    config['MultiThreaded.MaxCores'] == 1
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

    print('Spawning threads for integration')
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.get_context('spawn').Pool(processes=n_proc) as pool:
        pool.starmap(merge.integration_loop, join_args)
        pool.close()
        pool.join()
    print('Joining threads from integration')

    config['MultiThreaded.MaxCores'] == 4
    os.environ.pop('OPENBLAS_NUM_THREADS', None)
    os.environ.pop('OMP_NUM_THREADS', None)

    merger = PdfFileMerger()

    for i in range(n_proc):
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'.pdf')
        if os.path.exists(partfile):
            merger.append(partfile)

    merger.write(os.path.join(outdir, outname+'.pdf'))       
    merger.close()

    if os.path.exists(os.path.join(outdir, outname+'.pdf')):
        for i in range(n_proc):
            partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'.pdf')
            if os.path.exists(partfile):
                os.remove(partfile)

    merger = PdfFileMerger()

    for i in range(n_proc):
        partfile = os.path.join(dbgdir, 'rej_'+outname+'_p{}'.format(i)+'.pdf')
        if os.path.exists(partfile):
            merger.append(partfile)

    merger.write(os.path.join(dbgdir, 'rejected.pdf'))       
    merger.close()

    if os.path.exists(os.path.join(dbgdir, 'rejected.pdf')):
        for i in range(n_proc):
            partfile = os.path.join(dbgdir, 'rej_'+outname+'_p{}'.format(i)+'.pdf')
            if os.path.exists(partfile):
                os.remove(partfile)

    for i in range(n_proc):
        tmp_peak_dict = peak_dictionary.load_dictionary(os.path.join(dbgdir, outname+'_p{}.pkl'.format(i)))

        if i == 0:
            peak_dict = copy.deepcopy(tmp_peak_dict)

        for key in list(tmp_peak_dict.keys()):
            peaks, tmp_peaks = peak_dict.get(key), tmp_peak_dict[key]

            new_peaks = []

            if peaks is not None:
                for peak, tmp_peak in zip(peaks, tmp_peaks):
                    if tmp_peak.get_merged_intensity() > 0:
                        new_peaks.append(tmp_peak)
                    else:
                        new_peaks.append(peak)
            else:
                for tmp_peak in tmp_peaks:
                    if tmp_peak.get_merged_intensity() > 0:
                        new_peaks.append(tmp_peak)

            peak_dict[key] = new_peaks

    peak_dictionary.peak_dict = peak_dict

    peak_dictionary.clear_peaks()
    peak_dictionary.repopulate_workspaces()
    scale = peak_dictionary.save_hkl(os.path.join(outdir, outname+'.int'), adaptive_scale=adaptive_scale, scale=scale_factor)
    peak_dictionary.save(os.path.join(outdir, outname+'.pkl'))

    scale_file = open(os.path.join(outdir, 'scale.txt'), 'w')
    scale_file.write('{:10.4e}'.format(scale))
    scale_file.close()

    # ---

    LoadIsawUB(InputWorkspace='cws', Filename=os.path.join(dbgdir, tmp+'.mat'))

    peak_dictionary.save_calibration(os.path.join(outdir, outname+'_cal.nxs'))
    peak_dictionary.recalculate_hkl(fname=os.path.join(outdir, 'indexing.txt'))
    peak_dictionary.save_hkl(os.path.join(outdir, outname+'.int'), adaptive_scale=False, scale=scale)
    peak_dictionary.save_reflections(os.path.join(outdir, outname+'.hkl'), adaptive_scale=True)

    if sg is not None:
        peak_statistics = PeakStatistics(os.path.join(outdir, outname+'.hkl'), sg)
        peak_statistics.prune_outliers()
        peak_statistics.write_statisics()
        peak_statistics.write_intensity()

    absorption_file = os.path.join(outdir, 'absorption.txt')

    if chemical_formula is not None and z_parameter > 0 and sample_mass > 0:
        peak_dictionary.apply_spherical_correction(vanadium_mass, fname=absorption_file)
        peak_dictionary.save_hkl(os.path.join(outdir, outname+'_w_abs.int'), adaptive_scale=False, scale=scale)
        peak_dictionary.save_reflections(os.path.join(outdir, outname+'_w_abs.hkl'), adaptive_scale=False, scale=scale)

        if sg is not None:
            peak_statistics = PeakStatistics(os.path.join(outdir, outname+'_w_abs.hkl'), sg)
            peak_statistics.prune_outliers()
            peak_statistics.write_statisics()
            peak_statistics.write_intensity()

    peak_dictionary.save(os.path.join(outdir, outname+'.pkl'))

    for i in range(n_proc):
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'.hkl')
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'.int')
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'_fit.hkl')
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'_fit.int')
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'_nuc.hkl')
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'_nuc.int')
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'_sat.hkl')
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'_sat.int')
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'_nuc_fit.hkl')
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'_nuc_fit.int')
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'_sat_fit.hkl')
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'_sat_fit.int')
        if os.path.exists(partfile):
            os.remove(partfile)
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'.pkl')
        if os.path.exists(partfile):
            os.remove(partfile)

    fmt_summary = 3*'{:8}'+'{:8}'+6*'{:8}'+'{:4}'+6*'{:8}'+'\n'
    fmt_stats = 3*'{:8}'+'{:8}'+13*'{:10}'+'\n'
    fmt_params = 3*'{:8}'+'{:8}'+2*'{:10}'+6*'{:8}'+3*'{:8}'+'{:6}'+2*'{:6}'+6*'{:8}'+'\n'

    hdr_summary = ['#      h', '       k', '       l', '    d-sp', '  wl-min', '  wl-max', \
                   '  2t-min', '  2t-max', '  az-min', '  az-max', '   n', \
                   '  om-min', '  om-max', '  ch-min', '  ch-max', '  ph-min', '  ph-max']

    hdr_stats = ['#      h', '       k', '       l', '    d-sp',
                 ' chi2-1d', ' pk/bkg-1d', ' I/sig-1d',
                 ' chi2-2d', ' pk/bkg-2d', ' I/sig-2d',
                 ' chi2-1d', ' pk/bkg-1d', ' I/sig-1d',
                 ' chi2-2d', ' pk/bkg-2d', ' I/sig-2d', '   reason']

    hdr_params = ['#      h', '       k', '       l', '    d-sp', '         A', '         B', 
                  '    mu_0', '    mu_1', '    mu_2', ' sigma_0', ' sigma_1', ' sigma_2',
                  '  rho_12', '  rho_02', '  rho_01', '   pts', ' bound', '  type',
                  '    mu_0', '    mu_1', '    mu_2', ' sigma_0', ' sigma_1', ' sigma_2']

    peak_file = open(os.path.join(outdir, outname+'_summary.txt'), 'w')
    excl_file = open(os.path.join(dbgdir, 'rejected_summary.txt'), 'w')

    peak_file.write(fmt_summary.format(*hdr_summary))
    excl_file.write(fmt_summary.format(*hdr_summary))

    for i in range(n_proc):
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'_summary.txt')
        if os.path.exists(partfile):
            tmp_file = open(partfile, 'r')
            tmp_lines = tmp_file.readlines()
            for tmp_line in tmp_lines:
                peak_file.write(tmp_line)
            tmp_file.close()
            os.remove(partfile)

    for i in range(n_proc):
        partfile = os.path.join(dbgdir, 'rej_'+outname+'_p{}'.format(i)+'_summary.txt')
        if os.path.exists(partfile):
            tmp_file = open(partfile, 'r')
            tmp_lines = tmp_file.readlines()
            for tmp_line in tmp_lines:
                excl_file.write(tmp_line)
            tmp_file.close()
            os.remove(partfile)

    peak_file.close()
    excl_file.close()

    peak_file = open(os.path.join(outdir, outname+'_stats.txt'), 'w')
    excl_file = open(os.path.join(dbgdir, 'rejected_stats.txt'), 'w')

    peak_file.write(fmt_stats.format(*hdr_stats))
    excl_file.write(fmt_stats.format(*hdr_stats))

    for i in range(n_proc):
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'_stats.txt')
        if os.path.exists(partfile):
            tmp_file = open(partfile, 'r')
            tmp_lines = tmp_file.readlines()
            for tmp_line in tmp_lines:
                peak_file.write(tmp_line)
            tmp_file.close()
            os.remove(partfile)

    for i in range(n_proc):
        partfile = os.path.join(dbgdir, 'rej_'+outname+'_p{}'.format(i)+'_stats.txt')
        if os.path.exists(partfile):
            tmp_file = open(partfile, 'r')
            tmp_lines = tmp_file.readlines()
            for tmp_line in tmp_lines:
                excl_file.write(tmp_line)
            tmp_file.close()
            os.remove(partfile)

    peak_file.close()
    excl_file.close()

    peak_file = open(os.path.join(outdir, outname+'_params.txt'), 'w')
    excl_file = open(os.path.join(dbgdir, 'rejected_params.txt'), 'w')

    peak_file.write(fmt_params.format(*hdr_params))
    excl_file.write(fmt_params.format(*hdr_params))

    for i in range(n_proc):
        partfile = os.path.join(dbgdir, outname+'_p{}'.format(i)+'_params.txt')
        if os.path.exists(partfile):
            tmp_file = open(partfile, 'r')
            tmp_lines = tmp_file.readlines()
            for tmp_line in tmp_lines:
                peak_file.write(tmp_line)
            tmp_file.close()
            os.remove(partfile)

    for i in range(n_proc):
        partfile = os.path.join(dbgdir, 'rej_'+outname+'_p{}'.format(i)+'_params.txt')
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