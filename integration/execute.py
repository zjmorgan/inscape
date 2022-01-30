# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

def load_data(runs, facility, instrument, ipts, ub_file,
              tube_calibration, detector_calibration, reflection_condition,
              mod_vector_1, mod_vector_2, mod_vector_3,
              max_order, cross_terms, radius,
              chemical_formula, volume, z):
    
    # peak prediction parameters ---------------------------------------------------
    if instrument == 'CORELLI':
        min_wavelength = 0.63
        max_wavelength= 2.51
    elif instrument == 'MANDI':
        min_wavelength = 0.4
        max_wavelength = 4

    min_d_spacing = 0.7
    max_d_spacing= 20

    # peak centroid radius ---------------------------------------------------------
    centroid_radius = 0.125

    # goniometer axis --------------------------------------------------------------
    gon_axis = 'BL9:Mot:Sample:Axis3.RBV'
    
    merge_md = []
    merge_pk = []
    
    for i, r in enumerate(runs):
        
        print('Processing run : {}'.format(r))
        ows = '{}_{}'.format(instrument,r)
        omd = ows+'_md'
        opk = ows+'_pks'

        if not mtd.doesExist(omd):
            filename = '/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5'.format(instrument,ipts,instrument,r)
            LoadEventNexus(Filename=filename, OutputWorkspace=ows)

            if tube_calibration is not None:
                ApplyCalibration(Workspace=ows, CalibrationTable=mtd['tube_table'])

            if detector_calibration is not None:
                LoadParameterFile(Workspace=ows, Filename=detector_calibration)
                
            if mtd.doesExist('sa'):
                MaskDetectors(Workspace=ows, MaskedWorkspace=mtd['sa'])

            proton_charge = sum(mtd[ows].getRun().getLogData('proton_charge').value)/1e12
            print('The current proton charge : {}'.format(proton_charge))

            # NormaliseByCurrent(ows, OutputWorkspace=ows)

            if instrument == 'CORELLI':
                SetGoniometer(Workspace=ows, Axis0='{},0,1,0,1'.format(gon_axis))
            else:
                SetGoniometer(Workspace=ows, Goniometers='Universal')

            if type(ub_file) is list:
                LoadIsawUB(InputWorkspace=ows, Filename=ub_file[i])
            else:
                LoadIsawUB(InputWorkspace=ows, Filename=ub_file)

            if chemical_formula is not None:

                SetSampleMaterial(InputWorkspace=ows,
                                  ChemicalFormula=chemical_formula,
                                  ZParameter=z,
                                  UnitCellVolume=volume)

                AnvredCorrection(InputWorkspace=ows,
                                 OnlySphericalAbsorption=True,
                                 Radius=radius,
                                 OutputWorkspace=ows)

            ConvertUnits(InputWorkspace=ows, OutputWorkspace=ows, EMode='Elastic', Target='Momentum')

            if instrument == 'CORELLI':
                CropWorkspaceForMDNorm(InputWorkspace=ows, XMin=2.5, XMax=10, OutputWorkspace=ows)

            #ConvertUnits(InputWorkspace=ows, OutputWorkspace=ows, EMode='Elastic', Target='dSpacing')

            md = ConvertToMD(InputWorkspace=ows,
                        OutputWorkspace=omd,
                        QDimensions='Q3D',
                        dEAnalysisMode='Elastic',
                        Q3DFrames='Q_sample',
                        LorentzCorrection=False,
                        MinValues='-20,-20,-20',
                        MaxValues='20,20,20',
                        Uproj='1,0,0',
                        Vproj='0,1,0',
                        Wproj='0,0,1',
                        SplitInto=2,
                        SplitThreshold=50,
                        MaxRecursionDepth=13,
                        MinRecursionDepth=7)

        if not mtd.doesExist(opk):
            PredictPeaks(InputWorkspace=omd,
                         WavelengthMin=min_wavelength,
                         WavelengthMax=max_wavelength,
                         MinDSpacing=min_d_spacing,
                         MaxDSpacing=max_d_spacing,
                         OutputType='Peak',
                         ReflectionCondition=reflection_condition,
                         OutputWorkspace=opk)

            if max_order > 0:
                PredictSatellitePeaks(Peaks=opk,
                                      SatellitePeaks=opk,
                                      ModVector1=mod_vector_1,
                                      ModVector2=mod_vector_2,
                                      ModVector3=mod_vector_3,
                                      MaxOrder=max_order,
                                      CrossTerms=cross_terms,
                                      IncludeIntegerHKL=True,
                                      IncludeAllPeaksInRange=False)

            CentroidPeaksMD(InputWorkspace=omd,
                            PeakRadius=centroid_radius,
                            PeaksWorkspace=opk,
                            OutputWorkspace=opk)

            CentroidPeaksMD(InputWorkspace=omd,
                            PeakRadius=centroid_radius,
                            PeaksWorkspace=opk,
                            OutputWorkspace=opk)

            pk = IntegratePeaksMD(InputWorkspace=omd,
                             PeakRadius=centroid_radius,
                             BackgroundInnerRadius=centroid_radius+0.01,
                             BackgroundOuterRadius=centroid_radius+0.02,
                             PeaksWorkspace=opk,
                             OutputWorkspace=opk)

        if mtd.doesExist(ows):
            DeleteWorkspace(ows)
        
        merge_md.append(omd)
        merge_pk.append(opk)
    
    return md, pk