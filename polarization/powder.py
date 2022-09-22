from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

import os
import sys

filename = sys.argv[1]

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..', 'reduction'))
sys.path.append(directory)

import imp
import parameters

imp.reload(parameters)

dictionary = parameters.load_input_file(filename)

up_run_nos = dictionary['up-scans'] if type(dictionary['up-scans']) is list else [dictionary['up-scans']]
down_run_nos = dictionary['down-scans'] if type(dictionary['down-scans']) is list else [dictionary['down-scans']]

if np.any([type(run) is list for run in up_run_nos]):  
    up_run_nos = [run for run_no in up_run_nos for run in run_no]
if np.any([type(run) is list for run in down_run_nos]):  
    down_run_nos = [run for run_no in down_run_nos for run in run_no]

facility, instrument = 'HFIR', 'HB3A'
ipts = dictionary['ipts']
exp = dictionary['experiment']

working_directory = '/{}/{}/IPTS-{}/shared/'.format(facility,instrument,ipts)
shared_directory = '/{}/{}/shared/'.format(facility,instrument)

directory = os.path.dirname(os.path.abspath(filename))
outname = dictionary['name']

# outdir = os.path.join(directory, outname)
# if not os.path.exists(outdir):
#     os.mkdir(outdir)

scale = 1
banks = 1
mask_edge_pixels = dictionary['mask-edge-pixels']

bin_2d = dictionary['2d-binning']
bin_1d = dictionary['1d-binning']

dirname = '/HFIR/HB3A/IPTS-{}/shared/autoreduce/'
fname = 'HB3A_exp{:04}_scan{:04}.nxs'
filename = os.path.join(dirname,fname)

counts_file = os.path.join(shared_directory+'Vanadium', dictionary['vanadium-file'])

# ---

data_1d, data_2d = [], []
err_sq_1d, err_sq_2d = [], []

for s, scans in enumerate([up_run_nos,down_run_nos]):

    data_files = [filename.format(ipts,exp,s) for s in scans]

    data_ws = []

    for data_file in data_files:
        ws, _ = os.path.splitext(os.path.basename(data_file))
        data_ws.append(ws)
        if not mtd.doesExist(ws):
            Load(Filename=data_file, OutputWorkspace=ws)

    if not mtd.doesExist('van'):
        Load(Filename=counts_file, OutputWorkspace='van')

    if not mtd.doesExist('DEMAND'):
        LoadEmptyInstrument(InstrumentName='HB3A', OutputWorkspace='DEMAND')

    van_data = mtd['van'].getSignalArray().copy().reshape(3,512,512,-1)
    van_data /= mtd['van'].getExperimentInfo(0).run().getProperty('monitor').value

    data, norm = [], []

    nu, gamma = [], []
    sc_two_theta = []

    for ws in data_ws:

        ws_data = mtd[ws].getSignalArray().reshape(3,512,512,-1)

        det_trans = mtd[ws].getExperimentInfo(0).run().getProperty('det_trans').value
        _2theta = mtd[ws].getExperimentInfo(0).run().getProperty('2theta').value
        monitor = mtd[ws].getExperimentInfo(0).run().getProperty('monitor').value

        ic = mtd[ws].getExperimentInfo(0).run().getProperty('ic').value

        for i, (dt, tt, mn) in enumerate(zip(det_trans, _2theta, monitor)):

            ws_name = 'det_{}_{}'.format(ws,i)

            if not mtd.doesExist(ws_name):

                AddSampleLog(Workspace='DEMAND', LogName='det_trans', LogText=str(dt), LogType='Number Series', NumberType='Double')
                AddSampleLog(Workspace='DEMAND', LogName='2theta', LogText=str(tt), LogType='Number Series', NumberType='Double')
                LoadInstrument(Workspace='DEMAND', RewriteSpectraMap=False, InstrumentName='HB3A')   

                PreprocessDetectorsToMD(InputWorkspace='DEMAND', OutputWorkspace=ws_name)

            L2 = np.array(mtd[ws_name].column(1)).reshape(3,512,512)[0,mask_edge_pixels:512-mask_edge_pixels,mask_edge_pixels:512-mask_edge_pixels]
            two_theta = np.array(mtd[ws_name].column(2)).reshape(3,512,512)[0,mask_edge_pixels:512-mask_edge_pixels,mask_edge_pixels:512-mask_edge_pixels]
            az_phi = np.array(mtd[ws_name].column(3)).reshape(3,512,512)[0,mask_edge_pixels:512-mask_edge_pixels,mask_edge_pixels:512-mask_edge_pixels]

            r = L2

            x = r*np.sin(two_theta)*np.cos(az_phi)
            y = r*np.sin(two_theta)*np.sin(az_phi)
            z = r*np.cos(two_theta)

            n = np.rad2deg(np.arcsin(y/L2)).flatten()
            g = np.rad2deg(np.arctan(x/z)).flatten()

            tt = np.rad2deg(two_theta)

            nu.append(n)
            gamma.append(g)

            sc_two_theta.append(tt)

            data.append(ws_data[0,mask_edge_pixels:512-mask_edge_pixels,mask_edge_pixels:512-mask_edge_pixels,i].flatten())

            norm.append(van_data[0,mask_edge_pixels:512-mask_edge_pixels,mask_edge_pixels:512-mask_edge_pixels,:].sum(axis=2).flatten()*mn)

    data, norm = np.array(data).flatten(), np.array(norm).flatten()

    nu, gamma = np.array(nu).flatten(), np.array(gamma).flatten()
    sc_two_theta = np.array(sc_two_theta).flatten()

    data_bin_counts2d, gamma_bin_edges, nu_bin_edges = np.histogram2d(gamma, nu, bins=bin_2d, weights=data)
    norm_bin_counts2d, _,            _,              = np.histogram2d(gamma, nu, bins=bin_2d, weights=norm)

    gamma_bin_centers_grid, nu_bin_centers_grid = np.meshgrid(0.5*(gamma_bin_edges[1:]+gamma_bin_edges[:-1]),
                                                              0.5*(nu_bin_edges[1:]+nu_bin_edges[:-1]))

    data_norm2d = scale*data_bin_counts2d/norm_bin_counts2d
    err_sq_norm2d = scale**2*data_bin_counts2d/norm_bin_counts2d**2

    # ---

    data_bin_counts, two_theta_bin_edges = np.histogram(sc_two_theta, bins=mask_edge_pixels, weights=data)
    norm_bin_counts, _,                  = np.histogram(sc_two_theta, bins=mask_edge_pixels, weights=norm)

    two_theta_bin_centers_grid = 0.5*(two_theta_bin_edges[1:]+two_theta_bin_edges[:-1])

    data_norm = scale*data_bin_counts/norm_bin_counts
    err_sq_norm = scale**2*data_bin_counts/norm_bin_counts**2

    data_1d.append(data_norm)
    data_2d.append(data_norm2d)

    err_sq_1d.append(err_sq_norm)
    err_sq_2d.append(err_sq_norm2d)

ops = ['Up', 'Down', 'Sum', 'Difference']

data_1d_ops = [data_1d[0],data_1d[1],data_1d[1]+data_1d[0],data_1d[0]-data_1d[1]]
data_2d_ops = [data_2d[0],data_2d[1],data_2d[1]+data_2d[0],data_2d[0]-data_2d[1]]

err_sq_1d_ops = [err_sq_1d[0],err_sq_1d[1],err_sq_1d[1]+err_sq_1d[0],err_sq_1d[0]+err_sq_1d[1]]
err_sq_2d_ops = [err_sq_2d[0],err_sq_2d[1],err_sq_2d[1]+err_sq_2d[0],err_sq_2d[0]+err_sq_2d[1]]

with PdfPages(directory+'/{}.pdf'.format(outname)) as pdf:

    for op, data_1d_op, data_2d_op, err_sq_1d_op, err_sq_2d_op in zip(ops,data_1d_ops,data_2d_ops,err_sq_1d_ops,err_sq_2d_ops):

        fig, ax = plt.subplots(1, 1, num='2d_{}'.format(op))
        im = ax.pcolormesh(gamma_bin_centers_grid, nu_bin_centers_grid, data_2d_op.T)
        im.set_edgecolor('face')
        ax.set_xlabel(r'$\gamma$ [deg.]') 
        ax.set_ylabel(r'$\nu$ [deg.]') 
        cb = fig.colorbar(im)
        cb.ax.set_ylabel(r'Intensity [arb. unit]') 
        cb.ax.minorticks_on()
        ax.set_title(r'{}'.format(op)) 
        ax.minorticks_on()
        pdf.savefig()
        plt.close()

        fig, ax = plt.subplots(1, 1, num='1d_{}'.format(op))
        ax.errorbar(two_theta_bin_centers_grid, data_1d_op, yerr=np.sqrt(err_sq_1d_op), fmt='-o')
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.set_xlabel(r'$2\theta$ [deg.]') 
        ax.set_ylabel(r'Intensity [arb. unit]') 
        ax.set_title(r'{}'.format(op)) 
        ax.minorticks_on()
        pdf.savefig()
        plt.close()

X1d_up = np.stack((two_theta_bin_centers_grid, data_1d[0], np.sqrt(err_sq_1d[0]))).T
X1d_down = np.stack((two_theta_bin_centers_grid, data_1d[1], np.sqrt(err_sq_1d[1]))).T

np.savetxt(fname=os.path.join(directory,outname+'_up_1d.dat'), X=X1d_up, fmt='%.4e', delimiter=',', header='2theta,I,sig')
np.savetxt(fname=os.path.join(directory,outname+'_down_1d.dat'), X=X1d_down, fmt='%.4e', delimiter=',', header='2theta,I,sig')

X2d_up = np.stack((gamma_bin_centers_grid.flatten(), nu_bin_centers_grid.flatten(), data_2d[0].flatten(), np.sqrt(err_sq_2d[0]).flatten())).T
X2d_down = np.stack((gamma_bin_centers_grid.flatten(), nu_bin_centers_grid.flatten(), data_2d[1].flatten(), np.sqrt(err_sq_2d[1]).flatten())).T

np.savetxt(fname=os.path.join(directory,outname+'_up_2d.dat'), X=X2d_up, fmt='%.4e', delimiter=',', header='gamma,nu,I,sig')
np.savetxt(fname=os.path.join(directory,outname+'_down_2d.dat'), X=X2d_down, fmt='%.4e', delimiter=',', header='gamma,nu,I,sig')