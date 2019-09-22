import matplotlib
matplotlib.use('Agg')

import os, re
import pickle
import numpy as np
from pymc import Matplot
from pymc.database.hdf5 import load as pymc_load

import matplotlib.pyplot as plt

import plots

path = './run/spat_nmixture_nb/ungulates'
regexp = r'.*'
dirs = [k for k in os.listdir(path) if re.match(regexp, k)]

for subdir in dirs:

    try:
        output_dir = os.path.join(path, subdir)
        data_pickled = pickle.load(open(os.path.join(output_dir, 'data.p'), 'rb'))
        ls_dmat = data_pickled['ls_dmat']
        dmat = data_pickled['dmat']
        pred_dmat = data_pickled['pred_dmat']
        coords_grid = data_pickled['coords_grid']
        counts = data_pickled['counts']
        M = pymc_load(os.path.join(output_dir, subdir +'.hdf5'))

    except Exception, e:
        print e
        continue

    # Turn interactive plotting off
    plt.ioff()

    Matplot.autocorrelation(M.ls_beta, path=output_dir)
    Matplot.autocorrelation(M.beta, path=output_dir)

    plots.plot_traces(
        M, 'ls_beta', maxcol=3, maxrow=4, params_n=ls_dmat.shape[1],
        save_fig=True, path=output_dir
    )
    
    plots.plot_traces(
        M, 'beta', maxcol=3, maxrow=4, params_n=dmat.shape[1],
        save_fig=True, path=output_dir
    )

    plots.coeffs_plot(
        M, 'ls_beta', ls_dmat.columns, chain=None,
        save_fig=True, path=output_dir
    )
    
    plots.coeffs_plot(
        M, 'beta', dmat.columns, chain=None,
        save_fig=True, path=output_dir
    )

    plots.plot_posterior_panel(
        M.trace('ls_beta', chain=None)[:], ls_dmat.columns,
        show_stats=True, maxrow=3, save_fig=True, path=output_dir,
        fname='ls_beta_posteriors.png'
    )
    
    plots.plot_posterior_panel(
        M.trace('beta', chain=None)[:], dmat.columns,
        show_stats=True, maxrow=3, save_fig=True, path=output_dir,
        fname='beta_posteriors.png'
    )

    Matplot.plot(M.eps_sd, path=output_dir)
    Matplot.plot(M.tau_car, path=output_dir)
    Matplot.plot(M.disp, path=output_dir)

    # plot maps
    w = M.trace('w', chain=None)[:].mean(axis=0)
    ls_beta = M.trace('ls_beta', chain=None)[:].mean(axis=0)
    ls_pred = np.exp(np.array(pred_dmat).dot(ls_beta) + w) #/0.5**2

    layers = [ls_pred, w, pred_dmat.wolf, pred_dmat.conif]
    ls_pred_max = int(np.percentile(ls_pred, 95).round())
    lims = [(0,ls_pred_max), (-0.5,0.5), (-2,2), (-2,2)]
    titles = [
        'Relative density surface',
        'Spatial random effects',
        'Wolf space use',
        '% coniferous'
    ]
    main = ' '.join(subdir.split('_'))

    plots.plot_maps(
        layers, coords_grid, lims, titles, res=500,
        cmap='gist_earth', interp='nearest', figsize=(8,8), dpi=200, #spline16
        nrows=2, ncols=2, main=main, path=output_dir, save_fig=True,
        fname='prediction_maps_gist_earth.png'
    )
    
    plots.plot_maps(
        layers, coords_grid, lims, titles, res=500,
        cmap='afmhot', interp='nearest', figsize=(8,8), dpi=200, #spline16
        nrows=2, ncols=2, main=main, path=output_dir, save_fig=True,
        fname='prediction_maps_afmhot.png'
    )
    
    plots.plot_diagnostics2(
        M, counts, figsize=(14,7), save_fig=True, path=output_dir,
        fname='model_diagnostics.png',
    )



    
