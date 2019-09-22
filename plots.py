import os
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import matplotlib.lines as mpllines
import matplotlib.ticker as mticker
from scipy import ndimage as nd

from pymc import gelman_rubin

import seaborn as sns 
sns.set(color_codes=True)
sns.set_style("white")

plt.rcParams['figure.figsize'] = (8,8)
dpi = 200

def fill_depressions(data, invalid=None):
    if invalid is None: invalid = np.isnan(data)
    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]



#### DIAGNOSTICS ####

def plot_traces(
        model, var_name, maxcol=3, maxrow=4, params_n=11,
        save_fig=True, path='./', fname_prefix='', labels=None
):
    fig, ax = plt.subplots(maxrow,maxcol,figsize=(16,15))
    ax = ax.flatten()
    for p in range(params_n):
        for c in range(model.chains):
            samples = model.trace(var_name, chain=c)[:,p]
            ax[p].plot(samples)
        ax[p].set_ylabel('')
        if labels:
            ax[p].set_xlabel(labels[p], fontweight='bold', fontsize=12)
        else:
            ax[p].set_xlabel(var_name+'['+str(p)+']', fontweight='bold', fontsize=12)
        gr_reshape = (model.chains, samples.shape[0], 1)
        gr = gelman_rubin(model.trace(var_name, chain=None)[:,p].reshape(gr_reshape))
        gr = str(round(gr[0],4))
        ax[p].text(
            0.2, 0.9, gr, ha='center', va='center',
            transform=ax[p].transAxes, color='red'
        )
    if save_fig:
        fig.savefig(
            os.path.join(path, fname_prefix+'_'+var_name+'_traces.png'), dpi=dpi
        )

        
def plot_diagnostics2(
        model, counts, figsize=(8,4), save_fig=True,
        path='./', fname='diagnostics.png', _xlim=(), _ylim=()
):
    y_pred = model.trace('y_pred', chain=None)[:]

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax = ax.flatten()
    fit = model.trace('D', chain=None)[:].sum(axis=1)
    fit_pred = model.trace('D_pred', chain=None)[:].sum(axis=1)
    ax[0].scatter(fit, fit_pred, alpha=0.5, c='black')
    #ax[0].axis('equal')
    if _xlim:
        ax[0].set_xlim(_xlim)
    if _ylim:
        ax[0].set_ylim(_ylim)
    ax[0].plot(np.arange(0,20000,1),np.arange(0,20000,1),'k-', c='red') # identity line
    bpvalue = 'p-value: {p}'.format(
        p=str((fit_pred >= fit).astype(int).mean().round(2))
    )
    ax[0].text(
        0.2, 0.9, bpvalue, ha='center', va='center',
        transform=ax[0].transAxes, color='red', fontweight='bold', fontsize=14
    )
    ax[0].set_title('Bayesian predictive check I', fontweight='bold', fontsize=16)
    ax[0].set_xlabel('Chi-square discrepancy for actual data', fontweight='bold', fontsize=14)
    ax[0].set_ylabel('Chi-square discrepancy for simulated data', fontweight='bold', fontsize=14)
    # bayesian p-value
    #print (fit_pred >= fit).astype(int).mean()

    ax[1].hist(counts, bins=range(0,21), color='black')
    samples = np.random.choice(np.arange(0,y_pred.shape[0]),100)
    for s in samples:
        pp_hist = np.histogram(y_pred[s,:], bins=range(0,21))
        ax[1].plot(np.arange(0.5,20), pp_hist[0], color='red', alpha=0.1)
    ax[1].set_title('Bayesian predictive check II', fontweight='bold', fontsize=16)
    ax[1].set_xlabel('Count', fontweight='bold', fontsize=14)
    ax[1].set_ylabel('Nr of camera trap locations', fontweight='bold', fontsize=14)
    
    if save_fig:
        fig.savefig(
            os.path.join(path, fname), dpi=dpi
        )


#### POSTERIOR SUMMARIES ####

pretty_blue = '#89d1ea'

def hdi_of_mcmc(sample_vec, cred_mass = 0.95):
    """https://github.com/strawlab/best/blob/master/best/plot.py"""

    assert len(sample_vec), 'need points to find HDI'
    sorted_pts = np.sort(sample_vec)
    ci_idx_inc = int(np.floor(cred_mass*len(sorted_pts)))
    n_cis = len(sorted_pts) - ci_idx_inc
    ci_width = sorted_pts[ci_idx_inc:] - sorted_pts[:n_cis]
    min_idx = np.argmin(ci_width)
    hdi_min = sorted_pts[min_idx]
    hdi_max = sorted_pts[min_idx+ci_idx_inc]
    return hdi_min, hdi_max

def calculate_sample_statistics(sample_vec):
    """https://github.com/strawlab/best/blob/master/best/plot.py"""

    hdi_min, hdi_max = hdi_of_mcmc(sample_vec)
    # calculate mean
    mean_val = np.mean(sample_vec)
    # calculate mode
    mode_val = stats.mode(sample_vec)[0]
    return {
        'hdi_min':hdi_min,
        'hdi_max':hdi_max,
        'mean':mean_val,
        'mode':mode_val,
    }

def plot_posterior(
    sample_vec, ax, bins=30, title=None, label='', show_stats=True, 
    draw_zero=True, hist_color=pretty_blue, hist_alpha=1
):
    """https://github.com/strawlab/best/blob/master/best/plot.py"""

    stats = calculate_sample_statistics(sample_vec)
    hdi_min = stats['hdi_min']
    hdi_max = stats['hdi_max']

    ax.hist(
        sample_vec, rwidth=0.8, facecolor=hist_color, 
        edgecolor='none', bins=bins, alpha=hist_alpha
    )
    if title is not None:
        ax.set_title(title)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    if show_stats:
        ax.text(
            stats['mode'], 0.5, '%s = %.3f'%('mode', stats['mode']),
            transform=trans,
            horizontalalignment='center',
            verticalalignment='top',
        )
        ax.text(
            stats['mode'], 0.4, '%s = %.3f'%('mean', stats['mean']),
            transform=trans,
            horizontalalignment='center',
            verticalalignment='top',
        )
    if draw_zero:
        ax.axvline(0,linestyle=':')

    # plot HDI
    hdi_line, = ax.plot(
        [hdi_min, hdi_max], [0,0],
        lw=5.0, color='k'
    )
    hdi_line.set_clip_on(False)
    ax.text(
        hdi_min, 0.04, '%.3g'%hdi_min,
        transform=trans,
        horizontalalignment='center',
        verticalalignment='bottom',
    )
    ax.text(
        hdi_max, 0.04, '%.3g'%hdi_max,
        transform=trans,
        horizontalalignment='center',
        verticalalignment='bottom',
    )
    ax.text( 
        (hdi_min+hdi_max)/2, 0.14, '95% HDI',
        transform=trans,
        horizontalalignment='center',
        verticalalignment='bottom',
    )

    # make it pretty
    ax.spines['bottom'].set_position(('outward',2))
    for loc in ['left','top','right']:
        ax.spines[loc].set_color('none') # don't draw
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks([]) # don't draw
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    for line in ax.get_xticklines():
        line.set_marker(mpllines.TICKDOWN)
    ax.set_xlabel(label, fontweight='bold', fontsize=12)


def plot_posterior_panel(
        samples, names, maxcol=4, maxrow=4, 
        title=None, show_stats=False, save_fig=True,
        path='./', fname='posteriors.png'
):
    f = plt.figure(facecolor='white')
    for i in range(samples.shape[1]):
        ax = f.add_subplot(maxrow,maxcol,i+1,axisbg='none')
        plot_posterior(
            samples[:,i], ax, label=names[i], show_stats=show_stats
        )
    f.tight_layout()
    if save_fig:
        f.savefig(os.path.join(path, fname), dpi=dpi)

        
def coeffs_plot(
        model, var_name, columns, chain=None,
        save_fig=True, path='./'
):
    """
    """
    stats = model.trace(var_name).stats(chain=chain)
    mean = stats['mean']
    hpd95 = stats['95% HPD interval']
    xerr = [
        abs(mean-hpd95[0,:]),
        abs(hpd95[1,:]-mean)
    ]
    signif = hpd95[0,:]*hpd95[1,:] > 0
    y = np.arange(1,len(columns)+1)[::-1]
    f = plt.figure()
    plt.plot((0, 0), (0, mean.size+2), 'k-', linestyle='--')
    for v in y[signif]:
        plt.axhline(y=v, color='red', linewidth=0.3)
    ax = plt.errorbar(mean, y, xerr=xerr, fmt='o', elinewidth=1.8)
    plt.plot(mean[signif], y[signif], 'o', color='red')
    plt.yticks(y, columns)
    if save_fig:
        f.savefig(os.path.join(path, var_name+'_coefs.png'), dpi=dpi)

        

#### PREDICTIONS ####


def plot_maps(
        layers, coords, lims, titles, res=500,
        cmap='gist_earth', interp='spline16',
        nrows=2, ncols=2, main='', figsize=(8,8),
        save_fig=True, path='./',
        fname='prediction_maps.png', dpi=200,
        fill_dep=True, fill_dep_perc=None,
        cbar_shrink=0.7, cbar_font=10,
        hspace=0.001, wspace=0.001, left=0.05,
        right=0.95, top=0.88,
        title_fsize=12, style='white'
):

    sns.set_style(style, {'axes.linewidth': 0.0})
    
    # prepare grid params
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    xi = np.arange(coords_min[0]-2*res,coords_max[0]+3*res,res)
    yi = np.arange(coords_min[1]-2*res,coords_max[1]+3*res,res)
    X,Y = np.meshgrid(xi, yi)
    mgrid_coords = zip(X.ravel(),Y.ravel())
    orig_coords = zip(coords[:,0], coords[:,1])
    ind = [mgrid_coords.index(i) for i in orig_coords]

    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    ax = ax.flatten()
    
    for i,layer in enumerate(layers):

        grid = np.full(X.shape, np.nan).ravel()
        grid.flat[ind] = layer
        grid = grid.reshape(X.shape)

        # just for more smooth maps
        if fill_dep and fill_dep_perc is not None:
            invalid = (grid <= np.percentile(layer, fill_dep_perc[i]))
            grid2 = np.copy(grid)
            grid2 = fill_depressions(grid2, invalid=invalid)
            grid2 = fill_depressions(grid2)
            grid_valid = fill_depressions(grid2, invalid=invalid)[invalid]
            grid[invalid] = grid_valid

        im = ax[i].imshow(
            grid, cmap=cmap, origin='lower', interpolation=interp,
            vmin=lims[i][0], vmax=lims[i][1]
        )
        cbar = fig.colorbar(im, shrink=cbar_shrink, ax=ax[i])
        cbar.ax.tick_params(labelsize=cbar_font) 
        ax[i].set_title(titles[i], fontweight='bold', fontsize=title_fsize)
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        
    plt.suptitle(main, fontweight='bold', fontsize="x-large")
    plt.subplots_adjust(hspace=hspace, wspace=wspace, left=left, right=right, top=top)

    if save_fig:
        fig.savefig(os.path.join(path, fname), dpi=dpi)

