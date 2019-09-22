import os
import numpy as np
import pandas
from scipy.spatial.distance import cdist
from sklearn.preprocessing import scale


def get_neighb_matrix(coords, resolution=500, size=1):
    D = cdist(coords, coords, 'euclidean')
    lim = np.ceil(size*resolution*np.sqrt(2))
    NM = (D < lim).astype(int)
    np.fill_diagonal(NM, 0)
    return NM

    
def scale_columns(df, columns, exclude=[]):
    for c in columns:
        if not c in exclude:
            df[c] = scale(df[c])
    return df


def add_quadratic_terms(df, columns, exclude=[]):
    for c in columns:
        if not c in exclude:
            df[c+'2'] = df[c]**2
    return df


def load_data(species, msubtype):

    if msubtype == 'ungulates':
        ddf = pandas.read_csv('./data/camera_trapping_ungulates.csv')
        ddf['counts'] = ddf[species]
        # one deployment per location - no need for further aggregations
    else:
        ddf = pandas.read_csv('./data/camera_trapping_carnivores.csv')
        ddf['counts'] = ddf[species]
        # multiple deployments per location - aggregate data per location
        ddf2 = ddf.groupby(['location_id'])[['days','counts']].sum().reset_index()
        ddf2 = ddf2.merge(
            ddf.drop_duplicates('location_id')[['location_id','Xpl92','Ypl92']],
            on='location_id', how='inner'
        )
        ddf = ddf2
        ddf['deployment_id'] = ddf.location_id
        ddf = ddf.sort_values(by=['deployment_id'])
        ddf.reset_index(inplace=True, drop=True)

    # load landscape-level grid
    grid_ls = pandas.read_csv('./data/grid_500m.csv')
    coords_grid = grid_ls[['X', 'Y']].values
    coords_cams = ddf[['Xpl92', 'Ypl92']].values
    NM = get_neighb_matrix(coords_grid, resolution=500, size=1)

    D_grid_locs = cdist(coords_grid, coords_cams, 'euclidean')
    ddf['pixels'] = np.argmin(D_grid_locs, axis=0)

    ls_pixels = ddf.pixels.unique()
    ls_pixels.sort()
    ls_pixels_df = pandas.DataFrame({'pixels': ls_pixels})
    ls_pixels_df['pixels_new'] = ls_pixels_df.index

    ddf = ddf.merge(ls_pixels_df, on='pixels', how='left')

    covs_ls = pandas.read_csv('./data/covariates_landscape.csv')
    covs_ls.fillna(0, inplace=True) # workaround
    covs_ls = scale_columns(covs_ls, covs_ls.columns)
    covs_ls = add_quadratic_terms(covs_ls, ['open', 'conif'])

    if msubtype == 'ungulates':
        covs = pandas.read_csv('./data/covariates_site_ungulates.csv')
        covs = scale_columns(covs, covs.columns)
        covs = add_quadratic_terms(covs, ['open', 'conif', 'temp'])
    else:
        covs = pandas.DataFrame(np.ones(len(ddf)))

    return {
        'ddf': ddf,
        'covs_ls': covs_ls,
        'covs': covs,
        'coords_grid': coords_grid,
        'pixel_idx': ls_pixels_df.pixels.values,
        'site_idx': ddf.pixels_new.values,
        'NM': NM,
        'desc': species,
        'ls_resolution': 500 
    }
