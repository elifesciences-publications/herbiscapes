import os
import pickle
import numpy as np
import pandas
import patsy
import json
import re

import load_data
from run import get_rsr_mat, run_model


config_files = [
    k for k in os.listdir('./configs/') if re.match('.*\.json$', k)
]

for cfile in config_files:

    with open('./configs/' + cfile, 'r') as handle:
        mconfig = json.load(handle)
    species, c = mconfig
    
    # loop over species 
    for sp in species:
        data = load_data.load_data(species=sp, msubtype=c['msubtype'])
        pixel_idx = data['pixel_idx'] 
        site_idx = data['site_idx']
        formula_ls = c['formula_ls']
        formula = c['formula']
        covs_ls = data['covs_ls']
        covs = data['covs']
        counts = data['ddf'].counts.values.astype(int)
        days = data['ddf'].days.values.astype(int)
        NM = data['NM']

        ls_dmat = patsy.dmatrix(formula_ls, covs_ls.loc[pixel_idx], return_type='dataframe')
        pred_dmat = patsy.dmatrix(formula_ls, covs_ls, return_type='dataframe')
        dmat = patsy.dmatrix(formula, covs, return_type='dataframe')

        Q, K = get_rsr_mat(NM, pred_dmat, K_lim=20)

        ls_beta_inits = np.zeros(ls_dmat.shape[1])
        beta_inits = np.zeros(dmat.shape[1])

        model_params = {
           'counts': counts, 'days': days, 'NM': NM, 'Q': Q, 'K': K,
           'ls_dmat': np.array(ls_dmat), 'dmat': np.array(dmat),
           'pred_dmat': np.array(pred_dmat),
           'pixel_idx': pixel_idx, 'site_idx': site_idx,
           'ls_beta_inits': ls_beta_inits, 'beta_inits': beta_inits
        }

        print
        print 'MODEL: {m}'.format(m=data['desc'])
        print

        M = run_model(
            c['mtype'], c['msubtype'], model_params, data['desc'], niter=c['niter'],
            nburnin=c['nburnin'], nthin=c['nthin'], nchains=c['nchains'],
            burn_till_tuned=c['burn_till_tuned']
        )

        data_to_pickle = {
            'ls_dmat': ls_dmat,
            'dmat': dmat,
            'pred_dmat': pred_dmat,
            'coords_grid': data['coords_grid'],
            'counts': counts
        }
        output_dir = os.path.join('./run/', c['mtype'], c['msubtype'], data['desc'])

        # pickle data
        pickle.dump(data_to_pickle, open(os.path.join(output_dir, 'data.p'), 'wb'))

        # dump json with model config
        with open(os.path.join(output_dir,'mconfig.json'), 'w') as handle:
            json.dump(mconfig, handle, indent=4)
            
