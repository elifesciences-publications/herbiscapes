import os
#import datetime
import numpy as np
from pymc import MCMC, ZeroProbability, AdaptiveMetropolis
from models import spat_nmixture_nb


# prepare matrices for restricted spatial regression
def get_rsr_mat(NM, pred_dmat, K_lim=40):
    Npixels = NM.shape[0]
    Wplus = NM.sum(axis=0)
    A = NM
    D = np.diag(Wplus)
    X = np.array(pred_dmat)
    Q = D - A
    I = np.identity(X.shape[0])
    print 'Calculating Morans operator matrix...' 
    P = I - X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)
    omega = Npixels * P.dot(A).dot(P) / A.sum()
    print 'Extracting eigenvectors...'
    val,vec = np.linalg.eig(omega)
    K = vec[:,:K_lim].astype(float)
    return (Q, K)


def run_model(
        mtype, msubtype, model_params, desc,
        niter=20000, nburnin=15000, nthin=5, nchains=3, 
        am_delay=5000, am_interval=1000, burn_till_tuned=True
):
    """
    Run PYMC model
    """

    model = spat_nmixture_nb
    curr_dir = os.getcwd()
    output_dir = os.path.join(curr_dir, 'run', mtype, msubtype, desc)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ##### ---- RUN MODEL ---- #####
    os.chdir(output_dir)
    trace_file = os.path.join(output_dir, desc +'.hdf5')
    if os.path.exists(trace_file):
        os.remove(trace_file)

    while 1:
        try:
            M = MCMC(
                model.build_model(**model_params), 
                db='hdf5', dbname=trace_file
            )
            break
        except ZeroProbability:
            model_params.update({
                'ls_beta_inits': np.random.normal(0,1, model_params['ls_dmat'].shape[1]),
                'beta_inits': np.random.normal(0,1, model_params['dmat'].shape[1])
            })

    M.use_step_method(AdaptiveMetropolis, M.ls_beta, delay=am_delay, interval=am_interval)
    M.use_step_method(AdaptiveMetropolis, M.beta, delay=am_delay, interval=am_interval)
    M.use_step_method(AdaptiveMetropolis, M.eps, delay=am_delay, interval=am_interval)
    M.use_step_method(AdaptiveMetropolis, M.alpha, delay=am_delay, interval=am_interval)

    M.sample(niter, burn=nburnin, thin=nthin, burn_till_tuned=burn_till_tuned)
    
    for i in range(nchains-1):
        M.sample(niter, burn=nburnin, thin=nthin)
        
    M.db.close()
    os.chdir(curr_dir)
    return 0
