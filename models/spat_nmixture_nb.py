from pymc import *
import numpy as np
from scipy import stats

"""
Spatial (RSR) N-mixture Model
"""


def build_model(
    counts, days, NM, 
    ls_dmat, dmat, pred_dmat,
    pixel_idx, site_idx, Q, K,
    ls_beta_inits, beta_inits
):
    
    # Prepare data
    Npixels = NM.shape[0]
    Wplus = NM.sum(axis=0)

    Kll_max = 100
    Kll = np.arange(Kll_max+1)
    Kll_len = Kll_max + 1
    nsites = dmat.shape[0]
    
    alpha_Tau = K.T.dot(Q).dot(K)
        
    ls_beta = MvNormal('ls_beta',
        mu=np.zeros(ls_dmat.shape[1]), tau=np.eye(ls_dmat.shape[1])*0.001,
        value=ls_beta_inits
    ) 

    beta = MvNormal('beta',
        mu=np.zeros(dmat.shape[1]), tau=np.eye(dmat.shape[1])*0.001,
        value=beta_inits
    ) 

    # Johnson et al. 2013, Hughes & Haran 2013
    #tau_car = Gamma('tau_car', alpha=0.5, beta=0.0005, value=1)
    # Royle et al. 2007
    tau_car = Gamma('tau_car', alpha=0.1, beta=0.1, value=1)
    
    eps_sd = Uniform('eps_sd', 0, 100, value=1)
    eps_tau = Lambda('eps_tau', lambda sd=eps_sd: 1.0/sd**2)
    eps = Normal(
        'eps', mu=0, tau=eps_tau,
        value=np.random.normal(0, 1, dmat.shape[0])
    )    
    disp = Gamma('disp', 0.1, 0.01, 1)
    
    
    # ####----- SPATIAL RE (RSR) ------####

    # workaround: add small constant to Tau to make it positive definite
    alpha = MvNormal(
        'alpha', mu=np.zeros(K.shape[1]), tau=tau_car*alpha_Tau+0.000001
    )
    w = Lambda('w', lambda alpha=alpha, K=K: K.dot(alpha))

    
    ####----- RELATIVE DENSITY ------####

    @deterministic(plot=False)
    def _lambda(ls_beta=ls_beta, w=w[pixel_idx]): 
        return np.exp(
            np.dot(ls_dmat, ls_beta) + w
        )

    ####----- DETECTION RATE ------####

    # site-level detection rate (trapping rates)
    
    @deterministic(plot=False)
    def drate(beta=beta, eps=eps): 
        return np.exp(
            np.dot(dmat, beta) + eps    
        )
    
    @deterministic(plot=False)
    def p(_lambda=_lambda[site_idx], disp=disp):
       return disp/(disp+_lambda)

    @observed(dtype=int, plot=False)
    def y(value=counts, p=p, disp=disp, drate=drate*days):
        """ likelihood """
        N = stats.nbinom._logpmf(
            np.tile(Kll,nsites), disp, np.repeat(p,Kll_len)
        ).reshape(nsites,Kll_len)
        D = stats.poisson._logpmf(
            np.repeat(value,Kll_len),
            np.tile(Kll,nsites)*np.repeat(drate,Kll_len)
        ).reshape(nsites,Kll_len)
        ll = N + D
        ll = np.log(np.exp(ll).sum(axis=1)).sum()
        return ll


    
    ####----- DERIVED QUANTITIES ------####

    # New simulated dataset (given parameters)
    @deterministic(plot=False)
    def y_pred(p=p, disp=disp, drate=drate*days):
        N = np.random.negative_binomial(disp, p)
        return np.random.poisson(N*drate)
        
    # Expected value
    @deterministic(plot=False)
    def ExpY(l=_lambda[site_idx], dr=drate*days):
        return l*dr

    # Residuals
    @deterministic(plot=False)
    def resid(expy=ExpY):
        return counts - expy
    
    # Chi-square discrepancy for the actual data
    @deterministic(plot=False)
    def D(expy=ExpY):
        return (counts - expy)**2/(expy+0.00001)

    # Chi-square discrepancy for the simulated ('perfect') data
    @deterministic(plot=False)
    def D_pred(expy=ExpY, y_pred=y_pred):
        return (y_pred - expy)**2/(expy+0.00001)
    
    return locals()
