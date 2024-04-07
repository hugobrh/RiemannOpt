#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:06:28 2023

@author: hugobrehier
"""

import matplotlib.pyplot as plt

import numpy as np
import autograd
import autograd.numpy as anp
#import jax.numpy as jnp

import pymanopt
from pymanopt.manifolds import ComplexFixedRankEmbedded
from pymanopt.solvers import SteepestDescent
from helpers import hub,add_noise,wall_returns,lra_frob_svd,lra_hub_prox

#Via Riemannian gradient descent
def create_cost_function_egrad(Y,c):
    @pymanopt.function.Callable
    def cost(u, s, vh):
        Y_hat = u @ anp. diag(s) @ vh
        return anp.sum(hub(Y - Y_hat, c))

    @pymanopt.function.Callable
    def egrad(u, s, vh):
        ''' 
        the cost function being optimized has been defined
        in terms of the low-rank singular value decomposition of X, the
        gradient returned by the autodiff backends will have three components
        and will be in the form of a tuple egrad = (df/du, df/ds, df/dv).
        '''
        gu =  anp.conjugate(autograd.grad(cost,argnum=0)(u, s, vh))
        gs =  anp.conjugate(autograd.grad(cost,argnum=1)(u, s, vh))
        gvh = anp.conjugate(autograd.grad(cost,argnum=2)(u, s, vh))
        return (gu,gs,gvh)

    return cost, egrad

Y_mp = wall_returns(2e9, 2e9, 100, 67, 1)

# #Preprocess Y to samples of zero mean and unit norm
Y_mp = Y_mp - Y_mp.mean(axis=0)
col_norms = np.linalg.norm(Y_mp,axis=0)
Y_mp = Y_mp / col_norms[np.newaxis:,]

#Noise
t_df = 2.2
snr_db = 15
snr = 10**(snr_db/10)

noise_type = 'pt'
noise_dist = 'student'

np.random.seed(1)


Y_mp_noised = add_noise(Y_mp, snr, t_df,noise_type=noise_type,
                        noise_dist=noise_dist)

#Rank
R = 1

# Manopt
manifold = ComplexFixedRankEmbedded(*Y_mp_noised.shape, R)
cost, egrad = create_cost_function_egrad(Y_mp_noised,c=0.01)
problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad)
solver = SteepestDescent(maxiter=1000)
U,S,Vh = solver.solve(problem)

Y_lra_hub_rgd = U @ np.diag(S) @ Vh
Y_lra_hub_prox = lra_hub_prox(Y=Y_mp_noised, c=0.01,
                              mu=0.1, gam=1, tol=1e-5, maxits=100)
Y_lra_frob_svd = lra_frob_svd(Y=Y_mp_noised, r=R)

print(f'||Y_frob_svd - Y_mp|| : {np.linalg.norm(Y_lra_frob_svd - Y_mp)}')
print(f'||Y_hub_prox - Y_mp|| : {np.linalg.norm(Y_lra_hub_prox - Y_mp)}')
print(f'||Y_hub_rgd - Y_mp|| : {np.linalg.norm(Y_lra_hub_rgd - Y_mp)}')

fig, axs = plt.subplots(1,5,sharey=(True))
axs[0].imshow(np.abs(np.log(Y_mp)))
axs[0].set_title('L true')
axs[1].imshow(np.abs(np.log(Y_mp_noised)))
axs[1].set_title('L noisy')
axs[2].imshow(np.abs(np.log(Y_lra_frob_svd)))
axs[2].set_title('L SVD')
axs[3].imshow(np.abs(np.log(Y_lra_hub_prox)))
axs[3].set_title('L Prox')
axs[4].imshow(np.abs(np.log(Y_lra_hub_rgd)))
axs[4].set_title('L RGD')
# fig.suptitle(f'rank-{R} approximations')
# fig.tight_layout()
# plt.subplots_adjust(top=1.25)
plt.savefig('lras_huber.pdf')
plt.show()


# Monte Carlo
nb_mc = 100
dist_svd = []
dist_prox = []
dist_rgd = []

for i in range(nb_mc):
    #Noisy data
    Y_mp_noised = add_noise(Y_mp, snr, t_df,noise_type=noise_type,
                            noise_dist=noise_dist)
    # Manopt
    manifold = ComplexFixedRankEmbedded(*Y_mp_noised.shape, R)
    cost, egrad = create_cost_function_egrad(Y_mp_noised,c=0.01)
    problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad)
    solver = SteepestDescent(maxiter=1000)
    U,S,Vh = solver.solve(problem)
    
    #Results
    Y_lra_hub_rgd = U @ np.diag(S) @ Vh
    Y_lra_hub_prox = lra_hub_prox(Y=Y_mp_noised, c=0.01,
                                  mu=0.1, gam=1, tol=1e-5, maxits=100)
    Y_lra_frob_svd = lra_frob_svd(Y=Y_mp_noised, r=R)

    dist_svd.append(np.linalg.norm(Y_lra_frob_svd - Y_mp))
    dist_prox.append(np.linalg.norm(Y_lra_hub_prox - Y_mp))
    dist_rgd.append(np.linalg.norm(Y_lra_hub_rgd - Y_mp))


print(f'||Y_frob_svd - Y_mp|| : {np.mean(dist_svd)}')
print(f'||Y_hub_prox - Y_mp|| : {np.mean(dist_prox)}')
print(f'||Y_hub_rgd - Y_mp||  : {np.mean(dist_rgd)}')


np.savez("mc_dists", dist_svd=dist_svd, dist_prox=dist_prox, dist_rgd=dist_rgd)

'''
saved_dists = np.load("mc_dists.npz")
dist_svd, dist_prox, dist_rgd = saved_dists['dist_svd'], saved_dists['dist_prox'], saved_dists['dist_rgd']
'''

