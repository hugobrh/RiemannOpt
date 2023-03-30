#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:14:24 2023

@author: hugobrehier
"""

import matplotlib.pyplot as plt

import numpy as np
import autograd.numpy as anp
#import jax.numpy as jnp

import pymanopt
from pymanopt.manifolds import ComplexFixedRankEmbedded
from pymanopt.solvers import SteepestDescent
from pymanopt.solvers import ConjugateGradient

Y_mp = np.load('../TTW_Bscan_no_interior_walls_ricker_merged_3mm_curated_resampled.npy')
Y_mp = Y_mp / np.linalg.norm(Y_mp)
R = 5

#LRA with least squares via SVD
def lra_frob(X,r):
    '''
    Low rank approx in frobenius norm via SVD (eckart young thm).
    '''
    U,s,Vh = np.linalg.svd(X)
    return U[:,:r] @ np.diag(s[:r]) @ Vh[:r]

Y_lra_frob1 = lra_frob(Y_mp,R)

#Via Riemannian gradient descent (check)
manifold = ComplexFixedRankEmbedded(*Y_mp.shape, R)

@pymanopt.function.Autograd
def cost(u, s, vh):
    X = u @ anp.diag(s) @ vh
    return anp.linalg.norm(X - Y_mp)**2

problem = pymanopt.Problem(manifold, cost=cost)
solver = ConjugateGradient()
U,S,Vh = solver.solve(problem)
Y_lra_frob2 = U @ np.diag(S) @ Vh

np.linalg.norm(Y_lra_frob1 - Y_mp)/np.linalg.norm(Y_mp)
np.linalg.norm(Y_lra_frob2 - Y_mp)/np.linalg.norm(Y_mp)
np.linalg.norm(Y_lra_frob2 - Y_lra_frob1)/np.linalg.norm(Y_lra_frob1)

fig, axs = plt.subplots(1,3,sharey=(True))
axs[0].imshow(np.abs(Y_mp))
axs[0].set_title('Y true')
axs[1].imshow(np.abs(Y_lra_frob1))
axs[1].set_title('Y LRA Frob SVD')
axs[2].imshow(np.abs(Y_lra_frob2))
axs[2].set_title('Y LRA Frob RGD')

