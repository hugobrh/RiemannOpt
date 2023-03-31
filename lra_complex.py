#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:14:24 2023

@author: hugobrehier
"""

import matplotlib.pyplot as plt

import numpy as np
import autograd
import autograd.numpy as anp
#import jax.numpy as jnp

import pymanopt
from complex_fixed_rank import ComplexFixedRankEmbedded
from pymanopt.solvers import ConjugateGradient

np.random.seed(0)
Yl = np.random.randn(100,5) + 1j*np.random.randn(100,5)
Yr = np.random.randn(5,67) + 1j*np.random.randn(5,67)
Y_mp = Yl @ Yr

# #Preprocess Y to samples of zero mean and unit norm
# Y_mp = Y_mp - Y_mp.mean(axis=0)
# col_norms = np.linalg.norm(Y_mp,axis=0)
# Y_mp = Y_mp / col_norms[np.newaxis:,]

#rank
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
def create_cost_function_egrad(Y_mp):
    @pymanopt.function.Callable
    def cost(u, s, vh):
        X = u @ anp.diag(s) @ vh
        tmp = X - Y_mp
        return anp.real(anp.trace(tmp @ anp.conjugate(tmp).T))

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

manifold = ComplexFixedRankEmbedded(*Y_mp.shape, R)
cost, egrad = create_cost_function_egrad(Y_mp)
problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad)
solver = ConjugateGradient()
U,S,Vh = solver.solve(problem)
Y_lra_frob2 = U @ np.diag(S) @ Vh

print(f'||Y_mp - Y_lra_svd|| : {np.linalg.norm(Y_lra_frob1 - Y_mp)}')
print(f'||Y_mp - Y_lra_rgd|| : {np.linalg.norm(Y_lra_frob2 - Y_mp)}')
print(f'||Y_lra_svd - Y_lra_rgd|| : {np.linalg.norm(Y_lra_frob2 - Y_lra_frob1)}')


fig, axs = plt.subplots(1,3,sharey=(True))
axs[0].imshow(np.abs(Y_mp))
axs[0].set_title('Y true')
axs[1].imshow(np.abs(Y_lra_frob1))
axs[1].set_title('Y LRA Frob SVD')
axs[2].imshow(np.abs(Y_lra_frob2))
axs[2].set_title('Y LRA Frob RGD')
plt.show()

