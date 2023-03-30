#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:09:20 2023

@author: hugobrehier
"""

import matplotlib.pyplot as plt

import numpy as np
import autograd.numpy as anp
#import jax.numpy as jnp

import pymanopt
from pymanopt.manifolds import ComplexStiefel
from pymanopt.solvers import SteepestDescent,TrustRegions

@pymanopt.function.Autograd
def cost(W):
    return anp.linalg.norm(samples - samples @ W @ anp.conjugate(W).T)**2

dimension = 3
num_samples = 200
num_components = 2

#Matrix of dims (n,p)  not (p,n)
np.random.seed(123)
samples_re = np.random.randn(num_samples, dimension) 
samples_im = np.random.randn(num_samples, dimension)
samples = samples_re + 1j*samples_im

u = np.diag([3, 1, 2]) +1j*np.diag([3, 1, 2])
samples = samples @ u
samples = samples - samples.mean(axis=0)
 
manifold = ComplexStiefel(dimension, num_components)
problem = pymanopt.Problem(manifold, cost)
solver = SteepestDescent()
W_pca = solver.solve(problem)
Proj_pca = W_pca @ W_pca.conj().T

#Analytic W
eigenvalues, eigenvectors = np.linalg.eig(samples.conj().T @ samples)
indices = np.argsort(eigenvalues)[::-1][:num_components]
W_an = eigenvectors[:, indices]
Proj_an = W_an @ W_an.conj().T

np.linalg.norm(Proj_pca - Proj_an)

fig, axs = plt.subplots(1,2,sharey=(True))
axs[0].imshow(np.abs(Proj_pca))
axs[0].set_title('Projector PCA via Riem Grad')
axs[1].imshow(np.abs(Proj_an))
axs[1].set_title('Projector PCA via Analytic form')
