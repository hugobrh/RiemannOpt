import os
import matplotlib.pyplot as plt
import autograd.numpy as np

import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import SteepestDescent,TrustRegions


SUPPORTED_BACKENDS = ("Autograd", "Callable")


def create_cost_egrad_ehess(backend, samples, num_components):
    dimension = samples.shape[-1]
    egrad = ehess = None

    if backend == "Autograd":
        @pymanopt.function.Autograd
        def cost(w):
            return np.linalg.norm(samples - samples @ w @ w.T) ** 2
    elif backend == "Callable":
        @pymanopt.function.Callable
        def cost(w):
            return np.linalg.norm(samples - samples @ w @ w.T) ** 2

        @pymanopt.function.Callable
        def egrad(w):
            return -2 * (
                samples.T @ (samples - samples @ w @ w.T) +
                (samples - samples @ w @ w.T).T @ samples
            ) @ w

        @pymanopt.function.Callable
        def ehess(w, h):
            return -2 * (
                samples.T @ (samples - samples @ w @ h.T) @ w +
                samples.T @ (samples - samples @ h @ w.T) @ w +
                samples.T @ (samples - samples @ w @ w.T) @ h +
                (samples - samples @ w @ h.T).T @ samples @ w +
                (samples - samples @ h @ w.T).T @ samples @ w +
                (samples - samples @ w @ w.T).T @ samples @ h
            )
    return cost, egrad, ehess


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
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
    samples = np.abs(samples)
    
    cost, egrad, ehess = create_cost_egrad_ehess(
        backend, samples, num_components)
    manifold = Stiefel(dimension, num_components)
    problem = pymanopt.Problem(manifold, cost, egrad=egrad, ehess=ehess)
    if quiet:
        problem.verbosity = 0

    solver = SteepestDescent()
    estimated_span_matrix = solver.solve(problem)

    if quiet:
        return

    estimated_projector = estimated_span_matrix @ estimated_span_matrix.T

    eigenvalues, eigenvectors = np.linalg.eig(samples.T @ samples)
    indices = np.argsort(eigenvalues)[::-1][:num_components]
    span_matrix = eigenvectors[:, indices]
    projector = span_matrix @ span_matrix.T

    print("Frobenius norm error between estimated and closed-form projection "
          "matrix:", np.linalg.norm(projector - estimated_projector))
    
    fig, axs = plt.subplots(1,2,sharey=(True))
    axs[0].imshow(np.abs(estimated_projector))
    axs[0].set_title('Projector PCA via Riem Grad')
    axs[1].imshow(np.abs(projector))
    axs[1].set_title('Projector PCA via Analytic form')

run(backend='Autograd',quiet=False)
