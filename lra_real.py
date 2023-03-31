import os
import matplotlib.pyplot as plt

import autograd.numpy as np
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import FixedRankEmbedded
from pymanopt.solvers import ConjugateGradient


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


SUPPORTED_BACKENDS = ("Autograd", "Callable")


def create_cost_egrad(backend, A, rank):
    m, n = A.shape
    egrad = None

    if backend == "Autograd":
        @pymanopt.function.Autograd
        def cost(u, s, vt):
            X = u @ np.diag(s) @ vt
            return np.linalg.norm(X - A) ** 2
    else:
        raise ValueError("Unsupported backend '{:s}'".format(backend))

    return cost, egrad


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    np.random.seed(0)
    Yl = np.random.randn(100,5) + 1j*np.random.randn(100,5)
    Yr = np.random.randn(5,67) + 1j*np.random.randn(5,67)
    Y_mp = Yl @ Yr    
    
    # #Preprocess Y to samples of zero mean and unit norm
    # Y_mp = Y_mp - Y_mp.mean(axis=0)
    # col_norms = np.linalg.norm(Y_mp,axis=0)
    # Y_mp = Y_mp / col_norms[np.newaxis:,]

    matrix = np.abs(Y_mp)
    m,n = Y_mp.shape
    rank = 5
    
    cost, egrad = create_cost_egrad(backend, matrix, rank)
    manifold = FixedRankEmbedded(m, n, rank)
    problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad)
    if quiet:
        problem.verbosity = 0

    solver = ConjugateGradient()
    left_singular_vectors, singular_values, right_singular_vectors = \
        solver.solve(problem)
    lra_rgd = (left_singular_vectors @
                              np.diag(singular_values) @
                              right_singular_vectors)

    if not quiet:
        u, s, vt = la.svd(matrix, full_matrices=False)
        indices = np.argsort(s)[-rank:]
        lra_svd = (u[:, indices] @
                             np.diag(s[indices]) @
                             vt[indices, :])
        print(f'||Y_mp - Y_lra_svd|| : {np.linalg.norm(lra_svd - Y_mp)}')
        print(f'||Y_mp - Y_lra_rgd|| : {np.linalg.norm(lra_rgd - Y_mp)}')
        print(f'||Y_lra_svd - Y_lra_rgd|| : {np.linalg.norm(lra_rgd - lra_svd)}')

        
        fig, axs = plt.subplots(1,3,sharey=(True))
        axs[0].imshow(matrix)
        axs[0].set_title('Y true')
        axs[1].imshow(lra_svd)
        axs[1].set_title('Y LRA Frob SVD')
        axs[2].imshow(lra_rgd)
        axs[2].set_title('Y LRA Frob RGD')
        plt.show()
        
run(backend="Autograd",quiet=False)
