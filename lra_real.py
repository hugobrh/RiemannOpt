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
    elif backend == "Callable":
        @pymanopt.function.Callable
        def cost(u, s, vt):
            X = u @ np.diag(s) @ vt
            return la.norm(X - A) ** 2

        @pymanopt.function.Callable
        def egrad(u, s, vt):
            X = u @ np.diag(s) @ vt
            S = np.diag(s)
            gu = 2 * (X - A) @ (S @ vt).T
            gs = 2 * np.diag(u.T @ (X - A) @ vt.T)
            gvt = 2 * (u @ S).T @ (X - A)
            return gu, gs, gvt
    else:
        raise ValueError("Unsupported backend '{:s}'".format(backend))

    return cost, egrad


def run(backend=SUPPORTED_BACKENDS[0], quiet=True):
    Y_mp = np.load('../TTW_Bscan_no_interior_walls_ricker_merged_3mm_curated_resampled.npy')
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
    low_rank_approximation = (left_singular_vectors @
                              np.diag(singular_values) @
                              right_singular_vectors)

    if not quiet:
        u, s, vt = la.svd(matrix, full_matrices=False)
        indices = np.argsort(s)[-rank:]
        low_rank_solution = (u[:, indices] @
                             np.diag(s[indices]) @
                             vt[indices, :])
        print("Frobenius norm error:",
              la.norm(low_rank_approximation - low_rank_solution))
        
        fig, axs = plt.subplots(1,3,sharey=(True))
        axs[0].imshow(matrix)
        axs[0].set_title('Y true')
        axs[1].imshow(low_rank_solution)
        axs[1].set_title('Y LRA Frob SVD')
        axs[2].imshow(low_rank_approximation)
        axs[2].set_title('Y LRA Frob RGD')
        plt.show()
        
run(backend="Autograd",quiet=False)
