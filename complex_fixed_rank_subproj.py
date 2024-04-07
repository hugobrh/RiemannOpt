"""
Module containing manifolds of fixed rank matrices.
"""
import numpy as np

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold
from pymanopt.manifolds.complex_stiefel import ComplexStiefel 
from pymanopt.tools import ndarraySequenceMixin
from pymanopt.tools.multi import multiprod, multiherm, multihconj

class ComplexFixedRankSubspaceProjection(EuclideanEmbeddedSubmanifold):
    """
    Manifold of m-by-n matrices of rank k with two factor quotient geometry.

    function M = fixedrankfactory_2factors_subspace_projection(m, n, k)
    
    A point X on the manifold is represented as a structure with two
    fields: L and R. The matrix L (mxk) is orthonormal,
    while the matrix R (nxk) is a full column-rank
    matrix such that X = L*R'.
    
    Tangent vectors are represented as a structure with two fields: L, R.
    
    Note: L is orthonormal, i.e., columns are orthogonal to each other.
    Such a geometry might be of interest where the left factor has a
    subspace interpretation. A motivation is in Sections 3.3 and 6.4 of the
    paper below.
    
    Please cite the Manopt paper as well as the research paper:
        @Article{mishra2014fixedrank,
          Title   = {Fixed-rank matrix factorizations and {Riemannian} low-rank optimization},
          Author  = {Mishra, B. and Meyer, G. and Bonnabel, S. and Sepulchre, R.},
          Journal = {Computational Statistics},
          Year    = {2014},
          Number  = {3-4},
          Pages   = {591--621},
          Volume  = {29},
          Doi     = {10.1007/s00180-013-0464-z}
        }
    """

    def __init__(self, m, n, k):
        self._m = m
        self._n = n
        self._k = k
        self._stiefel_m = ComplexStiefel(m, k)

        name = ("Manifold of complex {m}-by-{n} matrices with rank {k} and quotient "
                "geometry".format(m=m, n=n, k=k))
        dimension = 2*(m + n - k) * k
        super().__init__(name, dimension, point_layout=2)

    @property
    def typicaldist(self):
        return 10*self._k
    
    def prepare(self,X):
        '''
        Some precomputations at the point X to be used in the inner product (and
        pretty much everywhere else
        '''
        return X[1].conj().T @ X[1]
    
    def skew(self,X):
        return 0.5*(X-X.conj().T)

    def symm(self,X):
        return 0.5*(X+X.conj().T)
    
    def stiefel_proj(self, X, U):
        return U - multiprod(X, multiherm(multiprod(multihconj(X), U)))
    
    def inner(self, X, eta, zeta):
        X_RtR = self.prepare(X)
        return np.real(np.trace(eta[0].conj().T @ zeta[0]) +
                       np.trace(np.linalg.solve(X_RtR,eta[1].conj().T @ zeta[1])))

    def norm(self, X, G):
        return np.sqrt(self.inner(X, G, G))

    def egrad2rgrad(self, X, egrad):
        X_RtR = self.prepare(X)
        rgradL = self.stiefel_proj(X[0],egrad[0])
        rgradR = egrad[1] @ X_RtR
        
        return _FixedRankTangentVector((rgradL,rgradR))
    
    def lyapunov_symmetric_eig(self,V, lbd, C, tol=None):
        ''' 
        Solves AX + XA = C when A = A', as a pseudo-inverse, given eig(A).
        function X = lyapunov_symmetric_eig(V, lambda, C)
        function X = lyapunov_symmetric_eig(V, lambda, C, tol)
        Same as lyapunov_symmetric(A, C, [tol]), where A is symmetric, its
        eigenvalue decomposition [V, lambda] = eig(A, 'vector') is provided as
        input directly, and C is a single matrix of the same size as A.
        See also: lyapunov_symmetric sylvester lyap sylvester_nocheck
        '''
        
        #AX + XA = C  is equivalent to DY + YD = M with
        #Y = V'XV, M = V'CV and D = diag(lambda).
        M = V.conj().T @ C @ V
        
        W = np.zeros((len(lbd),len(lbd)),dtype=np.complex128)
        for i in range(len(lbd)):
            for j in range(len(lbd)):
                W[i,j] = lbd[i] + lbd[j]
                
        Y = M / W
        absW = np.abs(W)
        
        if tol is None:
            tol = np.size(C) * np.spacing(np.max(absW))
        
        Y[np.where(absW <= tol)] = 0        
        X = V @ Y @ V.conj()        
        return X

    def nested_sylvester(self,sym_mat,asym_mat):
        lbd,V = np.linalg.eig(sym_mat)
        print(V.shape,lbd.shape)
        X = self.lyapunov_symmetric_eig(V, lbd, asym_mat)
        omega = self.lyapunov_symmetric_eig(V, lbd, X)
        return omega
        
    def proj(self, X, eta):
        X_RtR = self.prepare(X)
        
        eta_L = self.stiefel_proj(X[0],eta[0])
        eta_R = eta[1]
        eta = _FixedRankTangentVector((eta_L,eta_R))
        
        SS = X_RtR
        AS1 = 2*X_RtR @ self.skew(X[0].conj().T @ eta[0]) @ X_RtR
        AS2 = 2*self.skew(X_RtR @ (X[1].conj().T @ eta[1]))
        AS  = self.skew(AS1 + AS2)

        Omega = self.nested_sylvester(SS, AS)
        etaproj_L = eta[0] - X[0] @ Omega
        etaproj_R = eta[1] - X[1] @ Omega
        
        return _FixedRankTangentVector((etaproj_L,etaproj_R))
    
    def _tangent(self, X, Z):
        return self.proj(X, Z)
    
    def tangent2ambient(self, X, Z):
        return Z
    
    def uf(self,X):
        L,_,Rh = np.linalg.svd(X,full_matrices=False)
        return L @ Rh
        
    def retr(self, X, Z,t=1):
        Y_L = self.uf(X[0] + t*Z[0])
        Y_R = X[1] + t*Z[1]
        
        return (Y_L,Y_R)
        
    def rand(self):
        XL = self._stiefel_m.rand()
        XR = np.random.randn(self._n,self._k) + 1j*np.random.randn(self._n,self._k) 
        return (XL,XR)

    def randvec(self, X):
        eta_L = np.random.randn(self._m,self._k) + 1j*np.random.randn(self._m,self._k) 
        eta_R = np.random.randn(self._n,self._k) + 1j*np.random.randn(self._n,self._k) 
        eta = (eta_L,eta_R)
        eta = self.proj(X,eta)
        nrm = self.norm(X, eta)
        return _FixedRankTangentVector((eta_L/nrm,eta_R/nrm))

    def transp(self, X1, X2, G):
        return self.proj(X2,G)

    def zerovec(self, X):
        L_0 = np.zeros(self._m,self._k) +1j*np.zeros(self._m,self._k)
        R_0 = np.zeros(self._n,self._k) +1j*np.zeros(self._n,self._k)
        return _FixedRankTangentVector((L_0,R_0))
    

class _FixedRankTangentVector(tuple, ndarraySequenceMixin):
    def __repr__(self):
        return "_FixedRankTangentVector: " + super().__repr__()
    
    def __add__(self, other):
        return _FixedRankTangentVector((s + o for (s, o) in zip(self, other)))

    def __sub__(self, other):
        return _FixedRankTangentVector((s - o for (s, o) in zip(self, other)))

    def __mul__(self, other):
        return _FixedRankTangentVector((other * s for s in self))

    __rmul__ = __mul__

    def __div__(self, other):
        return _FixedRankTangentVector((val / other for val in self))

    def __neg__(self):
        return _FixedRankTangentVector((-val for val in self))

