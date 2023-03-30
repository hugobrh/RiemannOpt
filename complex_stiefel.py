import numpy as np
from scipy.linalg import expm

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold
from pymanopt.tools.multi import multiprod, multiherm, multiconj

class ComplexStiefel(EuclideanEmbeddedSubmanifold):
    """
    Factory class for the Stiefel manifold. Instantiation requires the
    dimensions n, p to be specified. Optional argument k allows the user to
    optimize over the product of k Stiefels.

    Elements are represented as n x p matrices (if k == 1), and as k x n x p
    matrices if k > 1 (Note that this is different to manopt!).
    """

    def __init__(self, n, p, k=1):
        self._n = n
        self._p = p
        self._k = k

        # Check that n is greater than or equal to p
        if n < p or p < 1:
            raise ValueError("Need n >= p >= 1. Values supplied were n = %d "
                             "and p = %d." % (n, p))
        if k < 1:
            raise ValueError("Need k >= 1. Value supplied was k = %d." % k)

        if k == 1:
            name = "Stiefel manifold St(%d, %d)" % (n, p)
        elif k >= 2:
            name = "Product Stiefel manifold St(%d, %d)^%d" % (n, p, k)
        dimension = int(k*(2*n*p - p**2))
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self._p * self._k)

    def inner(self, X, G, H):
        # Inner product (Riemannian metric) on the tangent space
        # For the stiefel this is the Frobenius inner product.
        return np.real(np.tensordot(np.conjugate(G), H, axes=G.ndim))


    def proj(self, X, U):
        return U - multiprod(X, multiherm(multiprod(multiconj(X), U)))

    # TODO(nkoep): Implement the weingarten map instead.
    def ehess2rhess(self, X, egrad, ehess, H):
        XtG = multiprod(multiconj(X), egrad)
        symXtG = multiherm(XtG)
        HsymXtG = multiprod(H, symXtG)
        return self.proj(X, ehess - HsymXtG)

    # Retract to the Stiefel using the qr decomposition of X + G.
    def retr(self, X, G):
        if self._k == 1:
            # # Calculate 'thin' qr decomposition of X + G
            # q, r = np.linalg.qr(X + G)
            # # Unflip any flipped signs
            # XNew = np.dot(q, np.diag(np.sign(np.sign(np.diag(r)) + 0.5)))
            
            U,_,Vh = np.linalg.svd(X+G,full_matrices=False)
            XNew = U @ Vh
        else:
            XNew = X + G
            
            # for i in range(self._k):
            #     q, r = np.linalg.qr(XNew[i])
            #     XNew[i] = np.dot(
            #         q, np.diag(np.sign(np.sign(np.diag(r)) + 0.5)))
            
            for i in range(self._k):
                U,_,Vh = np.linalg.svd(XNew[i],full_matrices=False)
                XNew[i] =  U @ Vh
                
        return XNew

    def norm(self, X, G):
        # Norm on the tangent space of the Stiefel is simply the Euclidean
        # norm.
        return np.linalg.norm(G)

    # Generate random Stiefel point using qr of random normally distributed
    # matrix.
    def rand(self):
        if self._k == 1:
            X = np.random.randn(self._n, self._p) +1j*np.random.randn(self._n, self._p) 
            q, r = np.linalg.qr(X)
            return q

        X = np.zeros((self._k, self._n, self._p),dtype=np.complex128)
        for i in range(self._k):
            X[i], r = np.linalg.qr(np.random.randn(self._n, self._p) +1j*np.random.randn(self._n, self._p))
        return X

    def randvec(self, X):
        U = np.random.randn(*np.shape(X) +1j*np.shape(X))
        U = self.proj(X, U)
        U = U / np.linalg.norm(U)
        return U

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def exp(self, X, U):
        # TODO: Simplify these expressions.
        if self._k == 1:
            W = expm(np.bmat([[X.conj().T.dot(U), -U.conj().T.dot(U)],
                              [np.eye(self._p,dtype=np.complex128), X.conj().T.dot(U)]]))
            Z = np.bmat([[expm(-X.conj().T.dot(U))], [np.zeros((self._p, self._p),dtype=np.complex128)]])
            Y = np.bmat([X, U]).dot(W).dot(Z)
        else:
            Y = np.zeros(np.shape(X),dtype=np.complex128)
            for i in range(self._k):
                W = expm(np.bmat([[X[i].conj().T.dot(U[i]), -U[i].conj().T.dot(U[i])],
                                  [np.eye(self._p), X[i].conj().T.dot(U[i])]]))
                Z = np.bmat([[expm(-X[i].conj().T.dot(U[i]))],
                             [np.zeros((self._p, self._p),dtype=np.complex128)]])
                Y[i] = np.bmat([X[i], U[i]]).dot(W).dot(Z)
        return Y

    def zerovec(self, X):
        if self._k == 1:
            return np.zeros((self._n, self._p),dtype=np.complex128)
        return np.zeros((self._k, self._n, self._p),dtype=np.complex128)
