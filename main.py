from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np


def centering(n):
  I = np.identity(n)
  O = np.ones((n, n))
  return I - O/n

def double_centering(D):
  n, n = D.shape
  J = centering(n)
  return (-1/2)*J*D*J

def eig_top(B, n):
  w, v = np.linalg.eig(B)
  i = np.argsort(w)[::-1]
  w, v = w[i].real, v[:,i].real
  return w[:n], v[:,:n]

"""Args:
X: input samples, array (num, dim)
n_components: dimension of output data

Returns:
Y: output samples, array (num, n_components)
"""
def MDS(X, n_components=2):
  D = squareform(pdist(X) ** 2)
  B = double_centering(D)
  w, E = eig_top(B, n_components)
  A = np.diag(np.sqrt(np.abs(w)))
  Y = E.dot(A)
  return Y
