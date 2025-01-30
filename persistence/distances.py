'''Module for topological distances

'''

import numpy as np
import ot, ot.plot

import matplotlib.pyplot as plt

import persim

def wasserstein1D(a,b, returnMatching=False):
    '''Compute the (1,2)-Wasserstein distance between two probability distributions on the same support
    
    The 1D Wasserstein computes a bijection between support points, i.e. permutes one probability vector to make it 
    as close as possible to the other. It does NOT consider extra points (such as the diagonal, as in persistent homology).
    
    INPUT:
    a,b are the two probability distributions. They are (n,1)-shaped numpy arrays of nonnegative floats that sum to 1.
    
    OUTPUT:
    - The Wasserstein distance
    - An (n x 2) matrix representing the matching. Row i represents the i-th point of a. Element [i,0] is an integer, 
    representing which point of b is matched to point i of a. Element [i,1] is the cost of this matched pair, i.e. the 
    Euclidean distance between the support points
    - A list of triples, each containing a matching pair and its cost
    '''
    
    n = a.shape[0]
    
    if b.shape[0] != n:
        raise ValueError
    
    xa = a.copy().reshape((n,1))
    xb = b.copy().reshape((n,1))

    # uniform distributions on the 
    da , db = np.ones((n,)) / n , np.ones((n,)) / n
    
    # compute the Euclidean distances between support points : Cost Matrix
    CM = ot.dist(xa,xb, metric='euclidean')
    
    # Compute the optimal matching
    Mat = ot.emd(da, db, CM)
    
    Dist = 0
    pairs = []
    matching = np.zeros( (n,2) )

    for i in range(n):
        match = int(np.argwhere( Mat[i,:]))
        dist = CM[i,match]
        Dist += dist
        pairs.append( [i,match, dist] )
        matching[i,0] = match
        matching[i,1] = dist
        
    pairs = list(sorted(pairs , key = lambda x: x[0]))
    
    if returnMatching:
        return Dist, matching, pairs
    else:
        return Dist

def plot1DMatching(a,b,pairs):
    
    n = a.shape[0]
    plt.figure(1)
    plt.plot(list(range(n)), a,  'xb', label='Source')
    plt.plot(list(range(n)), b,  'xr', label='Target')
    for p in pairs:
        i, j = p[0],p[1]
        plt.plot( [i,j], [a[i] , b[j]], c=[.5, .5, 0] )

    plt.legend(loc=0)
    plt.xticks(range(n))
    plt.title('Matching 1D');


def wasserstein2DNoDiag(dgm1 , dgm2, returnMatching=False):
    '''Compute the (1,2)-Wasserstein distance between two persistence diagrams, EXCLUDING the diagonal.
    Therefore dgm1 and dgm2 MUST have the same number of finite pairs. 
    
    '''
    
    dgm1 = [ x for x in dgm1 if x[1] != np.infty]
    dgm2 = [ x for x in dgm2 if x[1] != np.infty]
    
    n = len(dgm1)
    if n != len(dgm2):
        raise ValueError
        
    xa = np.array(dgm1).reshape((n,2))
    xb = np.array(dgm2).reshape((n,2))
    
    da , db = np.ones((n,)) / n , np.ones((n,)) / n
    
    
    CM = ot.dist(xa,xb, metric='euclidean')
    
    Mat = ot.emd(da, db, CM)

    Dist = 0
    pairs = []
    matching = np.zeros( (n,2) )

    for i in range(n):
        match = int(np.argwhere( Mat[i,:]))
        dist = CM[i,match]
        Dist += dist
        pairs.append( [i,match, dist] )
        matching[i,0] = match
        matching[i,1] = dist

    pairs = list(sorted(pairs , key = lambda x: x[0]))
    
    if returnMatching:
        return Dist, matching, pairs
    
    else:
        return Dist

def plot2DMatching(dgm1, dgm2, pairs):
    
    dgm1 = [ x for x in dgm1 if x[1] != np.infty]
    dgm2 = [ x for x in dgm2 if x[1] != np.infty]
    
    n = len(dgm1)
    if n != len(dgm2):
        raise ValueError
        
    xa = np.array(dgm1).reshape((n,2))
    xb = np.array(dgm2).reshape((n,2))
    
    plt.figure(1)
    plt.plot(xa[:, 0], xa[:, 1], 'xb', label='Source')
    plt.plot(xb[:, 0], xb[:, 1], 'xr', label='Target')
    for p in pairs:
        i, j = p[0],p[1]
        plt.plot( [xa[i,0],xb[j,0]], [xa[i,1] , xb[j,1]], c=[.5, .5, 0] )
    plt.plot([0,1], [0,1], 'k--')
    plt.legend(loc=0)
    plt.title('Matching PD')
    
    
def wassersteinDistanceDiags(dgm1 , dgm2):
    '''Use persim to compute the Wasserstein distance between two barcodes'''
    
    dgm1 = [ x for x in dgm1 if x[1] != np.infty]
    dgm2 = [ x for x in dgm2 if x[1] != np.infty]
    
    return persim.wasserstein(dgm1, dgm2)