'''Module for Computational Topology on multi-variate bernoulli's

'''

import numpy as np
import persim
from scipy.spatial import distance_matrix
from utils.utils import BMs

from .distances import wassersteinDistanceDiags, wasserstein2DNoDiag, wasserstein1D

## Permutation classes of filtrations

def compute_perm_classes( filtrations , d ):
    '''Compute the equivalence classes of orders induced by several filtrations on a common simplicial complex
    
    The function espects a matrix of ( 2**d , NumberOfFiltrations ) values, either int or float, and returns the 
    number of different classes and a list of indices for each.
    d is the dimension
    In case of draws, first comes first
    '''
    
    if not isinstance(filtrations, np.ndarray):
        raise ValueError("Incompatible type for filtration")
        
    
    D = 2**d
    
    if filtrations.shape[0] != D:
        raise ValueError("Incompatible length for filtration")
        
    NArrs = filtrations.shape[1]
        
    classes = np.zeros( filtrations.shape )
        
    positions = np.array(list(range(D)))  # this to ensure first comes first
    
    for i in range(NArrs):
        order = filtrations[:,i] 
        classes[:,i] = np.lexsort((positions, order))
        
        
    cl, indices = np.unique(classes, axis=1, return_inverse=True)
    
    NCl = cl.shape[1]
    
    return NCl, indices


def persistencePairs( filtration, d ):
    '''Compute persistence pairs of a filtration
    
    Vanilla implementation of persistent homology reduction
    filtration has to be (2**d , 1), d is the dimension of the space.
    Imports the boundary matrix from utils, reorders it according to the filtration. Defines a low function and 
    reduces the matrix. Then computes and returns the index and filtration value persistence pairs. 
    '''
    
    
    # Reorder the binary-ordered boundary matrix according to the filtration

    positions = np.array(list(range(2**d)))

    indices = np.lexsort((positions,filtration[:,0])) # make filtration a (2**d , ) array instead of (2**d , 1)
    boundary = BMs[d][:,indices] # reorder columns
    boundary = boundary[indices,:] # and reorder rows
    
    
    def low(v):
        '''Low function '''

        ind = np.where(v == 1)
        try:
            low = np.max(ind)
            return low
        except Exception as e:
            return -1
        
    
    # Compute the boundary matrix reduction, giving the reduced matrix and the list of lows

    D = 2**d
    lows = np.array( [np.infty]*D )

    for i in range(D):

        while True:
            l = low(boundary[:,i])
            if l == -1:
                lows[i] = -1
                break
            try:
                index = np.nonzero(lows == l)[0][0]
            except:
                lows[i] = l
                break

            boundary[:,i] = np.mod(boundary[:,i] + boundary[:,index] , 2)

    # now boundary is reduced and lows are correct
    
    # REMEMBER THE INDICES HAVE BEEN PERMUTED!
    
    available = [x for x in range(D)]
    pairs = []
    ValuePairs = []
    
    for i,n in enumerate(lows.astype(int)):
        if n != -1:
            
            # the correct index is the one in position indices[j]
            pairs.append( [indices[n],indices[i]] )
            available.remove(indices[i])
            available.remove(indices[n])
            
            
    for i in available:
        pairs.append( [indices[i], np.infty] )
        
    # index pairs are set, now value pairs.
    # This time, the indices in 'pairs' are the correct ones
    
    #Filtration = filtration[ : , 0 ]
        
    for b,d in pairs:
       
        if d != np.infty:       
            ValuePairs.append( [ filtration[b,0] , filtration[d,0] ])
        else:
            ValuePairs.append( [ filtration[b,0] , np.infty ])
    
    
    return pairs, ValuePairs


def bottleneckDistanceDiags(dgm1 , dgm2):
    '''Use persim to compute the bottleneck distance between two barcodes'''
    
    dgm1 = [ x for x in dgm1 if x[1] != np.infty]
    dgm2 = [ x for x in dgm2 if x[1] != np.infty]
    
    return persim.bottleneck(dgm1, dgm2)

def WasDistances(filtrations, d, noDiag = False):
    '''Return a matrix of 2-Wasserstein distances
    
    Given a matrix of filtrations (2**d , NumberOfFiltrations) it returns a (NumberOfFiltrations, NumberOfFiltrations)
    square symmetric matrix D where D_ij is the 2-Wasserstein distance between filtration i and j
    If noDiag is True, uses custom Wasserstein distance between diagrams WITHOUT THE DIAGONAL
    '''
    
    size = filtrations.shape[1]
    distances = np.zeros((size,size))
    VP = []
    
    if noDiag:
        wassersteinDist = wasserstein2DNoDiag
    else:
        wassersteinDist = wassersteinDistanceDiags
    
    for i in range(size):
        
        _ , VPi = persistencePairs( filtrations[:,[i]] , d)
        VP.append( VPi )
        
    for i in range(size):   

        for j in range(i,size):

            distances[i,j] = wassersteinDist(VP[i],VP[j])
            distances[j,i] = distances[i,j]
            
    return distances

def BotDistances(filtrations, d):
    '''Return a matrix of \infty-Wasserstein (bottleneck) distances
    
    Given a matrix of filtrations (2**d , NumberOfFiltrations) it returns a (NumberOfFiltrations, NumberOfFiltrations)
    square symmetric matrix D where D_ij is the bottleneck distance between filtration i and j
    '''
    
    size = filtrations.shape[1]
    distances = np.zeros((size,size))
    VP = []
    
    for i in range(size):
        
        _ , VPi = persistencePairs( filtrations[:,[i]] , d)
        VP.append( VPi )
        
    for i in range(size):   

        for j in range(i,size):

            distances[i,j] = bottleneckDistanceDiags(VP[i],VP[j])
            distances[j,i] = distances[i,j]
            
    return distances

def L2Distances(vectors, d):
    '''Return a matrix of L^2 distances
    
    Given a matrix of OA's (2**d , NumberOfOAs) it returns a (NumberOfOAs, NumberOfOAs)
    square symmetric matrix D where D_ij is the L^2 distance between OA i and j
    REMARK: THIS IS NOT BETWEEN FILTRATIONS BUT BETWEEN PMFS ASSOCIATED TO OAs
    '''
    
    if vectors.shape[0] != 2**d:
        raise ValueError
        
    distances = distance_matrix(vectors.T, vectors.T)
            
    return distances

def Was1DDistances(vectors, d):
    '''Return a matrix of 1D Wasserstein distances between pmf's
    
    Given a matrix of OA's (2**d , NumberOfOAs) it returns a (NumberOfOAs, NumberOfOAs)
    square symmetric matrix D where D_ij is the 1D Wasserstein distance between OA i and j
    REMARK: THIS IS NOT BETWEEN FILTRATIONS BUT BETWEEN PMFS ASSOCIATED TO OAs
    '''
    
    if vectors.shape[0] != 2**d:
        raise ValueError
        
    size = vectors.shape[1]
    distances = np.zeros((size,size))
        
    for i in range(size):   

        for j in range(i,size):

            distances[i,j] = wasserstein1D(vectors[:,i] , vectors[:,j])
            distances[j,i] = distances[i,j]
            
    return distances


def AvgDistances(mat, modulo):
    '''Given a block matrix, compute the average distance within-block and outside-block
    
    The distance matrix is intended between object arranged in families of cardinality 'modulo'. The objects
    within the same family are indexed consecutively, so the matrix contains mat.shape[0] / modulo blocks, each
    of size modulo.
    Only the above-diagonal half of the matrix is considered (excluding the diagonal).
    '''

    InDist = 0
    OutDist = 0
    InCount = 0
    OutCount = 0

    NRows = mat.shape[0]
    NOA = int(NRows/modulo)

    for i in range(NRows):
        UpLim = int(modulo*(np.ceil((i+1) / modulo)))

        for j in range(i+1,UpLim):

            InDist += mat[ i,j ]
            InCount += 1

        for h in range(UpLim, NRows):
            OutDist += mat[ i,h ]
            OutCount += 1
            
    return InDist/InCount , OutDist/OutCount


def DistancesDistr(mat, modulo):
    '''Given a block matrix, return the distribution of within-block and outside-block distances
    
    The distance matrix is intended between object arranged in families of cardinality 'modulo'. The objects
    within the same family are indexed consecutively, so the matrix contains mat.shape[0] / modulo blocks, each
    of size modulo.
    Only the above-diagonal half of the matrix is considered (excluding the diagonal).
    '''

    InDist = []
    OutDist = []
    InCount = 0
    OutCount = 0

    NRows = mat.shape[0]
    #NOA = int(NRows/modulo)

    for i in range(NRows):
        UpLim = int(modulo*(np.ceil((i+1) / modulo)))

        for j in range(i+1,UpLim):

            InDist.append( mat[ i,j ] )

        for h in range(UpLim, NRows):
            OutDist.append( mat[ i,h ] )
            
    return InDist , OutDist