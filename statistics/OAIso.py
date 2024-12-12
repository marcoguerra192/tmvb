''' Orthogonal Array Isomorphisms module

This module contains tools to work with isomorphisms of orthogonal arrays
'''

import numpy as np
import itertools

from utils.utils import orderings
from .OA import OA


def binaryReprContains(k :int , i :int):
    '''Check if the binary representation of k has a 1 in position 2^i
    
    i = 0 is the last digit to the right. To answer, divides the integers in blocks of
    length 2**(i+1) and checks if our number is in the rightward half of this interval.
    I.e. if the remainder of the integer division k // 2**(i+1) is at least 2**i (an clearly
    < 2**(i+1), which it is by definition of integer division. 
    '''
        
    if k <0 or i<0:
        raise ValueError
        
    if k < 2**i:
        return False
        
    # integer division
    
    q = k // 2**(i+1) # quotient
    
    r = k - q*2**(i+1) # remainder
    
    return r >= 2**i

def genPermutations( d : int ):
    '''Return all permutations of length d. They are d!
    
    '''
    perms = []
    indices = []
    for i in range(d):
        indices.append(i)
        
    for p in itertools.permutations(indices):
        perms.append(list(p))
        
    return perms

def swapDecomposition( P ):
    '''Decompose a permutation in a sequence of swaps.
    Assumes p is a list
    '''
    l = len(P)
    p = P.copy()
    
    swaps = []
    
    def findSwap(p):
        # Check where the permutation needs swapping
        for i in range(l-1):
            if p[i] > p[i+1]:
                return (i ,i+1)
        return None
    
    s = findSwap(p)
    while(s is not None):
        
        s1,s2 = s[0], s[1]
        swaps.append([s1,s2])
        tmp = p[s1]
        p[s1] = p[s2]
        p[s2] = tmp
        
        s = findSwap(p)
    
    return list(reversed(swaps))


def switchIsomorphism( k : int , i : int):
    '''Describes the permutation associated to an elementary switch of column i, at position k
    
    Describes permutation \pi(k), associated to switching column i. Columns go from 0 to d-1, where 0 is the rightmost element
    and d-1 the leftmost.
    Uses binaryReprContains, that checks if the binary representation of k has a 1 in position i. 
    If binaryReprContains(k,i) then k <- k - 2**i , else k <- k + 2**i
    Returns the new k
    '''
    
    if binaryReprContains(k,i):
        return k - 2**i
    
    else:
        return k + 2**i
    
def applySwitchIso( alpha , d ):
    '''Apply switch alpha. Return the corresponding permutation
    
    alpha should be a list of length 2**d. Returns a numpy array
    '''
    
    accumulator = np.array( list(range(2**d)) , dtype=int)
    
    for c in range(d):
        if alpha[c] == 1:
            # switch column d - c - 1
            i = d - c - 1

            perm = np.zeros( (2**d,) , dtype=int)
            for k in range(2**d):
                perm[k] = switchIsomorphism( k , i )
                
            accumulator = accumulator[ perm ]
    
    return accumulator
    
    
def swapIsomorphism( k : int , i : int, j : int):
    '''Describes the permutation associated to an elementary swap of columns i and j, at position k
    
    Describes permutation \pi(k), associated to swapping columnn i and j in the Full Factorial Design.
    Columns go from 0 to d-1, where 0 is the rightmost element and d-1 the leftmost.
    Uses binaryReprContains, that checks if the binary representation of k has a 1 in position i and/or j. 
    If binaryReprContains(k,i) and not binaryReprContains(k,j) then k <- k - 2**i + 2**j 
    else if binaryReprContains(k,j) and not binaryReprContains(k,i) k <- k - 2**j + 2**i
    else k <- k
    Returns the new k
    '''
    
    ci = binaryReprContains(k,i)
    cj = binaryReprContains(k,j)
    
    if ci and not cj:       
        return k - 2**i + 2**j
    
    if cj and not ci:        
        return k - 2**j + 2**i
    
    return k

def applySwapIso( p , d ):
    '''Apply the isomorphism permutation p. Decompose into swaps and apply them
    
    '''
    
    swaps = swapDecomposition(p)
    
    accumulator = np.array( list(range(2**d)) , dtype=int)
    
    for s in swaps:
        s1, s2 = s[0] , s[1]
        
        i = d - s1 - 1
        j = d - s2 - 1
        
        perm = np.zeros( (2**d,) , dtype=int)
        for k in range(2**d):
            perm[k] = swapIsomorphism( k , i , j )
                
        accumulator = accumulator[ perm ]
    
    return accumulator
    


def allIsomorphisms( d : int ):
    '''Generate all *different* isomorphisms of type switch and column permutation for an OA in dimension d
    
    Input: the dimension d (number of factors)
    Output: A 2**d x NumberOfIsomorphisms matrix of integers
    
    The output matrix contains in each column a permutation of the vector (0,1, ... ,d-1). The columns are all distinct. 
    Each column describes a permutation of the Full Factorial Design corresponding to a given isomorphism of OA's. The action
    of an isomorphism on a mass vector is simply the corresponding permutation of the mass vector. 
    The isomorphisms are generated by composing each of the possible 2**d switches and each of the d! permutations, in both orders.
    Only those resulting in a new permutation are retained. 
    '''
            
    perms = set()
    
    switches = orderings[d].copy()
    permutations = genPermutations(d)
    
    # Switch first
    for alpha in switches:
        
        accumulator = np.array( list(range(2**d)) , dtype=int) # identity
        
        accumulatorS = accumulator[applySwitchIso(alpha, d) ]
        
        for p in permutations:
            
            accumulator = accumulatorS[applySwapIso(p , d)]
            
            perms.add( tuple(accumulator.tolist()) )
            
    # Permutation first
    for p in permutations:
        
        accumulator = np.array( list(range(2**d)) , dtype=int) # identity
        
        accumulatorP = accumulator[applySwapIso(p , d)]
        
        for alpha in switches:
            
            accumulator = accumulatorP[applySwitchIso(alpha, d) ]
            
            perms.add( tuple(accumulator.tolist()) )
        
            
    
    matr = np.zeros( (2**d , len(perms)) , dtype=int )

    for i,p in enumerate(perms):
        arr = np.array(p)

        matr[:,i] = arr
            
    
    return matr


def allSwitchIsomorphisms( d : int ):
    '''Generate all *different* isomorphisms of type switch for an OA in dimension d
    
    Input: the dimension d (number of factors)
    Output: A 2**d x NumberOfIsomorphisms matrix of integers
    
    The output matrix contains in each column a permutation of the vector (0,1, ... ,d-1). The columns are all distinct. 
    Each column describes a permutation of the Full Factorial Design corresponding to a given isomorphism of OA's. The action
    of an isomorphism on a mass vector is simply the corresponding permutation of the mass vector. 
    The isomorphisms are generated by composing each of the possible 2**d switches.
    Only those resulting in a new permutation are retained. 
    '''
            
    perms = set()
    
    switches = orderings[d].copy()
    
    for alpha in switches:
        
        accumulator = np.array( list(range(2**d)) , dtype=int) # identity
        
        accumulatorS = accumulator[applySwitchIso(alpha, d) ]
        
        perms.add( tuple(accumulatorS.tolist()) )
        
    
    matr = np.zeros( (2**d , len(perms)) , dtype=int )

    for i,p in enumerate(perms):
        arr = np.array(p)

        matr[:,i] = arr
            
    
    return matr