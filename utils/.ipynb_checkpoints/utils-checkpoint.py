''' Utils

Utility module with things such as conversion of numbers modulo 2 and binary orderings of points in the d-hypercube
'''

import numpy as np


def WriteMod2( n : int):
    ''' Function that writes a number mod 2, returns a list of zeros and ones '''


    out = [0 for _ in range(d)]
    if n == 0:
        return out
    for i in range(d-1,-1,-1):
        if n >= 2**i:
            out[i] =1
            n = n - 2**i
    return list(reversed(out))

orderings = {}

d = 2
D = 2**d
ordering2 = list( range(D) )
ordering2 = list(map(WriteMod2, ordering2))

d = 3
D = 2**d
ordering3 = list( range(D) )
ordering3 = list(map(WriteMod2, ordering3))

d = 4
D = 2**d
ordering4 = list( range(D) )
ordering4 = list(map(WriteMod2, ordering4))

d = 5
D = 2**d
ordering5 = list( range(D) )
ordering5 = list(map(WriteMod2, ordering5))

d = 6
D = 2**d
ordering6 = list( range(D) )
ordering6 = list(map(WriteMod2, ordering6))

orderings = {2 : ordering2 , 3 : ordering3 , 4 : ordering4 , 5 : ordering5, 6 : ordering6}


momentMatrix = np.matrix( [[1,1],[0,1]] )
momentMatrix2 = momentMatrix
momentMatrix3 = np.kron( momentMatrix , np.kron( momentMatrix, momentMatrix ))
momentMatrix4 = np.kron( momentMatrix , momentMatrix3)
momentMatrix5 = np.kron( momentMatrix , momentMatrix4)
momentMatrix6 = np.kron( momentMatrix , momentMatrix5)

momentMatrices = {2 : momentMatrix2 , 3 : momentMatrix3 , 4 : momentMatrix4 , 5 : momentMatrix5 , 6 : momentMatrix6 }


BMs = {}
for d in [2,3,4,5,6]:
    D = 2**d
    BMs[d] = np.zeros( (D,D) )
    
    
def genBM( d = 3):
    ordering = orderings[d]
    BM = BMs[d]
    for i,s in enumerate(ordering):
        s = np.array(s)
        if np.sum(s) < 2: # only edges or higher have boundary
            continue
        else:
            ones = s.nonzero()[0]
            for h in ones:
                face = s.copy()
                face[h] = 0
                face = list(face)
                faceindex = ordering.index(face)
                BM[faceindex,i]=1
                
                
for d in [2,3,4,5,6]:
    genBM(d)
    
    

        
    

   
