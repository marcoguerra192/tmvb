''' Orthogonal Array module

This module contains tools to work with orthogonal arrays, e.g. to generate OA's, convert them to PMF's, to generate permutation- 
and switch- isomorphic arrays, etc.
'''

import numpy as np
import random
from collections import Counter
from utils.utils import orderings
from .pmf import PMF

import oapackage

class OA():
    ''' Class that defines an orthogonal array object
    
    Methods:
    '''
    def __init__(self, d, points ):
        '''Constructor
        
        Parameters: d int is the dimension of the OA (number of factors)
        points is a numpy array of size (N x 2**d) where N is the number of runs
        '''
        self.d = d
        self.D = 2**d
        
        if not isinstance(points, np.ndarray):
            raise ValueError("Incompatible type of points")
            
        self.N = points.shape[0]
        
        if points.shape[1] != self.d:
            raise ValueError("Incompatible shape of points")
            
        self.points = points
            
    def __str__(self):
        '''Overload print '''
        
        return str(self.points)
    
    def to_pmf(self, normalize = False):
        '''Method to turn an OA to a pmf. If not normalize, leave integer values'''
        
        lOA = self.points.tolist()
        if normalize:
            pmf = np.zeros( (self.D,1), dtype=float )
            
        else:
            pmf = np.zeros( (self.D,1), dtype=int )
        
        
        for i,l in enumerate(orderings[self.d]):

            c = lOA.count(l)
            pmf[i] = c
            
            
        if normalize:
            
            pmf = np.reshape(1/self.N * pmf , (self.D,1))
            
        return PMF(self.d, pmf, one = normalize)
    
    
    def switchCols(self, cols):
        '''Applies the switch isomorphism to column cols and returns a new OA object.'''
        
        pts = self.points.copy()
        
        for c in cols:
            pts[:,[c]] = np.mod( pts[:,[c]] + np.ones((self.N,1), dtype=int),2)
            
        return OA( self.d , pts )
    
    def permuteCols(self, perm):
        '''Permute columns in the OA by permutation perm
        
        Perm must be a numpy array of shape (self.d) that contains a permutation of (0 , 1 , ... , self.d - 1)
        Returns an OA with the permuted points.
        '''
        
        pts = self.points.copy()
        
        pts = np.copy(pts[ : , perm ])
        
        return OA(self.d , pts)
        
        
    def canonicalize(self):
        '''Make the all-0 point among the most common
        
        By applying switchIsomorphism on the appropriate columns, make sure that the all-0 point
        is the most common or at least among the equally-most common. HOWEVER first perform a random 
        shuffle of the points. This is to avoid cheating, as otherwise the canonicalization would
        always return the same OA as the one from which the isomorphism was computed.
        Returns a new OA object.
        '''
        
        # randomly shuffle the points along rows
        points = self.points.copy()
        perm = list(range(points.shape[0]))
        random.shuffle(perm)  # generate a shuffling of the row indices
        
        points = points[perm, :]
        
        
        count = Counter([tuple(p.tolist()) for p in points]) # count occurences of points
        top = max(count, key=count.get) # find the point with the most occurrences

        cols = np.nonzero(top)[0] # top==1 is where you need to switch values
        newOA = self.switchCols(cols)
        
        return newOA
        
    
    def randSwitchIsomorphism(self, n : int , canonical = False, returnSwitch = False):
        '''Method that returns an isomorphic OA obtained by a random switch of n factors
        
        One obtains an isomorphic OA if one switches 0's and 1's in any column. This method returns
        one such isomorphic OA obtained by randomly choosing n factors (columns) to switch.
        If canonical is True, it runs the canonicalization, whereby another switch is performed so as to 
        guaranteed that the all-0 point is (among) the most frequent. 
        Returns a new OA object.
        '''
        
        # choose randomly n columns
        cols = random.sample( list(range(self.d)) , n )
        
        # compute the switch
        newOA = self.switchCols(cols)
            
        if canonical:
            newOA = newOA.canonicalize()
            
        if returnSwitch:
            return newOA, cols
        
        return newOA
        
        
    

    
def genOA(runs = 20, d = 3):
    '''Generate Orthogonal Arrays
    
    Function to generate orthogonal arrays, employing library oapackage
    '''
    
    if runs not in [20,24]:
        raise ValueError("Can only work with 20 or 24 runs")
        
    if d not in [3,4,5,6]:
        raise ValueError("Can only work with dimensions 3,4,5 or 6")
    
    strength = 2
    NFactors = 2
    
    arrayclass = {}
    
    for ncols in range(3,d+1):
        arrayclass[ncols]=oapackage.arraydata_t(NFactors,runs,strength,ncols)
   
    OAs = {}
    for j in range(3,d+1):
        OAs[j] = []
    
    a1=arrayclass[3].create_root()
    
    alist=oapackage.extend_array(a1,arrayclass[3])
    for a1 in alist:
        OAs[3].append(np.array(a1))
        if d==3: continue
        
        blist=oapackage.extend_array(a1,arrayclass[4])
        for b1 in blist:
            OAs[4].append(np.array(b1))
            if d == 4: continue 
            
            clist=oapackage.extend_array(b1,arrayclass[5])
            for c1 in clist:
                OAs[5].append(np.array(c1))
                if d == 5: continue
                
                dlist=oapackage.extend_array(c1,arrayclass[6])
                for d1 in dlist:
                    OAs[6].append(np.array(d1))
                    
    
    oas = [ OA(d, np.reshape(np.array(oa) , (runs,d))) for oa in OAs[d] ]
    NOas = len(oas)
    OaShape = (runs, d)
    
    return oas, NOas, OaShape


def genFiltrations( runs = 20, d = 3, normalize = False ):
    '''Generate a matrix of the OA's and a matrix of the corresponding filtrations'''
    
    oas, noas, sh = genOA(runs, d)
    
    if normalize:
        datatype = np.float
        
    else:
        datatype = np.int
    
    FiltMatrix = np.zeros( (2**d , noas), dtype = datatype )
    
    for i,arr in enumerate(oas):
        
        var = arr.to_pmf(normalize = normalize)
        FiltMatrix[:,[i]] = var.compute_filtration()
        
    return FiltMatrix


        