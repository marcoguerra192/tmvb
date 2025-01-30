''' Probability mass function module

This module implements tools to work with probability mass function in the case of d-variate Bernoulli random variables. 
'''

import numpy as np
from utils.utils import orderings, momentMatrices, BMs


class PMF():
    '''Probability mass function class
    
    Arguments: dimension d, optionally values
    Methods:
    '''
    
    def __init__(self, d, values, one : bool = True):
        '''Constructor: 
        
        It is assumed that the values are ordered by the appropriate utils.utils.orderings list
        d : INT is the dimension of the Bernoulli variable
        values : list or numpy array is the values of the r.v.
        one : optional, boolean specifies if the pmf actually sums to one or is instead integer-valued and sums to some integer.
        This is useful for orthogonal arrays and computing persistence. Defaults to True. If True, the values MUST be floats. If
        False, the values MUST be ints.
        '''      
        
        self.d = d
        self.D = 2**d
        
        self.normalized = one
        
        if one:
            datatype = float
            if abs(np.sum(values) - 1.0) > .0000001:
                raise ValueError("Normalized pmf does not sum to 1")
                
        else:
            datatype = int
            
        if isinstance(values, list):

            if len(values) != self.D:
                raise ValueError( "The length of the value list does not match the dimension" )

            self.values = np.array( values , dtype = datatype )

        elif isinstance(values, np.ndarray):

            if values.shape != (self.D, 1):
                raise ValueError( "The length of the value list does not match the dimension" )

            self.values = values.astype(datatype)
            
        else:
            raise ValueError("Non-compatible type for values")
            
    def __str__(self):
        '''Overload print '''
        
        return str(self.values)
    
    def total(self):
        '''Return sum of pmf '''
        
        return np.sum(self.values)
                
    def normalize(self):
        '''Return a normalized pmf '''
        
        if self.normalized:
            return self
        
        else:
            
            total = np.sum(self.values)
            vals = np.reshape(1/total * self.values.astype(float) , (self.D,1))
            return PMF( self.d , vals , one = True )
        
    def is_normalized(self):
        return self.normalized
    
    
    def compute_moments(self):
        '''Return the (self.D x 1) array of moments of the pmf,
        ordered according to orderings[self.d]. If the pmf is not noraìmalized, 
        they will be integers >=1
        '''
        
        return np.array(np.reshape(momentMatrices[self.d] @ self.values , (self.D,1)))
        
        
           
    def compute_filtration(self):
        '''Return the filtration associated to the pmf
        as a vector (self.D x 1)
        '''
        
        if self.normalized:
            
            return np.ones( self.values.shape, dtype=float ) - self.compute_moments()
        
        else:
            return self.total() * np.ones( self.values.shape, dtype=int ) - self.compute_moments()

