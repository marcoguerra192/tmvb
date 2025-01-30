'''Module to generate the polytope '''

import numpy as np
from numpy.linalg import matrix_rank, pinv
from scipy.linalg import svd
from numpy.random import dirichlet as Dir
import cdd

from utils.utils import WriteMod2, orderings
from statistics.pmf import PMF


class Polytope():
    '''Class that defines a polytope and methods to work with it.
    
    Here we define a polytope as a subset of the hypercube, we compute the dimension of the polytope, and provide a basis.
    With this we can sample points on the polytope
    Then we will try to compute the stratification. 
    '''
    
    def __init__(self, d, rays):
        ''' Constructor
        
        d is the ambient dimension, the variation of the Bernoulli random variable.
        rays is a (2**d , NRays) numpy array
        '''
        
        self.d = d
        
        if not isinstance( rays , np.ndarray ):
            raise ValueError("Wrong type for rays")
        
        self.Rays = rays
        self.NRays = rays.shape[1]
        self.D = 2**d
        
        if rays.shape[0] != self.D:
            raise ValueError("Wrong size for ray points")
            
            
        self.Polygon = np.zeros((rays.shape[0], rays.shape[1]-1)) 

        self.V0 = rays[:,[0]]

        for i in range(self.NRays-1):
            self.Polygon[:,[i]] = rays[:,[i+1]] - self.V0

        self.DimPolytope = matrix_rank(self.Polygon)
        self.Basis = None
        self.Vertices = None
        self.Triangles = None
        self.AreaWeights = None
            
        
    def getDimension(self):
        '''Return the dimension of the polytope'''
        
        return self.DimPolytope
    
    
    def computeBasis(self, randomize = False):
        '''Compute a set of applied vectors in V0 that span the polytope. If randomize, first shuffle the vertices '''
        
        if randomize:
            perm = np.random.permutation(range(self.NRays-1))
            
        else:
            perm = np.array(range(self.NRays-1))
            
        poly = self.Polygon[ : , perm ]
        u, s, vt = np.linalg.svd(poly, full_matrices=True)
        indices = np.where( s >= 1E-15 )[0]
        
        self.Basis = self.Polygon[ :, indices ]
        
        return self.Basis
        
    def computeProjector(self):
        '''If DimPoly<=2 compute pseudoinverse of self.Basis to project points on a line/plane, for plotting'''
        
        if self.DimPolytope not in [1,2]:
            self.projector = None
            raise ValueError("Only works with DimPolytope = 1,2")
        
        if self.Basis is None:
            
            self.computeBasis();
            
        self.projector = pinv(self.Basis)
        return self.projector
    
    def computeVertices2D(self):
        '''IF DimPoly == 2, compute the projection of the Rays - V0 onto the plane, arranged counterclockwise 
        
        To do so, it projects the Polygon vertices (Rays - V0) onto the plane via the pseudoinverse projector.
        To arrange them counterclockwise, it computes the barycenter, the rims from the barycenter to each vertex
        of the polygon, and computes the polar coordinates angle. Then it sorts them by increasing angle.
        '''
        
        if self.DimPolytope != 2:
            self.Vertices = None
            raise ValueError("This can only work for a 2D polytope")
        
        self.Vertices = np.zeros((2 , self.NRays))
        for i in range(self.NRays-1):
            self.Vertices[:,1:] = self.projector @ self.Polygon
        
        # compute barycenter
        self.Barycenter = self.Vertices @ (1.0/self.NRays * np.ones((self.NRays,1)))
        
        # compute rims
        Rims = self.Vertices - np.tile(self.Barycenter,self.NRays)
        
        def cart2pol(x, y):
            #rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            return phi

        PolarRims = [ cart2pol( x[0] , x[1] ) for x in [ Rims[:,i] for i in range(self.NRays) ] ]
        permutation = np.argsort(PolarRims)
        
        self.Vertices = self.Vertices[:,permutation]
    
        
        return self.Vertices
    
    def triangulate2D(self):
        '''Build a triangulation of the 2D polytope and compute cumulative area fractions '''
        
        if self.Triangles is None:
            
            self.Triangles = []
            for i in range(self.NRays):

                self.Triangles.append([ self.Barycenter , self.Vertices[:,[i]] , self.Vertices[:, [(i+1)%self.NRays]] ])
        
            self.AreaWeights = np.zeros((len(self.Triangles),1))

            for i in range(self.NRays):

                self.AreaWeights[i] = 0.5 * np.abs(self.Triangles[i][0][0] * ( self.Triangles[i][1][1] - self.Triangles[i][2][1] ) + \
                                            self.Triangles[i][1][0] * (self.Triangles[i][2][1] - self.Triangles[i][0][1]) + \
                                            self.Triangles[i][2][0] * ( self.Triangles[i][0][1] - self.Triangles[i][1][1] ))

            self.AreaWeights = (1/np.sum(self.AreaWeights)) * self.AreaWeights
            self.AreaWeights = np.cumsum(self.AreaWeights)
            
            
    def UniformSample2D(self, NPoints):
        
        if self.DimPolytope != 2:
            raise ValueError("Dim is not 2")
            
        if self.Triangles is None:
            self.triangulate2D()
            
        sample2D = np.zeros( (2,NPoints) )
        samplePtp = np.zeros( ( self.D , NPoints ) )
        
        for i in range(NPoints):
            
            # choose a triangle with probability = fraction of the area
            selector = np.random.uniform()
            triangle = np.argmin( self.AreaWeights < selector )
            
            [v1 , v2 , v3] = self.Triangles[triangle]
            
            #randomly sample 2D barycentric coordinates
            sampleDir = Dir( [1]*3, 1 ).T
            
            #map to 2D polygon
            point = sampleDir[0]*v1 + sampleDir[1]*v2 + sampleDir[2]*v3
            sample2D[:,[i]] = point
            
        # map into the polytope WITH THE ORIGIN IN V0
        samplePtp = self.Basis @ sample2D
        
        # get the origin back in place and map wrt the Rays
        samplePtp += np.tile(self.V0 , NPoints)
        
        return sample2D, samplePtp
    
    
    def computeStratification(self):
        
        # ???
        pass
    


def computePolytope( d , Cond : list = None ):
    
    ''' Take a list of conditions and use cdd to compute extremal points (rays) '''
    
    D = 2**d # size of sample space
    conditions = [] # list of conditions 
    condition_type =[] # list of types for the corresponding condition: false=ineq, true=eq
    
    
    ordering = orderings[d]

    
    def findEntries( var : list , val : int = 1 ):
        # find the indices in 'ordering' where each variable 'var' equals 'val'. 'Val' defaults to 1
        def check( instance):
            return all([ instance[v-1] == val for v in var ])

        res = np.array(list(map( lambda x: check(x) , ordering)))
        res = np.where(res == True)
        return list(res[0])
    
    def writeCond( val , indices ):
        # set that variables indexed by 'indices' must sum to 'val'. Return the corresponding row of A
        row = [ float(val) ]
        row.extend( [0]*D )
        for i in indices:
            row[i +1] = -1.0 # REMEMBER +1 BEACAUSE WE INTRODUCED VAL AS THE FIRST ELEMENT

        return row
    
    for i in range(D):  # set probabilities positive
        
        cond = []
        cond.extend( [ 0.0] ) # p >= 0
        cond.extend( [0.0 if j != i else 1.0 for j in range(D)]  ) # put +1 in the right spot
        conditions.append( cond)
        condition_type.append(False)
        
    # set probabilities sum to 1
    conditions.append( writeCond( 1 , findEntries([]) )) # set condition on every variable, they must sum to 1
    condition_type.append(True) # it is an EQUALITY
    
    
    if Cond is not None:
        for Val , Vars in Cond:
            conditions.append( writeCond( Val , findEntries(Vars) )) # all probabilities where Vars = 1 sum to Val
            condition_type.append(True) #it is an equality
            
            
            
    size_placeholder = [ 0.0 for _ in range( D + 1 ) ]
    A = cdd.Matrix([size_placeholder], linear= False) # start by an empty condition
    for c,t in zip(conditions, condition_type):
        A.extend( [ c ] , linear=t )

    A.rep_type = cdd.RepType.INEQUALITY
    
    P = cdd.Polyhedron(A)
    R = P.get_generators()
    
    if R.row_size==0:
        raise ValueError('Empty Domain')
        
    flags = np.ones(( R.row_size,1))
    res = np.zeros( (R.row_size, R.col_size-1) )
    for i in range(R.row_size):
        t = R.__getitem__(i)
        flags[i]= t[0]
        res[i,:]= t[1:]
        
        
    if any(flags != 1):
        raise ValueError('Unbounded domain')
        
        
    res = np.matrix(res)
    Rays = res.T # transpose!
    
    ptp = Polytope(d , Rays)
    
    return ptp

# Conditions for dim = 2
conditionsA = [ (0.5 , [ 1 ]), (0.5 , [ 2 ]) , ( 0.5 , [1,2] ) ]

# An OA of strength 2 is given by moments E[x_i] = 0.5, E[x_ij] = 0.25

conditionsOA3 = [ (0.5 , [ 1 ]), (0.5 , [ 2 ]), (0.5 , [ 3 ]), (0.25 , [ 1,2 ]), (0.25 , [ 1,3 ]) , (0.25 , [ 2,3 ])]

conditionsOA4 = [ (0.5 , [ 1 ]), (0.5 , [ 2 ]), (0.5 , [ 3 ]), (0.5 , [ 4 ]),  (0.25 , [ 1,2 ]),  \
                  (0.25 , [ 1,3 ]) , (0.25 , [ 1,4 ]) , (0.25 , [ 2,3 ]), (0.25 , [ 2,4 ]), (0.25 , [ 3,4 ])]

conditionsOA5 = [ (0.5 , [ 1 ]), (0.5 , [ 2 ]), (0.5 , [ 3 ]), (0.5 , [ 4 ]), (0.5 , [ 5 ]) , (0.25 , [ 1,2 ]),  \
                  (0.25 , [ 1,3 ]) , (0.25 , [ 1,4 ]) , (0.25 , [ 1,5 ]) , (0.25 , [ 2,3 ]), (0.25 , [ 2,4 ]), \
                  (0.25 , [ 2,5 ]) , (0.25 , [ 3,4 ]) , (0.25 , [ 3,5 ]) , (0.25 , [ 4,5 ])]

conditionsOA6 = [ (0.5 , [ 1 ]), (0.5 , [ 2 ]), (0.5 , [ 3 ]), (0.5 , [ 4 ]), (0.5 , [ 5 ]) , (0.5 , [ 6 ]), \
                  (0.25 , [ 1,2 ]), (0.25 , [ 1,3 ]) , (0.25 , [ 1,4 ]) , (0.25 , [ 1,5 ]) , (0.25 , [ 1,6 ]) , \
                  (0.25 , [ 2,3 ]), (0.25 , [ 2,4 ]), (0.25 , [ 2,5 ]), (0.25 , [ 2,6 ]), \
                  (0.25 , [ 3,4 ]) , (0.25 , [ 3,5 ]) , (0.25 , [ 3,6 ]), \
                  (0.25 , [ 4,5 ]), (0.25 , [ 4,6 ]), \
                  (0.25 , [ 5,6 ]) ]