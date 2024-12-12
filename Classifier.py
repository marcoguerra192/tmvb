# Test our algorithm to classify OA's
# CASE D=5


import os
import os.path

import numpy as np
from statistics.pmf import PMF
from statistics.OA import OA, genOA, genFiltrations
from statistics.OAIso import allIsomorphisms, allSwitchIsomorphisms
from persistence.pers import compute_perm_classes, persistencePairs, WasDistances, BotDistances, Was1DDistances, DistancesDistr
from persistence.distances import wasserstein2DNoDiag, wasserstein1D
#from geometry.polytope import Polytope, computePolytope
#from geometry.polytope import conditionsOA3, conditionsOA4, conditionsOA5

import random

from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import pandas as pd


N = 24
d = 5


sampleChildren = 500
loop = 250


# generate a distinguished OA arr, and then a family of other OA's arr_i
# Test if arr_i is isomorphic to arr

print('Starting, N =',N, 'd =',d , flush=True)
print('Sampling ', sampleChildren, ' from each class')
print('Looping ', loop, ' times')

# generate all isomorphisms in dimension d
print('Computing isomorphisms', flush=True)
perms = allSwitchIsomorphisms(d)

# generate the fathers
print('Generating OAs', flush=True)

oas , _ , _ = genOA(N,d)

random.shuffle(oas)

sampleOAS = []
Labels = []

for oa in oas:
    
    pmf = oa.to_pmf(normalize=True).values
    sampleOAS.append(pmf)
    
    samplePerms = np.random.choice( perms.shape[1] , (sampleChildren,) )
    
    for i in samplePerms:
        
        sampleOAS.append( pmf[ perms[: , i]] )
        
print('OAs generated', flush=True)

Labels = [True]*(sampleChildren+1)
Labels.extend( [False]*((len(oas)-1)*(sampleChildren+1)) )

# shuffle

# reorder = list(range(len(sampleOAS)))
# random.shuffle(reorder)
# sampleOAS = [sampleOAS[i] for i in reorder]
# labels = [labels[i] for i in reorder]

# pick the first as our p1, and remove it from the p2's
arr = sampleOAS[0]
del sampleOAS[0]
del Labels[0]

# BALANCE THE DATASET

data = []  
labels = []
# add the positive ones
data.extend( sampleOAS[0:sampleChildren] )
labels.extend(Labels[0:sampleChildren] )

sampleNegatives = np.random.choice( list(range(sampleChildren, len(sampleOAS)) ) , (sampleChildren,) , replace = False)

print('Len sampleNeg ', len(sampleNegatives))
print('Max sampleNeg ', max(sampleNegatives))

print('Len labels ', len(Labels))
print('Len sampleOAs ', len(sampleOAS))


data.extend( [sampleOAS[i] for i in sampleNegatives ] )
labels.extend( [Labels[i] for i in sampleNegatives ] )

EPS = 1E-7

def testIso( p1 , p2 ):
    
        #print('p1 = ')
        #print(p1)
        Was1P = wasserstein1D( p1 , p2 )
        
        Was1F = None
        
        Was2D = None
        
        if Was1P > EPS:
            
            #print('1DP')
            #print(p2)
            
            return False
            
        # the Loop!
        
        Pmf1 = PMF(d,p1) # pmf object
        filt1 = Pmf1.compute_filtration() #compute filtration
        
        pp1 = persistencePairs(filt1, d) # compute pairs
        pp1 = pp1[1] # only value pairs, not the integers
        
        for i in range(loop):
            # create pmf objects
            
            Pmf2 = PMF(d,p2)
            filt2 = Pmf2.compute_filtration()

            Was1F = wasserstein1D( filt1 , filt2 )

            if Was1F > EPS: # filtration distance is positive, no need to perform 2D
                #print('1DFilt > 0', Was1F)
                #print(p2)
                
                randInt = np.random.randint(perms.shape[1])
                #print(newPermIndex)
                p2 = p2[perms[:,randInt]] # apply the isomorphism
                continue    

            #print('Zero in d_1 (moments)')
            #print('Pmf1 = ', Pmf1.values)
            #print('Pmf2 = ', Pmf2.values)
            
            #if randInt:
                #print('perm = ', perms[:,randInt])
            
            
            pp2 = persistencePairs(filt2, d) # compute pairs
            pp2 = pp2[1] # only value pairs, not the integers

            Was2D = wasserstein2DNoDiag( pp1, pp2 )

            if Was2D <= EPS:

                #print('Zero in d_2 (PD)')
                return True
            
            #print('d_1 (moments) = 0, d_2 > 0')
            #print('Pmf1 = ', Pmf1.values)
            #print('Pmf2 = ', Pmf2.values)

            randInt = np.random.randint(perms.shape[1])
            #print(newPermIndex)
            p2 = p2[perms[:,randInt]] # apply the isomorphism
            
                
        # if you get to the end of the loop, return false   
        return False
            
        
print('Testing', flush=True)

res = [ testIso( arr , p ) for p in data ]

#Score
print('Scoring', flush=True)

truePos = trueNeg = falsePos = falseNeg = 0

for r,l in zip(res, labels):
    if l:
        if r:
            truePos += 1
        else:
            falseNeg += 1
    else:
        if r:
            falsePos += 1
        else:
            trueNeg += 1
    
print('   Pos - Neg' )
print('Pos ', truePos, ' ',  falseNeg)
print('Neg ', falsePos, ' ',  trueNeg)
print(' ')

try: 
    prec = truePos/(truePos+falsePos)
except:
    prec = 1
    
try:
    recall = truePos/(truePos+falseNeg)
except:
    recall = 1

print('Total correct = ', (truePos+trueNeg)/len(labels) )
print('Precision', prec)
print('Recall', recall)
        
