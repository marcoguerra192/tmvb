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
sampleChildren = 32
loop = 500


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

print('Len oas', len(oas))
print('There are ', len(perms), ' switch isomorphisms')

sampleOAS = []
Labels = []

for oa in oas:
    
    pmf = oa.to_pmf(normalize=True).values
    sampleOAS.append(pmf) # append the first array
    
    # choose randomly a subset of 'sampleChildren' many from all isomorphisms
    samplePerms = np.random.choice( perms.shape[1] , (sampleChildren,) , replace=False)  
    
    for i in samplePerms: # for each of the sampleChildren chosen isomorphic arrays
        
        sampleOAS.append( pmf[ perms[: , i]] ) # apply the isomorphism and append this new array to the dataset
        
print('OAs generated', flush=True)

# The ground truth labels for the dataset:
# There are len(oas) blocks, each is (sampleChildren + 1) long (the original array, plus its isomorphs). 
# The first block is True (we take p1 from there), the others are all False
Labels = [True]*(sampleChildren+1)
Labels.extend( [False]*((len(oas)-1)*(sampleChildren+1)) )

# shuffle

# reorder = list(range(len(sampleOAS)))
# random.shuffle(reorder)
# sampleOAS = [sampleOAS[i] for i in reorder]
# labels = [labels[i] for i in reorder]

# pick the first as our p1, and remove it from the candidate p2's

arr = sampleOAS[0]
del sampleOAS[0]
del Labels[0]

# BUILD A BALANCED DATASET

data = []  # will contain arrays to test
labels = [] # will contain truth values (isomorphic or not)

# add the isomorphic ones: add to data the whole first block
data.extend( sampleOAS[0:sampleChildren] )
labels.extend(Labels[0:sampleChildren] ) # same for the labels

# now add the non-isomorphic ones: 
# chose 'sampleChildren' many from the indices from the second block until the end in sampleOAS
sampleNegatives = np.random.choice( list(range(sampleChildren, len(sampleOAS)) ) , (sampleChildren,) , replace = False)

# print to check
print('Len sampleNeg ', len(sampleNegatives))
print('Max sampleNeg ', max(sampleNegatives))

print('Len labels ', len(Labels))
print('Len sampleOAs ', len(sampleOAS))

# now add to data and labels the sampled non-isomorphic cases 
data.extend( [sampleOAS[i] for i in sampleNegatives ] )
labels.extend( [Labels[i] for i in sampleNegatives ] )

# Fix a tolerance for numerical zero
EPS = 1E-7


def testIso( p1 , p2 ):
    ''' Function that implements the classification algorithm
    
    '''
    
    #print('p1 = ')
    #print(p1)
    
    # Compute 1d Wasserstein between pmf's
    Was1P = wasserstein1D( p1 , p2 )

    Was1F = None

    Was2D = None

    if Was1P > EPS: # if not zero, they are guaranteed to be non-isomorphic

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

            # choose a random isomorphism
            randInt = np.random.randint(perms.shape[1])
            #print(newPermIndex)
            p2 = p2[perms[:,randInt]] # apply the isomorphism
            continue # next iteration in the loop

        #print('Zero in d_1 (moments)')
        #print('Pmf1 = ', Pmf1.values)
        #print('Pmf2 = ', Pmf2.values)

        #if randInt:
            #print('perm = ', perms[:,randInt])

        # If instead Was1F == 0, compute Was2D
        
        pp2 = persistencePairs(filt2, d) # compute pairs
        pp2 = pp2[1] # only value pairs, not the integers

        Was2D = wasserstein2DNoDiag( pp1, pp2 )

        if Was2D <= EPS: # If this distance is zero, we guess they are isomorphic

            #print('Zero in d_2 (PD)')
            return True

        #print('d_1 (moments) = 0, d_2 > 0')
        #print('Pmf1 = ', Pmf1.values)
        #print('Pmf2 = ', Pmf2.values)
        
        # Instead, if D2 is NOT zero, we keep on searching

        # choose a random isomorphism
        randInt = np.random.randint(perms.shape[1])
        #print(newPermIndex)
        p2 = p2[perms[:,randInt]] # apply the isomorphism
        
        # and pass to the next loop iteration


    # if you get to the end of the loop without ever finding zero, we guess they are not isomorphic   
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
except ZeroDivisionError:
    prec = 1
    
try:
    recall = truePos/(truePos+falseNeg)
except ZeroDivisionError:
    recall = 1

print('Total correct = ', (truePos+trueNeg)/len(labels) )
print('Precision', prec)
print('Recall', recall)
        
