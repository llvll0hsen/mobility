import os
import sys
import collections
import itertools

import numpy as np
import dill

from util import output_path_files
tracts_dist = dill.load(open(os.path.join(output_path_files,"mobility","antennas_dist.dill"),"rb"))

def get_diss_index(M):
    ''' 
    Dissimilarity index
    M: matrix with rows are spatial units e.g., antenna, postal unit and columns are social groups
    each cell represent number of user from a specific group in that location
    '''

    group_sizes = np.sum(M,0,dtype=np.float)
    m,n = M.shape
    M/= group_sizes[np.newaxis,:]
    r = np.zeros((n,n))
    for i,j in itertools.combinations(range(n),2):
        b = 0.
        for k in xrange(m):
            b+= np.abs(M[k,i] - M[k,j])
        r[i,j] = round(0.5*b,2) 
    return r

def get_exposure_index(M):
    '' '
    Exposure Index
    '''
    group_sizes = np.sum(M,0,dtype=np.float32)
    T = sum(group_sizes)
    tract_sizes = np.sum(M,1,dtype=np.float32)
    n = len(group_sizes)
    
    B = np.zeros((n,n))
    for i,j in itertools.combinations(range(n),2):
        bi = 0.
        bj = 0.
        pxy = sum((M[:,i]/group_sizes[i])*(M[:,j]/tract_sizes))
        pyx = sum((M[:,j]/group_sizes[j])*(M[:,i]/tract_sizes))
        
        B[i,j] = round(pxy,2) 
        B[j,i] = round(pyx,2)
    return B

def get_isolation_index(M):
    group_sizes = np.sum(M,0,dtype=np.float32)
    tract_sizes = np.sum(M,1,dtype=np.float32)
    n = len(group_sizes)
         
    B = np.zeros(n)
    for i in xrange(n):
        pxx = sum((M[:,i]/group_sizes[i])*(M[:,i]/tract_sizes))
        p = group_sizes[i]/T #minority proportion of the whole city
        B[i] = round((pxx - p)/(1-p),2)
    return B

def get_spatial_proximity(M,tract_ids,aid_indices,area=500):
    group_sizes = np.sum(M[tract_ids,:],0,dtype=np.float32)
    n = M.shape[1]
    #Pxx = np.zeros(n,dtype=float)
    p = np.zeros((n,n),dtype=float)
    

    indices = list(itertools.combinations(range(n),2))
    diag_ind = zip(*np.diag_indices(n))
    indices.extend(diag_ind)

    Ptt = np.zeros((n,n),dtype=float)
    for ui,uj in indices: 
        pxx = 0.
        pyy = 0.
        for i,j in itertools.combinations_with_replacement(tract_ids,2):
            try:
                d = tracts_dist[aid_indices[i]][aid_indices[j]]
            except Exception as e:
                d = tracts_dist[aid_indices[j]][aid_indices[i]]
            
            p[ui,uj] += M[i,ui]*M[j,uj]*np.exp(-1*d)
            ptt[ui,uj] += (M[i,ui]+M[i,uj])*(M[j,ui]+M[j,uj])*np.exp(-1*d)
        p[ui,uj] /= (group_sizes[ui] * group_sizes[uj])
        ptt[ui,uj] /= (group_sizes[ui] + group_sizes[uj])
        
        
#    p+= (M[i,ui]*M[j,ui]*np.exp(-1*d)) / (group_sizes[ui]**2) 
        #Pu[ui] = p
#    print Pu 
#    p[diag_ind] /= (group_sizes*2)  
#
#    for ui,uj in itertools.combinations(range(n),2):
#        p = 1e-5
#        for i,j in itertools.combinations_with_replacement(tract_ids,2):
#            try:
#                d = tracts_dist[aid_indices[i]][aid_indices[j]]
#            except Exception as e:
#                d = tracts_dist[aid_indices[j]][aid_indices[i]]
#            p+= ((M[i,ui]*M[i,uj])*(M[j,ui]*M[j,uj])*(np.exp(-1*d))) / ((group_sizes[ui]+group_sizes[uj])**2.) 
#        Ptt[ui,uj] = p
#    
#    print '---------'
#    print Ptt
#    print '---------'
    P = np.zeros((n,n)) 
    for ui,uj in itertools.combinations(range(n),2):
##        print ui,uj 
        n1 = group_sizes[ui]
        n2 = group_sizes[uj]
        P[ui,uj] = ((n1*p[ui,ui])*(n2*P[uj,uj])) / ((n1+n2)*Ptt[ui,uj])
    return P


def delta(M,tract_ids):
    '''
    Concentration refers to relative amount of physical space occupied by a 
    minority group in the urban environment.
    '''

    n = M.shape[0]
    m = M.shape[1]
    
    tract_areas = np.random.rand(n)
    A = 0.2
    tract_areas =/A
    group_sizes = np.sum(M,0,dtype=np.float32)
    
    DEL = np.zeros(m)

    for i in xrange(m):
        for j in xrange(n):
            DEL[i] = sum( (M[:,i]/group_sizes[i]) - tract_areas)

    return 0.5*DEL




if __name__ == '__main__':
    data = np.zeros((6,3))
    data[0,0] = 100
    data[0,1] = 0
    data[0,2] = 10
    
    data[1,0] = 120
    data[1,1] = 10
    data[1,2] = 20
    
    data[2,0] = 40
    data[2,1] = 0
    data[2,2] = 5
    
    data[3,0] = 50
    data[3,1] = 20
    data[3,2] = 50
    
    data[4,0] = 100
    data[4,1] = 20
    data[4,2] = 20

    data[5,0] = 30
    data[5,1] = 90
    data[5,2] = 20

    print data

    D = get_isolation_index(data)
    print D 
#    for row in B:
#        print " ".join(map(str,row))
    
    
