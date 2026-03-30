import numpy as np
from enum import Enum, auto

# Function that executes a tensor product over an ordered list of objects(arrays or matrices)
# specified by the argument "k".
def tensor(k):
    t = k[0]
    i = 1
    while i < len(k) :
        t = np.kron(t,k[i])
        i+=1
    return t

def ZZ(G, i, j, n): 
    Z = np.array([1,-1],dtype = 'float64')
    k = [[1,1]]*n 
    k = np.array(k,dtype = 'float64')    
    if G[i][j] != 0: 
        k[i] = Z
        k[j] = Z
        return tensor(k) 

def graph_to_hamiltonian(G,n): 
    H = np.zeros((2**n), dtype = 'float64') 
    Z = np.array([1,-1],dtype = 'float64')
    for i in range(n):
        j = i+1
        while j<n: 
            k = [[1,1]]*n 
            k = np.array(k,dtype = 'float64')
            if G[i][j] !=0: 
                k[i] = Z
                k[j] = Z
                H+= tensor(k)*G[i][j]
            j+=1
    return H
