import numpy as np
from utils.operators import tensor 

def graph_to_hamiltonian(G,n, ising=True): 
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
                if ising:
                    H+= tensor(k)*G[i][j] 
                else:
                    zz = tensor(k) 
                    H += 0.5 * G[i][j] * (1 - zz) 
            j+=1
    return H
