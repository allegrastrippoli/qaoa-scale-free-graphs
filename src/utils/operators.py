import numpy as np

# Function that executes a tensor product over an ordered list of objects(arrays or matrices)
# specified by the argument "k".
def tensor(k):
    t = k[0]
    i = 1
    while i < len(k) :
        t = np.kron(t,k[i])
        i+=1
    return t

def ZZ(G, i, j, n, ising=True): 
    Z = np.array([1,-1],dtype = 'float64')
    k = [[1,1]]*n 
    k = np.array(k,dtype = 'float64')    
    if G[i][j] != 0: 
        k[i] = Z
        k[j] = Z
        if ising:
            return tensor(k)*G[i][j] 
        else:
            zz = tensor(k) 
            return 0.5 * G[i][j] * (1 - zz) 
