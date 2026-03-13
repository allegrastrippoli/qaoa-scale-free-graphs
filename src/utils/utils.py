from collections import Counter

import networkx as nx
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

# Each edge contributes with either +1 (the edge is not cut) or -1 (the edge is cut) 
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


def compute_subgraph_for_edge(G, u, v):
    nodes =  set(G.neighbors(u)) | set(G.neighbors(v))
    sorted_nodes = sorted(nodes)
    mapping = {old: new for new, old in enumerate(sorted_nodes)}
    v_sub = nx.Graph()
    v_sub.add_nodes_from(sorted_nodes)
    v_sub.add_edge(u, v)
    for n in G.neighbors(u):
        v_sub.add_edge(u, n)
    for n in G.neighbors(v):
        v_sub.add_edge(v, n)
    v_sub = nx.relabel_nodes(v_sub, mapping)
    return mapping, v_sub, mapping[u], mapping[v]

# Generates a random k-sat instance
def k_sat_instance (n,k,m):
    instance=np.array([])
    for mind in range(m):
        clause = rng.choice(n, size=k, replace=False)
        instance=np.append(instance,clause)
    instance=(instance+1) *((-1)**np.random.randint(2,size=k*m))
    return  instance.reshape(m,k)

# Converts the k-sat instance into a Hamiltonian
def H_sat(inst,n):
    I=[1,1]
    k0=[1,0]
    k1=[0,1]
    inst_tp=0
    for mind in range(inst.shape[0]):
        clause=inst[mind]
        clause_list=[I]*n
        for lind in range(inst.shape[1]):
            if clause[lind]<0:
                clause_list[int(np.abs(clause[lind])-1)]=k1
            else:
                clause_list[int(np.abs(clause[lind])-1)]=k0
        clause_tp=tensor(clause_list)
        inst_tp=inst_tp+clause_tp
    return np.array(inst_tp)

def sample_from_state(state, n, shots=100, return_probs=False):
    state = state.reshape(-1)
    probs = np.abs(state) ** 2
    indices = np.random.choice(len(probs), size=shots, p=probs)
    most_common_index = Counter(indices).most_common(1)
    bitstring = format(most_common_index[0][0], f"0{n}b")
    if return_probs:
        return bitstring, probs
    return bitstring



