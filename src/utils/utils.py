from collections import Counter
from collections import deque
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

def find_assignment(G, constraints):
    assignment = {}
    for start in G.nodes:
        if start in assignment:
            continue
        assignment[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in G.neighbors(u):
                if (u, v) in constraints:
                    sign = constraints[(u, v)]
                elif (v, u) in constraints:
                    sign = constraints[(v, u)]
                else:
                    continue
                expected = assignment[u] if sign == 1 else 1 - assignment[u]
                if v not in assignment:
                    assignment[v] = expected
                    queue.append(v)
                else:
                    if assignment[v] != expected:
                        raise ValueError(f"Constraints are inconsistent node: {u}, {assignment[v]} != {expected}. Constraints {constraints}")
    return assignment

def reduce_hamiltonian(G, n, p, q, sgn):
    dim = 2**(n-1)
    H = np.zeros((dim), dtype='float64')
    Z = np.array([1, -1], dtype='float64')
    def new_index(i):
        if i < q:
            return i
        elif i > q:
            return i - 1
        else:
            raise ValueError("index q eliminated")
    for i in range(n):
        for j in range(i+1, n):
            k = [[1,1]]*(n-1) 
            k = np.array(k,dtype = 'float64')
            if G[i][j] == 0:
                continue
            if (i == p and j == q) or (i == q and j == p):
                k = [1] * 2**(n-1)
                k = np.array(k,dtype = 'float64')  
                H += G[i][j] * sgn * k
            elif i != q and j != q:
                k[new_index(i)] = Z
                k[new_index(j)] = Z
                H += tensor(k) * G[i][j]
            else:
                other = j if i == q else i
                if other == p:
                    continue  
                k[new_index(p)] = Z
                k[new_index(other)] = Z
                H += tensor(k) * G[i][j] * sgn
    return H

def update_mapping_after_removal(mapping, removed_id):
    new_mapping = {}
    for new_id, original_id in mapping.items():
        if new_id < removed_id:
            new_mapping[new_id] = original_id
        elif new_id > removed_id:
            new_mapping[new_id - 1] = original_id
    return new_mapping

def remove_node(G, j, mapping):
    G_new = G.copy()
    G_new.remove_node(j)
    old_nodes = list(G_new.nodes)
    labels = {old: new for new, old in enumerate(old_nodes)}
    G_new = nx.relabel_nodes(G_new, labels)
    return update_mapping_after_removal(mapping, j), G_new
