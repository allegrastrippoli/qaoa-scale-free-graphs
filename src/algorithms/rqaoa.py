from scipy.optimize import minimize
from algorithms.qaoa import QAOA
from algorithms.operators import tensor, ZZ
from collections import deque
from algorithms.basealgorithm import BaseAlgorithm
import networkx as nx
import numpy as np

class RQAOA(BaseAlgorithm):
    def __init__(self, depth, H, Q, G):      
        self.initial_graph = G
        self.H = H
        self.Q = Q
        self.G = G
        self.p = depth

        self.mapping = {i: i for i in list(self.G.nodes)}  # The arguments are “mappings”, maps the ids of the initial nodes to the new ids,
                                                                 # assigned after removing nodes during recursion. 
        self.constraints = {}                              # "constraints", stores correlations between edges
        self.history = []
        
    def expectation(self, angles):
        state = self.Q.qaoa_ansatz(angles)
        col_shape = (2**self.Q.n, 1)
        ex = np.vdot(state, state * self.Z_iZ_j.reshape(col_shape))
        return np.real(ex)
    
    def overlap(self, angles):
        state = self.Q.qaoa_ansatz(angles)
        g_ener = min(self.H)
        olap = 0
        for i in range(len(self.H)):
            if self.H[i] == g_ener:
                olap += np.absolute(state[i])**2
        return olap

    def update_mapping_after_removal(self, removed_id):
        new_mapping = {}
        for new_id, original_id in self.mapping.items():
            if new_id < removed_id:
                new_mapping[new_id] = original_id
            elif new_id > removed_id:
                new_mapping[new_id - 1] = original_id
        self.mapping = new_mapping

    def remove_node(self, j):
        G_new = self.G.copy()
        G_new.remove_node(j)
        old_nodes = list(G_new.nodes)
        labels = {old: new for new, old in enumerate(old_nodes)}
        G_new = nx.relabel_nodes(G_new, labels)
        self.update_mapping_after_removal(j)
        self.G = G_new

    def reduce_hamiltonian(self, p, q, sgn):
        n = len(self.G.nodes)
        G_arr = nx.to_numpy_array(self.G)
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
                if G_arr[i][j] == 0:
                    continue
                if (i == p and j == q) or (i == q and j == p):
                    k = [1] * 2**(n-1)
                    k = np.array(k,dtype = 'float64')  
                    H += G_arr[i][j] * sgn * k
                elif i != q and j != q:
                    k[new_index(i)] = Z
                    k[new_index(j)] = Z
                    H += tensor(k) * G_arr[i][j]
                else:
                    other = j if i == q else i
                    if other == p:
                        continue  
                    k[new_index(p)] = Z
                    k[new_index(other)] = Z
                    H += tensor(k) * G_arr[i][j] * sgn
        return H
    
    def save_history(self, angles):
        ground_state = format(np.argmin(self.Q.H), f"0{len(self.G.nodes)}b")
        overlap = self.overlap(angles)

        self.history.append({
            "num_nodes": len(self.G.nodes),
            "ground_state": ground_state,
            "overlap": overlap,
            "angles": angles.copy()
        })
    
    def rqaoa(self, angles):
        def is_terminal():
            for comp in nx.connected_components(self.G):
                sub = self.G.subgraph(comp)
                if sub.number_of_nodes() > 2:
                    return False
            return True
        def extract_terminal_constraints(angles):
            for comp in nx.connected_components(self.G):
                sub = self.G.subgraph(comp)
                if sub.number_of_edges() == 1:
                    i, j = list(sub.edges)[0]
                    Z_iZ_j = ZZ(nx.to_numpy_array(self.G), i, j, len(self.G.nodes))
                    self.Z_iZ_j = Z_iZ_j
                    exp = self.expectation(angles)
                    self.constraints[(self.mapping[i], self.mapping[j])] = np.sign(exp)
        if is_terminal():
            self.save_history(angles)
            extract_terminal_constraints(angles)
            self.angles = angles
            return 
        self.save_history(angles)
        magnitude = {}
        for (i,j) in self.G.edges:
            Z_iZ_j = ZZ(nx.to_numpy_array(self.G), i, j, len(self.G.nodes))
            self.Z_iZ_j = Z_iZ_j
            magnitude[(i, j)] = self.expectation(angles)
        (i,j), max_magn = max(magnitude.items(), key=lambda item: abs(item[1]))
        s = np.sign(max_magn) 
        self.constraints[(self.mapping[i],self.mapping[j])] = s
        self.H = self.reduce_hamiltonian(i, j, s)    
        self.Q = QAOA(self.p, self.H)  
        bds= [(0,2*np.pi+0.1)]*self.p + [(0,1*np.pi+0.1)]*self.p
        res = minimize(self.Q.expectation,angles,method='L-BFGS-B', jac=None, bounds=bds, options={'maxiter': 1000})
        self.remove_node(j)
        return self.rqaoa(res.x)
    
    def run(self, initial_angles=[]):
        if not initial_angles:
            initial_angles = self.initialize_angles()
        self.rqaoa(initial_angles)
        assignment = self.find_assignment(self.constraints)
        maxcut = [assignment[key] for key in sorted(assignment)]
        self.best_bitstring = ''.join(map(str, maxcut))
        
    def find_assignment(self, constraints):
        assignment = {}
        for start in self.initial_graph.nodes:
            if start in assignment:
                continue
            assignment[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in self.initial_graph.neighbors(u):
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
