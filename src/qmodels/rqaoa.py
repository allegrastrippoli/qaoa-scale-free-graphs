from utils.utils import graph_to_hamiltonian, reduce_hamiltonian, remove_node, find_assignment, ZZ
from scipy.optimize import minimize
from qmodels.qaoa import QAOA
import networkx as nx
import numpy as np

class RQAOA:
    def __init__(self, depth, H, Q, G):      
        self.initial_graph = G
        self.H = H
        self.Q = Q
        self.G = G
        self.p = depth

        self.mapping = {i: i for i in list(self.G.nodes)}  # The arguments are “mappings”, maps the ids of the initial nodes to the new ids,
                                                           # assigned after removing nodes during recursion. 
        self.constraints = {}                              # "constraints", stores correlations between edges
        
    def expectation(self, Z_iZ_j, angles):
        state = self.Q.qaoa_ansatz(angles)
        col_shape = (2**self.Q.n, 1)
        ex = np.vdot(state, state * Z_iZ_j.reshape(col_shape))
        return np.real(ex)
    
    def overlap(self, angles):
        state = self.Q.qaoa_ansatz(angles)
        g_ener = min(self.H)
        olap = 0
        for i in range(len(self.H)):
            if self.H[i] == g_ener:
                olap += np.absolute(state[i])**2
        return olap
    
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
                    exp = self.expectation(Z_iZ_j, angles)
                    self.constraints[(self.mapping[i], self.mapping[j])] = np.sign(exp)
        if is_terminal():
            # print(f"Ground state: {format(np.argmin(Q.H), f"0{len(G.nodes)}b")}, Overlap: {self.overlap(Q, Q.H, angles)}") 
            extract_terminal_constraints(angles)
            return self.constraints
        # print(f"Ground state: {format(np.argmin(Q.H), f"0{len(G.nodes)}b")}, Overlap: {self.overlap(Q, Q.H, angles)}") 
        magnitude = {}
        for (i,j) in self.G.edges:
            Z_iZ_j = ZZ(nx.to_numpy_array(self.G), i, j, len(self.G.nodes))
            magnitude[(i, j)] = self.expectation(Z_iZ_j, angles)
        (i,j), max_magn = max(magnitude.items(), key=lambda item: abs(item[1]))
        s = np.sign(max_magn) 
        self.constraints[(self.mapping[i],self.mapping[j])] = s
        # print(f"Added constraint: {(mapping[i],mapping[j]), s} new values {i, j,  s}")
        self.H = reduce_hamiltonian(nx.to_numpy_array(self.G), len(self.G.nodes), i, j, s)    
        self.Q = QAOA(self.p, self.H)  
        bds= [(0,2*np.pi+0.1)]*self.p + [(0,1*np.pi+0.1)]*self.p
        res = minimize(self.Q.expectation,angles,method='L-BFGS-B', jac=None, bounds=bds, options={'maxiter': 1000})
        self.mapping, self.G = remove_node(self.G, j, self.mapping)
        return self.rqaoa(res.x)
    
    def compute_bitstring(self, constraints):
        # print(f"Correlation map: {constraints}")
        assignment = find_assignment(self.initial_graph, constraints)
        maxcut = [assignment[key] for key in sorted(assignment)]
        return  ''.join(map(str, maxcut))
      
    def run(self, initial_angles=None):
        if initial_angles is None:
            initial_angles=[]
            for i in range(2*self.p):
                if i < self.p:
                    initial_angles.append(random.uniform(0,2*np.pi))
                else:
                    initial_angles.append(random.uniform(0,np.pi))
        return self.rqaoa(initial_angles)    


