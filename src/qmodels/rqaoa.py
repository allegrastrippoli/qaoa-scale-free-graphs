from utils.utils import graph_to_hamiltonian, reduce_hamiltonian, remove_node, find_assignment, ZZ
from scipy.optimize import minimize
from qmodels.qaoa import QAOA
import networkx as nx
import numpy as np

class RQAOA:
    def __init__(self, G, p):
        self.G = G
        self.p = p
        
    def expectation(self, Q, Z_iZ_j, angles):
        state = Q.qaoa_ansatz(angles)
        col_shape = (2**Q.n, 1)
        ex = np.vdot(state, state * Z_iZ_j.reshape(col_shape))
        return np.real(ex)
    
    def overlap(self, Q, H, angles):     
        state = Q.qaoa_ansatz(angles)                                   
        g_ener = min(H)
        olap = 0
        for i in range(len(H)):
            if H[i] == g_ener:
                olap+= np.absolute(state[i])**2
        return olap
    
    def rqaoa(self, G, p, Q, angles, mapping, constraints):
        def is_terminal(G):
            for comp in nx.connected_components(G):
                sub = G.subgraph(comp)
                if sub.number_of_nodes() > 2:
                    return False
            return True
        def extract_terminal_constraints(G, Q, angles, mapping, constraints):
            for comp in nx.connected_components(G):
                sub = G.subgraph(comp)
                if sub.number_of_edges() == 1:
                    i, j = list(sub.edges)[0]
                    Z_iZ_j = ZZ(nx.to_numpy_array(G), i, j, len(G.nodes))
                    exp = self.expectation(Q, Z_iZ_j, angles)
                    # print(f"Added constraint: {(mapping[i],mapping[j]), np.sign(exp)} new values {i, j,  np.sign(exp)}")
                    constraints[(mapping[i], mapping[j])] = np.sign(exp)
            return constraints
        if is_terminal(G):
            # print(f"Ground state: {format(np.argmin(Q.H), f"0{len(G.nodes)}b")}, Overlap: {self.overlap(Q, Q.H, angles)}") 
            return extract_terminal_constraints(G, Q, angles, mapping, constraints)
        # print(f"Ground state: {format(np.argmin(Q.H), f"0{len(G.nodes)}b")}, Overlap: {self.overlap(Q, Q.H, angles)}") 
        magnitude = {}
        for (i,j) in G.edges:
            Z_iZ_j = ZZ(nx.to_numpy_array(G), i, j, len(G.nodes))
            magnitude[(i, j)] = self.expectation(Q, Z_iZ_j, angles)
        (i,j), max_magn = max(magnitude.items(), key=lambda item: abs(item[1]))
        s = np.sign(max_magn) 
        constraints[(mapping[i],mapping[j])] = s
        # print(f"Added constraint: {(mapping[i],mapping[j]), s} new values {i, j,  s}")
        H_new = reduce_hamiltonian(nx.to_numpy_array(G), len(G.nodes), i, j, s)    
        bds= [(0,2*np.pi+0.1)]*self.p + [(0,1*np.pi+0.1)]*self.p
        Q_new = QAOA(p, H_new)  
        res = minimize(Q_new.expectation,angles,method='L-BFGS-B', jac=None, bounds=bds, options={'maxiter': 1000})
        mapping, G_new = remove_node(G, j, mapping)
        return self.rqaoa(G_new, p, Q_new, res.x, mapping, constraints)
    
    def compute_bitstring(self, constraints):
        # print(f"Correlation map: {constraints}")
        assignment = find_assignment(self.G, constraints)
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
        H = graph_to_hamiltonian(nx.to_numpy_array(self.G), len(self.G.nodes))    
        Q = QAOA(self.p, H) 
        return self.rqaoa(self.G, self.p, Q, initial_angles, {i: i for i in list(self.G.nodes)}, {})    
