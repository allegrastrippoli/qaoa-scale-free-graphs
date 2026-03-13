from utils.utils import *
from utils.operators import ZZ, graph_to_hamiltonian
from scipy.optimize import minimize
from algorithms.qaoa import QAOA
import networkx as nx
import numpy as np
import random

class LightCone:
    def __init__(self, G, u, v, p, ising=True):
        self.ising=ising
        self.u, self.v = u, v
        self.mapping, self.v_sub, self.new_u, self.new_v = compute_subgraph_for_edge(G, u, v)
        v_sub_arr = nx.to_numpy_array(self.v_sub)
        self.n_sub: int = len(self.v_sub.nodes)
        self.H = graph_to_hamiltonian(v_sub_arr,  self.n_sub, self.ising)
        self.Z_uZ_v = ZZ(v_sub_arr, self.new_u, self.new_v, self.n_sub, self.ising)
        self.Q = QAOA(p, self.H)
        
    def expectation(self, angles, ising=True):
        state = self.Q.qaoa_ansatz(angles)
        col_shape = (2**self.Q.n, 1)
        ex = np.vdot(state, state * self.Z_uZ_v.reshape(col_shape))
        if ising: 
            return np.real(ex)
        else:
            return (1-np.real(ex))/2

    def overlap(self, angles):     
        state = self.Q.qaoa_ansatz(angles)                                   
        g_ener = min(self.H)
        olap = 0
        for i in range(len(self.H)):
            if self.H[i] == g_ener:
                olap+= np.absolute(state[i])**2
        return olap
    
class LCQAOA:
    def __init__(self, G, p):
        self.G = G
        self.p = p
        self.light_cones = []
        for u, v in self.G.edges:
            self.light_cones.append(LightCone(G, u, v, p))
        self.history = []

    def sum_of_expectations(self, angles):
        total_energy = 0.0
        for lc in self.light_cones:
            total_energy += lc.expectation(angles)
        return total_energy
    
    def find_bitstring(self, shots=100):
        edge_weights = {}
        for lc in self.light_cones:
            expectation = lc.expectation(self.angles)
            # If min(expval), prob_different = 1.0
            # If max(expval), prob_different = 0.0
            # If expval = 0, prob_different = 0.5
            prob_different = 0.5 * (1 - expectation)
            edge_weights[(lc.u, lc.v)] = prob_different
        samples = {}
        for _ in range(shots):
            assignment = {}
            nodes = list(self.G.nodes())
            np.random.shuffle(nodes)
            assignment[nodes[0]] = np.random.randint(0, 2)
            for i in range(1, len(nodes)):
                u = nodes[i]
                neighbors = [n for n in self.G.neighbors(u) if n in assignment]
                if not neighbors:
                    assignment[u] = np.random.randint(0, 2)
                else:
                    votes_for_0 = 0
                    votes_for_1 = 0
                    for v in neighbors:
                        p_diff = edge_weights.get((u, v)) or edge_weights.get((v, u))
                        val_v = assignment[v]
                        if val_v == 0:
                            votes_for_1 += p_diff
                            votes_for_0 += (1 - p_diff)
                        else:
                            votes_for_0 += p_diff
                            votes_for_1 += (1 - p_diff)
                    total_votes = votes_for_0 + votes_for_1
                    prob_1 = votes_for_1 / total_votes
                    assignment[u] = 1 if np.random.random() < prob_1 else 0
            sorted_nodes = sorted(self.G.nodes())
            bitstring = "".join(str(assignment[n]) for n in sorted_nodes)
            samples[bitstring] = samples.get(bitstring, 0) + 1        
        return max(samples.items(), key=lambda x:x[1])[0]
    
    def save_history(self):
        for lc in self.light_cones:
            overlap = lc.overlap(self.angles)
            ground_state = format(np.argmin(lc.H), f"0{lc.n_sub}b")
            self.history.append({
                "edge": (lc.u, lc.v),
                "ground state": ground_state,
                "overlap": overlap,
                "angles": self.angles
            })
        
    def run(self): 
        initial_angles=[]
        bds= [(0,2*np.pi+0.1)]*self.p + [(0,1*np.pi+0.1)]*self.p
        for i in range(2*self.p):
            if i < self.p:
                initial_angles.append(random.uniform(0,2*np.pi))
            else:
                initial_angles.append(random.uniform(0,np.pi))
        res = minimize(self.sum_of_expectations,initial_angles,method='L-BFGS-B', jac=None, bounds=bds, options={'maxiter': 1000})
        self.angles = res.x  
        self.best_bitstring = self.find_bitstring()
        self.save_history()
