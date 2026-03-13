from utils.operators import graph_to_hamiltonian
import networkx as nx
from algorithms.lcqaoa import LCQAOA
from algorithms.rqaoa import RQAOA
from algorithms.qaoa import QAOA

class AlgorithmFactory:
    @staticmethod
    def create(name, G, p):
        if name == "qaoa":
            H = graph_to_hamiltonian(nx.to_numpy_array(G), len(G.nodes))
            return QAOA(p, H)
        if name == "rqaoa":
            H = graph_to_hamiltonian(nx.to_numpy_array(G), len(G.nodes))
            Q = QAOA(p, H)
            return RQAOA(p, H, Q, G)
        if name == "lcqaoa":
            return LCQAOA(G, p)
        raise ValueError("Unknown algorithm")



