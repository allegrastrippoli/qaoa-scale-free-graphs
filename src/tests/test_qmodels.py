from utils.utils import graph_to_hamiltonian
from qmodels.lightcones import Simulation, LightCone
from qmodels.rqaoa import RQAOA
from qmodels.qaoa import QAOA
from utils.plots import  *
import networkx as nx
import numpy as np
import itertools
import random

# To run 
# cd qaoa-scale-free-graphs/src
# python -m pytest .
# def maxcut_value(G, bitstring):
#     cut = 0
#     for i, j in G.edges:
#         if bitstring[i] != bitstring[j]:
#             cut += 1
#     return cut

# def brute_force_maxcut(G):
#     n = len(G.nodes)
#     best_value = -1
#     best_bitstring = None
#     for bits in itertools.product([0,1], repeat=n):
#         value = maxcut_value(G, bits)
#         if value > best_value:
#             best_value = value
#             best_bitstring = bits
#     return best_value, best_bitstring

# def test_rqaoa_matches_maxcut():
#     random.seed(0)
#     np.random.seed(0)
#     G = nx.Graph()
#     G.add_edges_from([(0,1),(1,2),(2,0)])
#     H = graph_to_hamiltonian(G=nx.to_numpy_array(G), n=len(G.nodes))
#     Q = QAOA(depth=1, H=H)
#     rq = RQAOA(depth=1, H=H, Q=Q, G=G)
#     constraints = rq.run()
#     bitstring = rq.compute_bitstring(constraints)
#     bitlist = [int(b) for b in bitstring]
#     rqaoa_value = maxcut_value(G, bitlist)
#     exact_value, _ = brute_force_maxcut(G)
#     assert rqaoa_value == exact_value
    
def test_energy_landscape_regular_graph(ising):
    p = 1
    graphs = []
    G1 = nx.Graph()
    G1.add_nodes_from(range(4))
    G1.add_edges_from([(0,1),(0,2),(0,3),(1,3),(1,2)])
    G2 = nx.Graph()
    G2.add_nodes_from(range(8))
    G2.add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,5),(1,6),(1,7)])
    G3 = nx.Graph()
    G3.add_nodes_from(range(5))
    G3.add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4)])
    graphs.append(G1)
    graphs.append(G2)
    graphs.append(G3)
    for i, G in enumerate(graphs):
        L = LightCone(G, 0, 1, p, ising=ising)
        name_str = ""
        if ising:
            name_str = f"ising_{i}"
        else:
            name_str = f"standard_{i}"
        plot_energy_landscape(L.expectation, save_fig=True, index=f"light_cone_{name_str}")
