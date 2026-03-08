from utils.utils import graph_to_hamiltonian, sample_from_state
from utils.generate import generate_scale_free_graph
from utils.plots import max_neighborhood_size, plot_subgraphs_maxns
from qmodels.lightcones import Simulation, LightCone
from qmodels.rqaoa import RQAOA
from qmodels.qaoa import QAOA
import networkx as nx
import numpy as np

if __name__ == "__main__":
    p = 1  
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,4)])
    H = graph_to_hamiltonian(nx.to_numpy_array(G), len(G.nodes))  
    Q = QAOA(p, H)  
    Q.run()
    print(f"Vanilla Overlap: {Q.olap}, Ground State: {format(np.argmin(H), f"0{len(G.nodes)}b")}")
    print(f"Vanilla Sample: {sample_from_state(Q.f_state, n=len(G.nodes))}")
    print("=====================================================")
    S = Simulation(G, p)
    S.run() 
    # for lc in S.light_cones:
    #     print(f"Light Cone Overlap: {lc.overlap(Q.angles)}, Ground state: {format(np.argmin(lc.H), f"0{lc.n_sub}b")}")
    print(f"Light Cone Sample: {S.sample_from_expectations(Q.angles)}")
    print("=====================================================")
    G_new = G.copy()
    R = RQAOA(G_new, p)
    constraints = R.run(initial_angles=Q.angles)
    print(f"RQAOA Sample: {R.compute_bitstring(constraints)}")

