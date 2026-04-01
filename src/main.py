from tests.test_example import *
from tests.test_energy_landscape import compute_energy_landscape
import networkx as nx
import numpy as np

if __name__ == "__main__":
    # run_example_graph()
    # run_example_max_cut()
    # run_example_regular_graph()
    
    G = nx.Graph()
    G.add_nodes_from(range(3))
    # G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)])
    G.add_edges_from([(0,1),(1,2),(2,0)])
    compute_energy_landscape(G=G, algo="lcqaoa", gamma_start=0, gamma_end=np.pi, beta_start=0, beta_end=np.pi/2)
    compute_energy_landscape(G=G, algo="aqaoa", index=1)
    
    # load_pre_computed_energies()
    # optimize_angles_increasing_n_nodes_fixed_gamma(start_n_nodes=30, end_n_nodes=111, gamma=3, multistart=True, n_iter=10, n_graphs=5)
    # load_pre_computed_energies(from_opt_angles=True)


