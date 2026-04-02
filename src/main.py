from tests.test_example import *
from tests.test_energy_landscape import compute_energy_landscape
from tests.test_optimized_angles import optimize_angles_increasing_n_nodes_fixed_gamma
import networkx as nx
import numpy as np

if __name__ == "__main__":
    # run_example_graph()
    # run_example_max_cut()
    # run_example_regular_graph()
    
    # G = nx.Graph()
    # G.add_nodes_from(range(5))
    # G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)])
    # run_name="test_energy_landscape"
    # compute_energy_landscape(run_name=run_name, G=G, algo="lcqaoa", gamma_start=0, gamma_end=np.pi, beta_start=0, beta_end=np.pi/2)
    # compute_energy_landscape(run_name=run_name,G=G, algo="aqaoa", index=2)
    
    run_name = "optimize_angles_increasing_n_nodes_fixed_gamma"
    optimize_angles_increasing_n_nodes_fixed_gamma(run_name=run_name, start_n_nodes=800, end_n_nodes=1000, gamma=3, multistart_iter=100, n_graphs=10, algo="aqaoa")
    
    # load_pre_computed_energies()
    # optimize_angles_increasing_n_nodes_fixed_gamma(start_n_nodes=30, end_n_nodes=111, gamma=3, multistart=True, n_iter=10, n_graphs=5)
    # load_pre_computed_energies(from_opt_angles=True)


