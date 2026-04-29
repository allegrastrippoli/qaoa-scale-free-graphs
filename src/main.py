from tests.test_example import *
from tests.test_energy_landscape import compute_energy_landscape
from tests.test_optimized_angles import optimize_angles_increasing_n_nodes_fixed_gamma
import networkx as nx
import numpy as np


if __name__ == "__main__":
    # run_example_graph()
    # run_example_max_cut()
    # run_example_regular_graph()
    # test_energy_landscape()
    
    run_name = "optimize_angles_increasing_n_nodes_fixed_gamma"
    optimize_angles_increasing_n_nodes_fixed_gamma(run_name=run_name, start_n_nodes=800, end_n_nodes=1000, gamma=3, multistart_iter=100, n_graphs=10, algo="aqaoa")
    


