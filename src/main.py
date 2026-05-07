from tests.test_example import *
from tests.test_energy_landscape import compute_energy_landscape
from tests.test_optimized_angles import optimize_angles_increasing_n_nodes_fixed_gamma, star_components_in_scale_free_graphs
import networkx as nx
import numpy as np


if __name__ == "__main__":
    # test_qaoa()
    # test_lcqaoa()
    # run_example_regular_graph()
    # test_energy_landscape()
    
    run_name = "optimize_angles_increasing_n_nodes_fixed_gamma"
    optimize_angles_increasing_n_nodes_fixed_gamma(run_name=run_name, start_n=100, end_n=200, iter=100, n_graphs=10, m=3)
    
    # run_name = "star_components_in_scale_free_graphs"
    # star_components_in_scale_free_graphs(run_name=run_name, start_n_nodes=100, end_n_nodes=101, m=3, iterations=100, n_graphs=1, algo_name="aqaoa",step=50)
    
    # What percentage of the total scale free graph cost depends on star graph contribution?
    # Generate a scale free graph and compute total cost
    # Find all star components and compute total cost
    # Compute the ratio 
    


