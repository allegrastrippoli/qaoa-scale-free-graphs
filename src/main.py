from tests.test_example import *
from tests.test_energy_landscape import run_example_energy_landscape
from utils.generate import generate_scale_free_graph
from tests.test_optimized_angles import optimize_angles_increasing_n_nodes
import networkx as nx
import numpy as np
import json

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    # test_qaoa()
    # test_lcqaoa()
    # run_example_regular_graph(**config)
    # run_example_energy_landscape(**config)
    
    run_name = "optimize_angles_increasing_n_nodes_fixed_gamma"
    optimize_angles_increasing_n_nodes(run_name=run_name, start_n=100, end_n=200, m=3)
    
    # What percentage of the total scale free graph cost depends on star graph contribution?
    # Generate a scale free graph and compute total cost
    # Find all star components and compute total cost
    # Compute the ratio 
    # run_name = "star_components_in_scale_free_graphs"
    # star_components_in_scale_free_graphs(run_name=run_name, start_n_nodes=100, end_n_nodes=101, m=3, iterations=100, n_graphs=1, algo_name="aqaoa",step=50)
       


