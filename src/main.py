from tests.test_example import *
from tests.test_energy_landscape import run_example_energy_landscape
from utils.generate import generate_scale_free
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
    
    # run_name = "optimize_angles_increasing_n_nodes_fixed_gamma"
    # optimize_angles_increasing_n_nodes(run_name=run_name, start_n=100, end_n=200, m=3)
    
    # scaling_values=[2, 2.25, 2.50, 2.75, 3, 3.25, 3.50, 3.75, 4]

    run_name = "optimize_angles_increasing_n_nodes"
    optimize_angles_increasing_n_nodes(run_name=run_name, start_n=100, end_n=201, step=50, fun=generate_scale_free, scaling_values=[2,3], k_min=5, strictlyEnforceMinimumDegree=True)
