from tests.run_experiments import *

if __name__ == "__main__":
    # # hello world
    # run_example_graph()
    
    # show max cut 
    # run_example_max_cut()
    
    optimize_angles_increasing_n_nodes_fixed_gamma(start_n_nodes=30, end_n_nodes=111, gamma=3, multistart=True, n_iter=10, n_graphs=5)
    
