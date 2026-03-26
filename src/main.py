from tests.test_example import *
from tests.test_energy_landscape import run_example_scale_free_graph

if __name__ == "__main__":
    # # hello world
    # run_example_graph()
    
    # show max cut 
    # run_example_max_cut()
    
    # run_example_regular_graph(ising=True)
    
    run_example_scale_free_graph(n_nodes=10, gamma=3)
    # load_pre_computed_energies()

    # optimize_angles_increasing_n_nodes_fixed_gamma(start_n_nodes=30, end_n_nodes=111, gamma=3, multistart=True, n_iter=10, n_graphs=5)
    
    # load_pre_computed_energies(from_opt_angles=True)
