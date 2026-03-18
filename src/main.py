from tests.run_experiments import *

if __name__ == "__main__":
    # # hello world
    # run_example_graph()
    
    # show max cut 
    # run_example_max_cut()

    # # test energy landscape for small 3 and 4-regular 
    # run_energy_landscape_regular_graph(ising=True)
    # run_energy_landscape_regular_graph(ising=False)
    
    # # test scale free graph
    # run_example_scale_free_graph(80, 4, 5)
    
    # # test optimized angles
    run_optimized_angles(num_nodes=20, gamma=3, multistart=True, n_iter=10, n_graphs=10)
    # compare_optimized_angles_with_energy_landscape()

