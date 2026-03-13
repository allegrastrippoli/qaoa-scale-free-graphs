from utils.hamiltonians import graph_to_hamiltonian
from tests.test_qmodels import *
from qmodels.lightcones import LightCone
from qmodels.rqaoa import RQAOA
from qmodels.qaoa import QAOA

if __name__ == "__main__":
    pass
    # test_energy_landscape_regular_graph(True)
    # test_energy_landscape_regular_graph(False)
    # test_scale_free_graph()
    
    # generate_graphs(20, 50, 2.4)
    # p = 1
    # graphs = load_all_graphs()
    # compute_optimized_angles(graphs, p)
    # gammas, betas = load_gamma_beta()
    # plot_optimized_angles(gammas, betas)
    
    p = 1 
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0)])
    H = graph_to_hamiltonian(nx.to_numpy_array(G), len(G.nodes))
    Q = QAOA(p, H)
    rq = RQAOA(p, H, Q, G)
    constraints = rq.run()
    print(constraints)
    print(rq.sample(constraints))
    S = Simulation(G, p)
    S.run()
        

