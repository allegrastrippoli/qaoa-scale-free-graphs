from algorithms.algofactory import AlgorithmFactory
from utils.plots import  *
from utils.generate import *
from utils.file_utils import *
from utils.utils import *
from algorithms.lcqaoa import *
import networkx as nx
from paths import *

def run_example_graph():
    p = 1 
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)])
    q = AlgorithmFactory.create("qaoa", G, p)
    q.run()
    print(f"{q.best_bitstring=}\n",
          f"{q.angles=}\n", 
          f"{q.olap=}\n")
    print("---------------------------------------------------------")
    rq = AlgorithmFactory.create("rqaoa", G, p)
    rq.run()
    print(f"{rq.best_bitstring=}\n",
          f"{rq.mapping=}\n", 
          f"{rq.constraints=}\n", 
          f"{rq.history=}\n")
    print("---------------------------------------------------------")
    lc = AlgorithmFactory.create("lcqaoa", G, p)
    lc.run()
    print(f"{lc.best_bitstring=}\n",
          f"{lc.history=}\n")
    
def run_energy_landscape_regular_graph(ising):
    p = 1
    graphs = []
    G1 = nx.Graph()
    G1.add_nodes_from(range(4))
    G1.add_edges_from([(0,1),(0,2),(0,3),(1,3),(1,2)])
    G2 = nx.Graph()
    G2.add_nodes_from(range(8))
    G2.add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,5),(1,6),(1,7)])
    G3 = nx.Graph()
    G3.add_nodes_from(range(5))
    G3.add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4)])
    graphs.append(G1)
    graphs.append(G2)
    graphs.append(G3)
    run_name="run_energy_landscape_regular_graph"
    for i, G in enumerate(graphs):
        L = LightCone(G, 0, 1, p, ising=ising)
        energy_to_csv(L.expectation, filename=csv_energy_landscape_path(run_name=run_name, index=i))
        gammas, betas, E = load_energy_from_csv(filename=csv_energy_landscape_path(run_name=run_name, index=i))
        plot_energy_landscape(gammas, betas, E, filename=fig_energy_landscape_path(run_name=run_name, index=i), save_fig=True)

def run_example_scale_free_graph(num_nodes, gamma, top_n):
    p = 1
    G = generate_bounded_scale_free_graph(num_nodes, gamma)
    run_name="test_example_scale_free_graph"
    graph_info(G, graphs_info_filename=csv_graphs_info_path(run_name=run_name, index=0) , graph_filename=graphs_path(run_name, 0))
    plot_degree_distribution(G, gamma, filename=fig_degree_distribution_path(run_name=run_name, index=0))
    light_cones= LCQAOA(G, p, ising=False)
    top_n_edges = top_n_max_neighborhood_size(G, top_n)
    energies = {}
    for i, lc in enumerate(light_cones.light_cones):
        if (lc.u, lc.v) in top_n_edges or (lc.v, lc.u) in top_n_edges:
            energy_to_csv(lc.expectation, filename=csv_energy_landscape_path(run_name=run_name, index=i))
            gammas, betas, E = load_energy_from_csv(filename=csv_energy_landscape_path(run_name=run_name, index=i))
            energies[(lc.u, lc.v)] = [gammas, betas, E]
    plot_top_n_subgraphs(G, energies, top_n, top_n_edges, filename=fig_energy_landscape_path(run_name=run_name, index=0))
    
    
def run_optimized_angles(num_nodes, gamma):
    p = 1 
    graphs = []
    run_name = "test_optimized_angles"
    for i in range(50):
        G = generate_bounded_scale_free_graph(num_nodes, gamma)
        plot_degree_distribution(G, gamma, filename=fig_degree_distribution_path(run_name=run_name, index=i))
        graphs.append(G)
        graph_info(G, graphs_info_filename=csv_graphs_info_path(run_name=run_name, index=0) , graph_filename=graphs_path(run_name, i))
    for j in range(10):
        angles = initialize_angles(p)
        optimized_angles_to_csv("lcqaoa", graphs, p,  csv_optimized_angles_path(run_name=run_name, index=j), csv_history_path(run_name=run_name, index=j), angles)
        # optimized_angles_to_csv("qaoa", graphs, p,  csv_optimized_angles_path(run_name=run_name, index=j), csv_history_path(run_name=run_name, index=j), angles)
        gammas, betas = load_optimized_angles(csv_optimized_angles_path(run_name=run_name, index=j))
        plot_optimized_angles(gammas, betas, fig_optimized_angles_path(run_name=run_name, index=j))
        print("Finished optimization round")
          
def compare_optimized_angles_with_energy_landscape():
    p = 1
    run_name = "test_optimized_angles"
    _, _, graphs_dir = get_run_dirs(run_name)
    graphs = load_generated_graphs(graphs_dir)
    for i, G in enumerate(graphs):
        if i <= 10:
            light_cones = AlgorithmFactory.create("lcqaoa", G, p)
            energy_to_csv(light_cones.expectation, filename=csv_energy_landscape_path(run_name=run_name, index=i))
            # Q = AlgorithmFactory.create("qaoa", G, p)
            # energy_to_csv(Q.expectation, filename=csv_energy_landscape_path(run_name=run_name, index=i))
            gammas, betas, E = load_energy_from_csv(filename=csv_energy_landscape_path(run_name=run_name, index=i))
            plot_energy_landscape(gammas, betas, E, filename=fig_energy_landscape_path(run_name=run_name, index=i), save_fig=True)
            print("Processed graph ", i)
                
    
            
  