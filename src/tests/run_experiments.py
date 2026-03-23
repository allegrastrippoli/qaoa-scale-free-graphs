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
    q.run(n_iter=5, multistart=True)
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
    lc.run(n_iter=5, multistart=True)
    print(f"{lc.best_bitstring=}\n",
          f"{lc.history=}\n")
    
    
def run_example_max_cut():
    p = 1 
    run_name = f"run_example_max_cut"
    G = generate_bipartite_ring_network(5,1,4)
    exact_value, exact_bitstring = brute_force_maxcut(G)
    exact_bitstring = ''.join(str(b) for b in exact_bitstring)
    plot_max_cut(G, exact_bitstring, fig_max_cut(run_name, "_exact_cut"))
    q = AlgorithmFactory.create("qaoa", G, p)
    q.run(n_iter=100, multistart=True)
    opt_value = maxcut_value(G, q.best_bitstring)
    ratio = opt_value / exact_value
    print(f"{q.best_bitstring=}\n",
          f"{q.angles=}\n", 
          f"{q.olap=}\n",
          f"{ratio=}")
    plot_max_cut(G, q.best_bitstring, fig_max_cut(run_name,  "_qaoa"))
    print("---------------------------------------------------------")
    rq = AlgorithmFactory.create("rqaoa", G, p)
    rq.run()
    opt_value = maxcut_value(G, rq.best_bitstring)
    ratio = opt_value / exact_value
    print(f"{rq.best_bitstring=}\n",
          f"{rq.mapping=}\n", 
          f"{rq.constraints=}\n", 
          f"{rq.history=}\n",
          f"{ratio=}")
    plot_max_cut(G, rq.best_bitstring, fig_max_cut(run_name,  "_rqaoa"))
    print("---------------------------------------------------------")
    lc = AlgorithmFactory.create("lcqaoa", G, p)
    lc.run(n_iter=100, multistart=True)
    opt_value = maxcut_value(G, lc.best_bitstring)
    ratio = opt_value / exact_value
    print(f"{lc.best_bitstring=}\n",
          f"{lc.history=}\n",
          f"{ratio=}")
    plot_max_cut(G, lc.best_bitstring, fig_max_cut(run_name,  "_lcqaoa"))
    
    
def run_example_regular_graph(ising):
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
    run_name="run_example_regular_graphs"
    for i, G in enumerate(graphs):
        L = LightCone(G, 0, 1, p, ising=ising)
        energy_to_csv(L.expectation, filename=csv_energy_landscape_path(run_name=run_name, index=i))
        gammas, betas, E = load_energy_from_csv(filename=csv_energy_landscape_path(run_name=run_name, index=i))
        plot_energy_landscape(gammas, betas, E, filename=fig_energy_landscape_path(run_name=run_name, index=i), save_fig=True)


def create_graph(run_name, fun, n_nodes, gamma, index=0):
    G = fun(n_nodes, gamma)
    graph_info(G, graphs_info_filename=csv_graphs_info_path(run_name=run_name, index=index), graph_filename=graphs_path(run_name=run_name, index=index))
    plot_degree_distribution(G, gamma, filename=fig_degree_distribution_path(run_name=run_name, index=index))
    return G

# 1. generate a scale free graph
# 2. plot the degree distribution
# 4. plot the graph
# 3. plot the energy landscape ONLY for the nodes with the highest degree 
def run_example_scale_free_graph(n_nodes, gamma, top_n):
    p = 1
    G = create_graph(run_name, generate_scale_free_graph, n_nodes, gamma)
    run_name="test_example_scale_free_graph"
    top_n_edges = top_n_max_neighborhood_size(G, top_n)
    light_cones= LCQAOA(G, p, ising=False, edges_subset=top_n_edges)
    energies = {}
    for i, lc in enumerate(light_cones.light_cones):
        energy_to_csv(lc.expectation, filename=csv_energy_landscape_path(run_name=run_name, index=i))
        gammas, betas, E = load_energy_from_csv(filename=csv_energy_landscape_path(run_name=run_name, index=i))
        energies[(lc.u, lc.v)] = [gammas, betas, E]
    edge_color_map, edge_colors, _, node_colors = get_colors(G, top_n, top_n_edges)
    plot_full_graph(G, node_colors, edge_colors, fig_full_graph(run_name=run_name, index=0))
    plot_top_n_subgraphs(G, energies, edge_color_map=edge_color_map, filename=fig_energy_landscape_path(run_name=run_name, index=0))
    
    
def optimize_angles_fixed_n_nodes_fixed_gamma(n_nodes, gamma, multistart, n_iter, n_graphs):
    p = 1 
    graphs = []
    run_name = "optimize_angles_fixed_n_nodes_fixed_gamma"
    for i in range(n_graphs):
        G = create_graph(run_name=run_name, fun=generate_bounded_scale_free_graph, n_nodes=n_nodes, gamma=gamma, index=i)
        graphs.append(G)
    optimized_angles_to_csv("lcqaoa", graphs, p,  csv_optimized_angles_path(run_name=run_name, index=0), csv_history_path(run_name=run_name, index=0),  n_iter=n_iter, multistart=multistart)
    gammas, betas = load_optimized_angles(csv_optimized_angles_path(run_name=run_name, index=0))
    plot_optimized_angles(betas, gammas, fig_optimized_angles_path(run_name=run_name, index=0))
    

def optimize_angles_increasing_n_nodes_fixed_gamma(start_n_nodes, end_n_nodes, gamma, multistart, n_iter, n_graphs):
    p = 1 
    run_name = "optimize_angles_increasing_n_nodes_fixed_gamma"
    n_nodes_lst = np.arange(start_n_nodes, end_n_nodes, 20)
    print(n_nodes_lst)
    gammas_lst = []
    betas_lst = []
    n_colors = len(n_nodes_lst)
    for j, n_node in enumerate(n_nodes_lst):
        graphs = []
        for i in range(n_graphs):
            G = create_graph(run_name=run_name, fun=generate_bounded_scale_free_graph, n_nodes=n_node, gamma=gamma, index=i)
            graphs.append(G)
        optimized_angles_to_csv("lcqaoa", graphs, p,  csv_optimized_angles_path(run_name=run_name, index=j), csv_history_path(run_name=run_name, index=j),  n_iter=n_iter, multistart=multistart)
        gammas, betas = load_optimized_angles(csv_optimized_angles_path(run_name=run_name, index=j))
        gammas_lst.append(gammas)
        betas_lst.append(betas)
    plot_optimized_angles_fixed_clusters(betas_lst, gammas_lst, n_colors, fig_optimized_angles_path(run_name=run_name, index=0))
        
    
def compare_optimized_angles_with_energy_landscape():
    p = 1
    run_name = " "
    _, _, graphs_dir = get_run_dirs(run_name)
    graphs = load_generated_graphs(graphs_dir)
    for i, G in enumerate(graphs):
        light_cones = AlgorithmFactory.create("lcqaoa", G, p)
        energy_to_csv(light_cones.expectation, filename=csv_energy_landscape_path(run_name=run_name, index=i))
        gammas, betas, E = load_energy_from_csv(filename=csv_energy_landscape_path(run_name=run_name, index=i))
        plot_energy_landscape(gammas, betas, E, filename=fig_energy_landscape_path(run_name=run_name, index=i), save_fig=True)
        print("Processed graph ", i)


def plot_dataset_graphs():
    run_name = " "
    _, _, graphs_dir = get_run_dirs(run_name)
    graphs = load_generated_graphs(graphs_dir)
    top_n = 3
    for i, G in enumerate(graphs):
        top_n_edges = top_n_max_neighborhood_size(G, top_n)
        _, edge_colors, _, node_colors = get_colors(G, top_n, top_n_edges)
        plot_full_graph(G, fig_full_graph(run_name=run_name, index=i), node_colors=node_colors, edge_colors=edge_colors)
