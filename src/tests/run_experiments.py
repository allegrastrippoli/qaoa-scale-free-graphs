from utils.utils import *
from utils.plots import  *
from utils.generate import *
from utils.file_utils import *
from algorithms.lcqaoa import *
import networkx as nx
import numpy as np

    
def test_energy_landscape_regular_graph(ising):
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
    for i, G in enumerate(graphs):
        L = LightCone(G, 0, 1, p, ising=ising)
        name_str = ""
        if ising:
            name_str = f"ising_{i}"
        else:
            name_str = f"standard_{i}"
        # plot_energy_landscape(L.expectation, save_fig=True, index=f"light_cone_{name_str}")
        energy_to_csv(L.expectation)
        plot_energy_from_csv(filename="./utils/csv/energy_landscape.csv", save_fig=True, index=f"light_cone_{name_str}")

def test_scale_free_graph():
    top_n = 5
    num_nodes = 50
    gamma = 2.4
    p = 1
    G = generate_bounded_scale_free_graph(num_nodes, gamma)
    degrees = [G.degree(n) for n in G.nodes()]
    max_ns, max_edge = max_neighborhood_size(G)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, Connected: {nx.is_connected(G)}, Max degree: {max(degrees)}, Min degree: {min(degrees)}, Avg degree: {np.mean(degrees):.2f}, Max neighborhood size: {max_ns}")
    fig = plot_degree_distribution(G, gamma)
    lc = LCQAOA(G, p)
    plot_subgraphs_maxns(G, lc, top_n)
