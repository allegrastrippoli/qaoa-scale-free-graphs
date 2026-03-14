from algorithms.algofactory import AlgorithmFactory
from algorithms.lcqaoa import LCQAOA
from utils.generate import *
from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd
import csv 

def light_cones_to_csv(G, light_cones, top_n):
    top_n_edges = top_n_max_neighborhood_size(G, top_n)
    for i, lc in enumerate(light_cones.light_cones):
        if (lc.u, lc.v) in top_n_edges or (lc.v, lc.u) in top_n_edges:
            energy_to_csv(lc.expectation, index=i)

# Given one graph, computes the energy for each possible value of gamma and beta 
# Args: fun -> a function that computes the expectation value of an hamiltonian
def energy_to_csv(fun, filename):
    n_points = 100
    gammas = np.linspace(0, 2*np.pi, n_points)
    betas = np.linspace(0, np.pi/2, n_points)
    data = []
    for gamma in gammas:
        for beta in betas:
            data.append((gamma, beta, fun([gamma, beta])))
    df = pd.DataFrame(data, columns=["gamma", "beta", "energy"])
    df.to_csv(filename, index=False)

def load_energy_from_csv(filename):
    df = pd.read_csv(filename)
    gammas = np.sort(df["gamma"].unique())
    betas = np.sort(df["beta"].unique())
    n_gamma = len(gammas)
    n_beta = len(betas)
    E = df["energy"].values.reshape(n_gamma, n_beta)
    return gammas, betas, E
    
# Given a set of graphs, stores the optimal values of gamma and beta 
# Args: algo_name -> a string among "qaoa", "rqaoa" and "lcqaoa"
def optimized_angles_to_csv(algo_name, graphs, p, filename):
    data = []
    for i, G in enumerate(graphs):
        algo = AlgorithmFactory.create(algo_name, G, p)
        algo.run()
        gamma, beta = algo.angles 
        data.append(i, gamma, beta)
    df = pd.DataFrame(data, columns=["graph_id", "gamma", "beta"])
    df.to_csv(filename, index=False)

# Input: a directory
# Output: two numpy arrays storing optimized gammas and betas 
def load_optimized_angles(filename):
    df = pd.read_csv(filename)
    gammas = df["gamma"].to_numpy()
    betas = df["beta"].to_numpy()
    return gammas, betas

# Input: a directory
# Output: a list of nx.Graph graphs  
def load_generated_graphs(filename):
    graphs = []
    for gml_file in Path(filename).glob("*.gml"):
        G = nx.read_gml(gml_file)
        graphs.append(G)
    return graphs

# Given a number n, it creates n bounded scale free graphs with fixed number of nodes and gamma 
def generate_graphs(n, num_nodes, gamma, graphs_info_filename, graph_dir):
        # writer.writerow(["id", "nodes", "edges", "connected", "max degree", "min degree", "avg degree", "max neighborhood size", "graph_file"])
        data = []
        for i in range(n):
            G = generate_bounded_scale_free_graph(num_nodes, gamma)
            degrees = [G.degree(n) for n in G.nodes()]
            max_ns, max_edge = max_neighborhood_size(G)
            graph_path = f"{graph_dir}/graph_{i}.gml"
            nx.write_gml(G, graph_path)
            data.append([i, G.number_of_nodes(), G.number_of_edges(), nx.is_connected(G), max(degrees), min(degrees), f"{np.mean(degrees):.2f}", max_ns, graph_path])
        df = pd.DataFrame(data, columns=["graph_id", "nodes", "edges", "connected", "max_degree", "min_degree", "avg_degree", "max_neighborhood size", "graph_file"])
        df.to_csv(graphs_info_filename, index=False)
