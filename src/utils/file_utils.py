from algorithms.algofactory import AlgorithmFactory
from utils.generate import *
from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd
import csv 

# Given one graph, computes the energy for each possible value of gamma and beta 
# Args: fun -> a function that computes the expectation value of an hamiltonian
def energy_to_csv(fun, filename="./utils/csv/energy_landscape.csv"):
    n_points = 100
    gammas = np.linspace(0, 2*np.pi, n_points)
    betas = np.linspace(0, np.pi/2, n_points)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma", "beta", "energy"])
        for gamma in gammas:
            for beta in betas:
                energy = fun([gamma, beta])
                writer.writerow([gamma, beta, energy])

# Given a set of graphs, stores the optimal values of gamma and beta 
# Args: algo_name -> a string among "qaoa", "rqaoa" and "lcqaoa"
def optimized_angles_to_csv(algo_name, graphs, p, filename="./utils/csv/optimized_angles.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "gamma", "beta"])
        for i, G in enumerate(graphs):
            algo = AlgorithmFactory.create(algo_name, G, p)
            algo.run()
            gamma, beta = algo.angles 
            writer.writerow([i, gamma, beta])

# Input: a directory
# Output: a list of nx.Graph graphs  
def load_generated_graphs(directory="./utils/graphs"):
    graphs = []
    for gml_file in Path(directory).glob("*.gml"):
        G = nx.read_gml(gml_file)
        graphs.append(G)
    return graphs

# Input: a directory
# Output: two numpy arrays storing optimized gammas and betas 
def load_optimized_angles(csv_file="./utils/csv/optimized_angles.csv"):
    df = pd.read_csv(csv_file)
    gammas = df["gamma"].to_numpy()
    betas = df["beta"].to_numpy()
    return gammas, betas

# Given a number n, it creates n bounded scale free graphs with fixed number of nodes and gamma 
def generate_graphs(n, num_nodes, gamma, csv_filename="./utils/csv/graphs_info.csv",  graph_dir="./utils/graphs"):
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "nodes", "edges", "connected", "max degree", "min degree", "avg degree", "max neighborhood size", "graph_file"])
        for i in range(n):
            G = generate_bounded_scale_free_graph(num_nodes, gamma)
            degrees = [G.degree(n) for n in G.nodes()]
            max_ns, max_edge = max_neighborhood_size(G)
            graph_path = f"{graph_dir}/graph_{i}.gml"
            nx.write_gml(G, graph_path)
            writer.writerow([i, G.number_of_nodes(), G.number_of_edges(), nx.is_connected(G), max(degrees), min(degrees), f"{np.mean(degrees):.2f}", max_ns, graph_path])
