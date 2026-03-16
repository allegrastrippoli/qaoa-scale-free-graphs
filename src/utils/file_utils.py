from algorithms.algofactory import AlgorithmFactory
from pathlib import Path
from utils.utils import *
import networkx as nx
import numpy as np
import pandas as pd

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
def optimized_angles_to_csv(algo_name, graphs, p, filename, history_filename=None, angles=[]):
    data = []
    for i, G in enumerate(graphs):
        algo = AlgorithmFactory.create(algo_name, G, p)
        if not angles:
            algo.run()
        else:
            algo.run(angles)
        gamma, beta = algo.angles     
        data.append([i, gamma, beta])
        if hasattr(algo, "history") and algo.history:
            history_to_csv(algo_name, algo.best_bitstring, algo.history, history_filename)
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
def load_generated_graphs(filepath):
    graphs = []
    for gml_file in Path(filepath).glob("*.gml"):
        G = nx.read_gml(gml_file)
        graphs.append(G)
    return graphs


def graph_info(G, graphs_info_filename, graph_filename):
    degrees = [G.degree(n) for n in G.nodes()]
    # max_ns, max_edge = max_neighborhood_size(G)
    nx.write_gml(G, graph_filename)
    data = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "connected": nx.is_connected(G),
        "max_degree": max(degrees),
        "min_degree": min(degrees),
        "avg_degree": np.mean(degrees),
        # "max_neighborhood_size": max_ns,
        # "graph_file": graph_filename
    }
    df = pd.DataFrame([data])
    df.to_csv(graphs_info_filename, mode='a', index=False, header=False)

def history_to_csv(algo_name, best_bitstring, history, filename):
    data = []
    data.append({"best_bitstring" : best_bitstring})
    if algo_name == "lcqaoa":
        for row_data in history:
            data.append({
            "edge": row_data["edge"],
            "ground_state": row_data["ground_state"],
            "overlap": row_data["overlap"],
            "angles": row_data["angles"]
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, mode='a', index=False, header=False)
    
