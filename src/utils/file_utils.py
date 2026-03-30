from utils.utils import *
import networkx as nx
import numpy as np
import pandas as pd

def graph_info(G, graphs_info_filename, graph_filename):
    degrees = [G.degree(n) for n in G.nodes()]
    max_ns, max_edge = max_neighborhood_size(G)
    nx.write_gml(G, graph_filename)
    data = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "connected": nx.is_connected(G),
        "max_degree": max(degrees),
        "min_degree": min(degrees),
        "avg_degree": np.mean(degrees),
        "max_neighborhood_size": max_ns,
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
    else:
        raise TypeError("not implemented yet")
