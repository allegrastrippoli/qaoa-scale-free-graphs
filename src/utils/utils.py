import numpy as np
import pandas as pd
import networkx as nx
import itertools
import os 

def graph_info(G, gamma, graphs_info_filename, graph_filename):
    degrees = [G.degree(n) for n in G.nodes()]
    max_ns, max_edge = max_neighborhood_size(G)
    nx.write_gml(G, graph_filename)
    triangles_per_node = nx.triangles(G)
    triangles = sum(triangles_per_node.values()) // 3
    data = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "g" : gamma,    
        "connected": nx.is_connected(G),
        "max_degree": max(degrees),
        "min_degree": min(degrees),
        "avg_degree": np.mean(degrees),
        "max_neighborhood_size": max_ns,
        "triangles": triangles, }
    df = pd.DataFrame([data])
    header = not os.path.exists(graphs_info_filename)
    df.to_csv(graphs_info_filename, mode='a', index=False, header=header)

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

def compute_subgraph_for_edge(G, u, v):
    nodes =  set(G.neighbors(u)) | set(G.neighbors(v))
    sorted_nodes = sorted(nodes)
    mapping = {old: new for new, old in enumerate(sorted_nodes)}
    v_sub = nx.Graph()
    v_sub.add_nodes_from(sorted_nodes)
    v_sub.add_edge(u, v)
    for n in G.neighbors(u):
        v_sub.add_edge(u, n)
    for n in G.neighbors(v):
        v_sub.add_edge(v, n)
    v_sub = nx.relabel_nodes(v_sub, mapping)
    return mapping, v_sub, mapping[u], mapping[v]

def neighborhood_size(G: nx.Graph, edge: tuple) -> int:
    a, b = edge
    degree_a = G.degree(a)
    degree_b = G.degree(b)
    neighbors_a = set(G.neighbors(a))
    neighbors_b = set(G.neighbors(b))
    common_neighbors = len(neighbors_a & neighbors_b)
    return degree_a + degree_b - common_neighbors

def max_neighborhood_size(G: nx.Graph) -> tuple[int, tuple]:
    if G.number_of_edges() == 0:
        return 0, None
    max_size = 0
    max_edge = None
    for edge in G.edges():
        size = neighborhood_size(G, edge)
        if size > max_size:
            max_size = size
            max_edge = edge
    return max_size, max_edge

def maxcut_value(G, bitstring):
    cut = 0
    for i, j in G.edges:
        if bitstring[i] != bitstring[j]:
            cut += 1
    return cut

def brute_force_maxcut(G):
    n = len(G.nodes)
    best_value = -1
    best_bitstring = None
    for bits in itertools.product([0,1], repeat=n):
        value = maxcut_value(G, bits)
        if value > best_value:
            best_value = value
            best_bitstring = bits
    return best_value, best_bitstring
