import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import itertools


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


def top_n_max_neighborhood_size(G: nx.Graph, n: int):
    if G.number_of_edges() == 0:
        return 0, None
    sizes = []
    for edge in G.edges():
        sizes.append((edge, neighborhood_size(G, edge)))
    top_n_sizes = sorted(sizes, key=lambda x:x[1])[-n:]
    return [edge for edge, _ in top_n_sizes]


def edge_neighborhood_subgraph(G: nx.Graph, edge: tuple) -> nx.Graph:
    u, v = edge
    nodes = {u, v}
    nodes.update(G.neighbors(u))
    nodes.update(G.neighbors(v))
    return G.subgraph(nodes).copy()


def get_colors(G, top_n, top_n_edges):   
    cmap = plt.get_cmap("tab10", top_n)
    edge_color_map = {edge: cmap(i) for i, edge in enumerate(top_n_edges)}
    edge_colors = []
    for edge in G.edges():
        if edge in edge_color_map:
            edge_colors.append(edge_color_map[edge])
        elif (edge[1], edge[0]) in edge_color_map:
            edge_colors.append(edge_color_map[(edge[1], edge[0])])
        else:
            edge_colors.append("black")
    node_color_map = {node: "lavender" for node in G.nodes()}
    for (u, v), color in edge_color_map.items():
        node_color_map[u] = color
        node_color_map[v] = color
    node_colors = [node_color_map[node] for node in G.nodes()]
    return edge_color_map, edge_colors, node_color_map, node_colors
    
    
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
