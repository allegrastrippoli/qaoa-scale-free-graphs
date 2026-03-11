import networkx as nx
import numpy as np

def relabel_white_black(G: nx.Graph, n: int) -> nx.Graph:
    mapping = {}
    for i in range(n):
        mapping[f"w_{i}"] = i
    for i in range(n):
        mapping[f"b_{i}"] = n + i
    return mapping, nx.relabel_nodes(G, mapping)

def generate_bipartite_ring_network(n: int, p: int, q: int) -> nx.Graph:
    if n < 2:
        raise ValueError(f"n must be > 1, got {n}")
    if p < 1 or p >= n / 2:
        raise ValueError(f"p must satisfy 0 < p < n/2, got p={p} with n={n}")
    if q < 2*p or q >= n:
        raise ValueError(f"q must satisfy 0 < q < n, got q={q} with n={n}")
    G = nx.Graph()
    for i in range(n):
        G.add_node(f"b_{i}", color="black")
        G.add_node(f"w_{i}", color="white")
    for i in range(n):
        for j in range(1, p + 1):
            next_idx = (i + j) % n
            G.add_edge(f"b_{i}", f"b_{next_idx}")
            G.add_edge(f"w_{i}", f"w_{next_idx}")
    for i in range(n):
        for j in range(1, q + 1):
            next_idx = (i + j) % n
            G.add_edge(f"b_{i}", f"w_{next_idx}")
            
    mapping = {old: new for new, old in enumerate(list(G.nodes))}
    G = nx.relabel_nodes(G, mapping)
    return G

def generate_scale_free_graph(num_nodes: int, gamma: float) -> nx.Graph:
    if num_nodes < 2:
        G = nx.Graph()
        if num_nodes == 1:
            G.add_node(0)
        return G
    alpha = max(0.3, 4.5 / gamma - 0.5)
    m = max(1, min(5, int(np.ceil(5 - gamma))))
    G = nx.Graph()
    seed_size = min(m + 1, num_nodes)
    for i in range(seed_size):
        G.add_node(i)
    for i in range(seed_size):
        for j in range(i + 1, seed_size):
            G.add_edge(i, j)
    for new_node in range(seed_size, num_nodes):
        G.add_node(new_node)
        existing_nodes = list(range(new_node))
        delta = 0.5
        weights = np.array([(G.degree(node) + delta) ** alpha for node in existing_nodes])
        probabilities = weights / weights.sum()
        num_targets = min(m, len(existing_nodes))
        targets = np.random.choice(existing_nodes,size=num_targets,replace=False, p=probabilities)
        for target in targets:
            G.add_edge(new_node, target)
    return G

def generate_bounded_scale_free_graph(num_nodes: int, gamma: float, max_node_degree: int = 24) -> nx.Graph:
    if num_nodes < 2:
        G = nx.Graph()
        if num_nodes == 1:
            G.add_node(0)
        return G
    alpha = max(0.3, 4.5 / gamma - 0.5)
    m = max(1, min(5, int(np.ceil(5 - gamma))))
    seed_size = min(m + 1, num_nodes, max_node_degree + 1)
    G = nx.Graph()
    for i in range(seed_size):
        G.add_node(i)
    for i in range(seed_size):
        for j in range(i + 1, seed_size):
            G.add_edge(i, j)
    for new_node in range(seed_size, num_nodes):
        G.add_node(new_node)
        existing_nodes = list(range(new_node))
        delta = 0.5
        weights = np.zeros(len(existing_nodes))
        for idx, node in enumerate(existing_nodes):
            if G.degree(node) < max_node_degree:
                weights[idx] = (G.degree(node) + delta) ** alpha
        total_weight = weights.sum()
        if total_weight == 0:
            valid_targets = [n for n in existing_nodes if G.degree(n) < max_node_degree]
            if not valid_targets:
                break # Cannot add more edges without violating constraints
            probabilities = np.ones(len(valid_targets)) / len(valid_targets)
            target_pool = valid_targets
        else:
            probabilities = weights / total_weight
            target_pool = existing_nodes
        num_targets = min(m, len(np.nonzero(probabilities)[0]) if total_weight > 0 else len(target_pool))
        if num_targets > 0:
            targets = np.random.choice(target_pool, size=num_targets, replace=False, p=probabilities if total_weight > 0 else None)
            for target in targets:
                G.add_edge(new_node, target)
    return G
