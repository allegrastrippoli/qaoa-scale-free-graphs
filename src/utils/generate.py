import networkx as nx
import numpy as np
from paths import *
from utils.utils import *

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

# def generate_scale_free_graph(num_nodes: int, gamma: float) -> nx.Graph:
#     if num_nodes < 2:
#         G = nx.Graph()
#         if num_nodes == 1:
#             G.add_node(0)
#         return G
#     alpha = max(0.3, 4.5 / gamma - 0.5)
#     m = max(1, min(5, int(np.ceil(5 - gamma))))
#     G = nx.Graph()
#     seed_size = min(m + 1, num_nodes)
#     for i in range(seed_size):
#         G.add_node(i)
#     for i in range(seed_size):
#         for j in range(i + 1, seed_size):
#             G.add_edge(i, j)
#     for new_node in range(seed_size, num_nodes):
#         G.add_node(new_node)
#         existing_nodes = list(range(new_node))
#         delta = 0.5
#         weights = np.array([(G.degree(node) + delta) ** alpha for node in existing_nodes])
#         probabilities = weights / weights.sum()
#         num_targets = min(m, len(existing_nodes))
#         targets = np.random.choice(existing_nodes,size=num_targets,replace=False, p=probabilities)
#         for target in targets:
#             G.add_edge(new_node, target)
#     return G

def _sample_power_law_degrees(n, gamma, k_min, k_max, rng):
    """Sample n integer degrees from p(k) proportional to k^(-gamma) on [k_min, k_max]."""
    ks = np.arange(k_min, k_max + 1)
    weights = ks.astype(float) ** (-gamma)
    weights /= weights.sum()
    return rng.choice(ks, size=n, p=weights)


def _enforce_min_degree(G, k_min):
    """Add edges so every node has degree >= k_min.

    Strategy: when multiple nodes are deficient, pair them with each other
    so a single edge fixes two deficits. When no deficient partner is
    available, connect to the lowest-degree non-neighbor in the graph,
    which keeps the added edges away from the hubs.
    """
    while True:
        deficient = sorted(v for v in G.nodes if G.degree(v) < k_min)
        if not deficient:
            break
        u = deficient[0]
        forbidden = set(G.neighbors(u))
        forbidden.add(u)

        # Prefer another deficient non-neighbor.
        partners = [v for v in deficient if v != u and v not in forbidden]
        if not partners:
            # Fall back to the lowest-degree non-neighbor in the whole graph.
            partners = [v for v in G.nodes if v not in forbidden]
            if not partners:
                raise RuntimeError(
                    f"Node {u} is already connected to every other node; "
                    f"cannot reach k_min={k_min}."
                )
        v = min(partners, key=G.degree)
        G.add_edge(u, v)


def generate_scale_free(n, gamma, k_min, k_max=None, seed=None,
                        max_attempts=50, strictlyEnforceMinimumDegree=False):
    if k_min < 1:
        raise ValueError("k_min must be >= 1.")
    if n < k_min + 1:
        raise ValueError("n is too small for the requested k_min.")
    rng = np.random.default_rng(seed)
    if k_max is None:
        k_max = n - 1
    k_max = min(k_max, n - 1)
    seq = None
    for _ in range(max_attempts):
        s = _sample_power_law_degrees(n, gamma, k_min, k_max, rng)
        if s.sum() % 2 == 1:
            i = int(rng.integers(0, n))
            s[i] = s[i] + 1 if s[i] < k_max else s[i] - 1
        if nx.is_graphical(s.tolist()):
            seq = s
            break
    if seq is None:
        raise RuntimeError(
            f"Could not sample a graphical sequence after {max_attempts} attempts. "
            "Try larger n, larger k_min, or a different gamma."
        )
    nx_seed = int(rng.integers(0, 2**31 - 1))
    MG = nx.configuration_model(seq.tolist(), seed=nx_seed)
    G = nx.Graph(MG)                          
    G.remove_edges_from(nx.selfloop_edges(G)) 
    components = [set(c) for c in nx.connected_components(G)]
    if len(components) > 1:
        components.sort(key=len, reverse=True)
        giant = components[0]
        for comp in components[1:]:
            u = min(comp,  key=G.degree)
            v = min(giant, key=G.degree)
            G.add_edge(u, v)
            giant |= comp

    if strictlyEnforceMinimumDegree:
        _enforce_min_degree(G, k_min)
    return G

def generate_3_regular_graph(n, seed=None):
    if n < 4 or n % 2 != 0:
        raise ValueError("n must be an even integer >= 4 for a 3-regular graph.")
    return nx.random_regular_graph(d=3, n=n, seed=seed)
