import numpy as np
import networkx as nx
import itertools
from paths import *

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
        raise ValueError(f"q must satisfy 2*p < q < n, got q={q} with n={n}")
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

def _sample_power_law_degrees(n, alpha, k_min, k_max, rng):
    ks = np.arange(k_min, k_max + 1)
    weights = ks.astype(float) ** (-alpha)
    weights /= weights.sum()
    return rng.choice(ks, size=n, p=weights)

def _enforce_min_degree(G, k_min):
    while True:
        deficient = sorted(v for v in G.nodes if G.degree(v) < k_min)
        if not deficient:
            break
        u = deficient[0]
        forbidden = set(G.neighbors(u))
        forbidden.add(u)
        partners = [v for v in deficient if v != u and v not in forbidden]
        if not partners:
            partners = [v for v in G.nodes if v not in forbidden]
            if not partners:
                raise RuntimeError(f"Node {u} is already connected to every other node; "
                                   f"cannot reach k_min={k_min}.")
        v = min(partners, key=G.degree)
        G.add_edge(u, v)

def generate_scale_free(n, alpha, k_min, k_max=None, seed=None,
                        max_attempts=50, strictlyEnforceMinimumDegree=True):
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
        s = _sample_power_law_degrees(n, alpha, k_min, k_max, rng)
        if s.sum() % 2 == 1:
            i = int(rng.integers(0, n))
            s[i] = s[i] + 1 if s[i] < k_max else s[i] - 1
        if nx.is_graphical(s.tolist()):
            seq = s
            break
    if seq is None:
        raise RuntimeError(
            f"Could not sample a graphical sequence after {max_attempts} attempts. "
            "Try larger n, larger k_min, or a different alpha."
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
