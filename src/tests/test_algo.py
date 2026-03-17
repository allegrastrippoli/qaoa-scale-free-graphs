from algorithms.algofactory import AlgorithmFactory
import networkx as nx
import numpy as np
import itertools
import random 

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

def test_rqaoa_matches_maxcut():
    p = 1 
    G = nx.Graph()
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0)])
    algo = AlgorithmFactory.create("qaoa", G, p)
    # algo = AlgorithmFactory.create("rqaoa", G, p)
    # algo = AlgorithmFactory.create("lcqaoa", G, p)
    algo.run()
    bitlist = [int(b) for b in algo.best_bitstring]
    value = maxcut_value(G, bitlist)
    exact_value, _ = brute_force_maxcut(G)
    assert value == exact_value
