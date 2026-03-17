from algorithms.algofactory import AlgorithmFactory
from utils.plots import *
from utils.utils import *
import networkx as nx
import numpy as np
import itertools
import random 

#  python -m pytest .

def test_rqaoa_matches_maxcut():
    p = 1 
    G = nx.Graph()
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)])
    algo = AlgorithmFactory.create("qaoa", G, p)
    # algo = AlgorithmFactory.create("rqaoa", G, p)
    # algo = AlgorithmFactory.create("lcqaoa", G, p)
    algo.run()
    bitlist = [int(b) for b in algo.best_bitstring]
    value = maxcut_value(G, bitlist)
    exact_value, exact_bitstring = brute_force_maxcut(G)
    assert value == exact_value
