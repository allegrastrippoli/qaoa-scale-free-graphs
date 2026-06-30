from plots import plot_max_cut 
from utils import generate_bipartite_ring_network, brute_force_maxcut, maxcut_value
from algorithms.algofactory import AlgorithmFactory
from algorithms.lcqaoa import LightCone
from classic_maxcut import maxcut_gurobi
from paths import *
import networkx as nx
import numpy as np
import time

def test_rqaoa():
    print("TO DO: Not implemented yet")
    pass

def test_qaoa():
    p = 1 
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)])
    q = AlgorithmFactory.create(algo="qaoa", G=G, p=p)
    q.run(n_iter=100)
    print("---------------------------------------------------------")
    print(f"Hamiltonian: {q.H}")
    print(f"Ground State: {format(np.argmin(q.H), f"0{len(G.nodes)}b")}")
    print(f"Bitstring: {q.best_bitstring}")
    print(f"Optimal Angles: {q.angles}")
    print(f"Minimum Energy: {q.energy}")
    print(f"Overlap (Probability of Smapling the Ground State): {round(q.olap*100, 1)}%")
 
def test_aqaoa():
    p = 1 
    G = generate_bipartite_ring_network(5, 1, 2)
    # plot_full_graph(G)
    a = AlgorithmFactory.create(algo="aqaoa", G=G, p=p)
    a.run(n_iter=100)
    print("---------------------------------------------------------")
    print(f"AQAOA Minimum Energy: {a.energy}")
    print(f"AQAOA Optimal Angles: {a.angles}")
    print("---------------------------------------------------------")
    maxcut_gurobi(G) 
    print("---------------------------------------------------------")
    q = AlgorithmFactory.create(algo="qaoa", G=G, p=p)
    q.run(n_iter=100)
    print(f"QAOA Bitstring: {q.best_bitstring}")
    print(f"QAOA Angles: {q.angles}")
    print(f"Ground State: {format(np.argmin(q.H), f"0{len(G.nodes)}b")}")
    print(f"Overlap (Probability of Smapling the Ground State): {round(q.olap*100, 1)}%")
 
def test_lcqaoa():
    p = 1 
    G = nx.Graph()
    G.add_nodes_from(range(4))
    # G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)])
    G.add_edges_from([(0,1),(1,2),(2,3),(3,4)])
    lc = AlgorithmFactory.create(algo="lcqaoa", G=G, p=p)
    lc.run(n_iter=100)
    print(f"Bitstring: {lc.best_bitstring}")
    print(f"Optimal Angles: {lc.angles}")
    print(f"History: {lc.history}")
    
       
def test_max_cut():
    p = 1 
    run_name = f"test_max_cut"    
    rp = RunPaths(run_name)
    G = generate_bipartite_ring_network(5,1,4)
    exact_value, exact_bitstring = brute_force_maxcut(G)
    exact_bitstring = ''.join(str(b) for b in exact_bitstring)
    plot_max_cut(G=G, best_bitstring=exact_bitstring, filename=rp.fig(OutputFile.MAX_CUT, index="_exact_cut"))
    q = AlgorithmFactory.create(algo="qaoa", G=G, p=p)
    q.run(n_iter=100)
    opt_value = maxcut_value(G, q.best_bitstring)
    ratio = opt_value / exact_value
    print(f"{q.best_bitstring=}\n",
          f"{q.angles=}\n", 
          f"{q.energy=}\n",
          f"{q.olap=}\n",
          f"{ratio=}")
    plot_max_cut(G=G, best_bitstring=q.best_bitstring, filename=rp.fig(OutputFile.MAX_CUT, index="_qaoa"))
    lc = AlgorithmFactory.create(algo="lcqaoa", G=G, p=p)
    lc.run(n_iter=100)
    opt_value = maxcut_value(G, lc.best_bitstring)
    ratio = opt_value / exact_value
    print(f"{lc.best_bitstring=}\n",
          f"{lc.angles=}\n", 
          f"{lc.energy=}\n",
          f"{lc.history=}\n",
          f"{ratio=}")
    plot_max_cut(G=G, best_bitstring=lc.best_bitstring, filename=rp.fig(OutputFile.MAX_CUT, index="_lcqaoa"))    
