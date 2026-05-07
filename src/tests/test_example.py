from utils.generate import generate_bipartite_ring_network
from utils.utils import brute_force_maxcut, maxcut_value
from optimization.energylandscape import EnergyLandscape
from algorithms.algofactory import AlgorithmFactory
from algorithms.lcqaoa import LightCone
from utils.plots import plot_max_cut, plot_energy_landscape
from tests.test_energy_landscape import compute_energy_landscape
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from paths import *

def test_qaoa():
    p = 1 
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)])
    q = AlgorithmFactory.create(algo="qaoa", G=G, p=p)
    q.run(iter=100)
    print("---------------------------------------------------------")
    print(f"Hamiltonian: {q.H}")
    print(f"Ground State Energy: {q.min}")
    print(f"Bitstring: {q.best_bitstring}")
    print(f"Optimal Angles: {q.angles}")
    print(f"Minimum Energy: {q.angles}")
    print(f"Overlap (Probability of Smapling the Ground State): {round(q.olap*100, 1)}%")
 
def test_lcqaoa():
    p = 1 
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)])
    lc = AlgorithmFactory.create(algo="lcqaoa", G=G, p=p)
    lc.run(iter=100)
    print(f"Bitstring: {lc.best_bitstring}")
    print(f"Optimal Angles: {lc.angles}")
    print(f"Minimum Energy: {lc.angles}")
    print(f"History: {lc.best_bitstring}")
       
def run_example_max_cut():
    p = 1 
    run_name = f"run_example_max_cut"    
    rp = RunPaths(run_name)
    G = generate_bipartite_ring_network(5,1,4)
    exact_value, exact_bitstring = brute_force_maxcut(G)
    exact_bitstring = ''.join(str(b) for b in exact_bitstring)
    plot_max_cut(G=G, best_bitstring=exact_bitstring, filename=rp.fig(category=Category.MAX_CUT, index="_exact_cut"))
    q = AlgorithmFactory.create(algo="qaoa", G=G, p=p)
    q.run(iter=100)
    opt_value = maxcut_value(G, q.best_bitstring)
    ratio = opt_value / exact_value
    print(f"{q.best_bitstring=}\n",
          f"{q.angles=}\n", 
          f"{q.energy=}\n",
          f"{q.olap=}\n",
          f"{ratio=}")
    plot_max_cut(G=G, best_bitstring=q.best_bitstring, filename=rp.fig(category=Category.MAX_CUT, index="_qaoa"))
    lc = AlgorithmFactory.create(algo="lcqaoa", G=G, p=p)
    lc.run(iter=100)
    opt_value = maxcut_value(G, lc.best_bitstring)
    ratio = opt_value / exact_value
    print(f"{lc.best_bitstring=}\n",
          f"{lc.angles=}\n", 
          f"{lc.energy=}\n",
          f"{lc.history=}\n",
          f"{ratio=}")
    plot_max_cut(G=G, best_bitstring=lc.best_bitstring, filename=rp.fig(category=Category.MAX_CUT, index="_lcqaoa"))    
    
def run_example_regular_graph(**kwargs):
    p = 1
    run_name=f"run_example_regular_graph"
    rp = RunPaths(run_name)
    graphs = []
    G1 = nx.Graph()
    G1.add_nodes_from(range(4))
    G1.add_edges_from([(0,1),(0,2),(0,3),(1,3),(1,2)])
    G2 = nx.Graph()
    G2.add_nodes_from(range(8))
    G2.add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,5),(1,6),(1,7)])
    G3 = nx.Graph()
    G3.add_nodes_from(range(5))
    G3.add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4)])
    graphs.append(G1)
    graphs.append(G2)
    graphs.append(G3)
    el = EnergyLandscape()
    scaling = kwargs.get("scaling", True)
    for i, G in enumerate(graphs):
        L = LightCone(G, 0, 1, p, scaling)
        el.compute(fun=L.expectation, **kwargs)
        el.save(filename=rp.log(category=Category.ENERGY_LANDSCAPE , index=i))
        gammas, betas, energies2d = el.grid()
        plot_energy_landscape(gammas=gammas, betas=betas, E=energies2d, save_fig=True, filename=rp.fig(category=Category.ENERGY_LANDSCAPE, index=i))

