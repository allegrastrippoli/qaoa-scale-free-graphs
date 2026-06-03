from utils.plots import plot_max_cut, plot_energy_landscape, plot_metrics, plot_degree_distribution
from utils.generate import generate_bipartite_ring_network
from utils.utils import brute_force_maxcut, maxcut_value
from utils.file_utils import graph_info
from optimization.energylandscape import EnergyLandscape
from optimization.optimizedangles import OptimizedAngles
from algorithms.algofactory import AlgorithmFactory
from algorithms.lcqaoa import LightCone
from paths import *
import networkx as nx
import numpy as np

def test_qaoa():
    p = 1 
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)])
    q = AlgorithmFactory.create(algo="qaoa", G=G, p=p)
    q.run(iter=100)
    print("---------------------------------------------------------")
    print(f"Hamiltonian: {q.H}")
    print(f"Ground State: {format(np.argmin(q.H), f"0{len(G.nodes)}b")}")
    print(f"Bitstring: {q.best_bitstring}")
    print(f"Optimal Angles: {q.angles}")
    print(f"Minimum Energy: {q.energy}")
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
    print(f"History: {lc.history}")
       
def test_max_cut():
    p = 1 
    run_name = f"test_max_cut"    
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
    
def test_regular_graphs(**kwargs):
    p = 1
    run_name=f"test_regular_graphs"
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
    for i, G in enumerate(graphs):
        L = LightCone(G, 0, 1, p)
        el.compute(fun=L.expectation, **kwargs)
        el.save(filename=rp.log(category=Category.ENERGY_LANDSCAPE , index=i))
        gammas, betas, energies2d = el.grid()
        plot_energy_landscape(gammas=gammas, betas=betas, E=energies2d, save_fig=True, filename=rp.fig(category=Category.ENERGY_LANDSCAPE, index=i))


def create_graph(rp, fun, n, g, *args, index=None, **kwargs):
    if n <= 0:
        raise ValueError("Number of nodes must be > 0")
    G = fun(n=n, gamma=g, *args, **kwargs)
    graph_info(G=G, graphs_info_filename=rp.log(category=Category.GRAPHS_INFO), graph_filename=rp.graphs(category=Category.GRAPH, index=index))
    plot_degree_distribution(G=G, filename=rp.fig(category=Category.DEGREE_DISTRIBUTION, index=index))
    return G

def generate_dataset(rp, fun, n_nodes_lst, scaling_values, n_graphs, *args, **kwargs):
    graphs = []
    g_values = []
    for n in n_nodes_lst:
        for g in scaling_values:
            for j in range(n_graphs):
                G = create_graph(rp=rp, fun=fun, g=g, n=n, index=f"_{n}_nodes{j}", *args, **kwargs)
                graphs.append(G)
                g_values.append(g)
    return graphs, g_values

# run_name = "test_optimized_angles"
def test_optimized_angles(run_name, start_n, end_n,*args, fun=nx.barabasi_albert_graph, scaling_values=[3], n_iter=100, n_graphs=1, algo_name="aqaoa", p=1, step=50, index=0, **kwargs):
    rp = RunPaths(run_name)
    n_nodes_lst = np.arange(start_n, end_n, step)
    print(f"{n_nodes_lst=}")
    rows = []
    print("Generate Dataset... 👾")
    graphs, g_values = generate_dataset(rp=rp, fun=fun, n_nodes_lst=n_nodes_lst, scaling_values=scaling_values, n_graphs=n_graphs, *args, **kwargs)
    print("Optimization Start... 😙")
    oa = OptimizedAngles()
    for G, g in zip(graphs, g_values):
        row = oa.compute(G=G, algo_name=algo_name, p=p, iter=n_iter, g=g, *args, **kwargs)
        rows.append(row)
    print("Store data... ✅")
    filename = rp.log(category=Category.OPTIMIZED_ANGLES)
    oa.build_dataframe(rows)
    oa.save(filename=filename)
    print("Plot results... 🎨")
    plot_metrics(rp=rp, filename=filename)
    print("Done 🥵")
    G = graphs[0]
    gammas, betas, energies2d = compute_energy_landscape(rp=rp, G=G)
    plot_energy_landscape(gammas=gammas, betas=betas, E=energies2d, oa=oa, filename=rp.fig(category=Category.ENERGY_LANDSCAPE, index=index))
    
def compute_energy_landscape(rp: RunPaths, G: nx.Graph, p=1, index=0, algo="aqaoa", **kwargs):
    el = EnergyLandscape()
    q = AlgorithmFactory.create(algo=algo, G=G, p=p, **kwargs)
    el.compute(fun=q.expectation, **kwargs)
    el.save(filename=rp.log(category=Category.ENERGY_LANDSCAPE, index=index))
    gammas, betas, energies2d = el.grid()
    return gammas, betas, energies2d

def run_example_energy_landscape(**kwargs):
    run_name="test_energy_landscape"
    rp = RunPaths(run_name)
    G = nx.Graph()
    G.add_nodes_from(range(5))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)])
    # compute_energy_landscape(rp=rp, G=G, algo="qaoa", **kwargs)
    gammas, betas, energies2d = compute_energy_landscape(rp=rp,G=G, algo="aqaoa", index=1)
    plot_energy_landscape(gammas=gammas, betas=betas, E=energies2d, filename=rp.fig(category=Category.ENERGY_LANDSCAPE))
    
