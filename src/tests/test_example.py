from utils.generate import generate_bipartite_ring_network
from utils.utils import brute_force_maxcut, maxcut_value
from optimization.energylandscape import EnergyLandscape
from algorithms.algofactory import AlgorithmFactory
from algorithms.lcqaoa import LightCone
from utils.plots import plot_max_cut, plot_energy_landscape
import networkx as nx
from paths import *

def run_example_graph():
    p = 1 
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)])
    q = AlgorithmFactory.create(name="qaoa", G=G, p=p)
    q.run(multistart_iter=5)
    print(f"{q.best_bitstring=}\n",
          f"{q.angles=}\n", 
          f"{q.olap=}\n")
    print("---------------------------------------------------------")
    rq = AlgorithmFactory.create(name="rqaoa", G=G, p=p)
    rq.run()
    print(f"{rq.best_bitstring=}\n",
          f"{rq.mapping=}\n", 
          f"{rq.constraints=}\n", 
          f"{rq.history=}\n")
    print("---------------------------------------------------------")
    lc = AlgorithmFactory.create(name="lcqaoa", G=G, p=p)
    lc.run(multistart_iter=5)
    print(f"{lc.best_bitstring=}\n",
          f"{lc.history=}\n")
      
def run_example_max_cut():
    p = 1 
    run_name = f"run_example_max_cut"    
    rp = RunPaths(run_name)
    G = generate_bipartite_ring_network(5,1,4)
    exact_value, exact_bitstring = brute_force_maxcut(G)
    exact_bitstring = ''.join(str(b) for b in exact_bitstring)
    plot_max_cut(G=G, best_bitstring=exact_bitstring, filename=rp.fig(category=Category.MAX_CUT, index="_exact_cut"))
    q = AlgorithmFactory.create("qaoa", G, p)
    q.run(multistart_iter=100)
    opt_value = maxcut_value(G, q.best_bitstring)
    ratio = opt_value / exact_value
    print(f"{q.best_bitstring=}\n",
          f"{q.angles=}\n", 
          f"{q.olap=}\n",
          f"{ratio=}")
    plot_max_cut(G=G, best_bitstring=q.best_bitstring, filename=rp.fig(category=Category.MAX_CUT, index="_qaoa"))
    print("---------------------------------------------------------")
    rq = AlgorithmFactory.create("rqaoa", G, p)
    rq.run()
    opt_value = maxcut_value(G, rq.best_bitstring)
    ratio = opt_value / exact_value
    print(f"{rq.best_bitstring=}\n",
          f"{rq.mapping=}\n", 
          f"{rq.constraints=}\n", 
          f"{rq.history=}\n",
          f"{ratio=}")
    plot_max_cut(G=G, best_bitstring=rq.best_bitstring, filename=rp.fig(category=Category.MAX_CUT, index="_rqaoa"))
    print("---------------------------------------------------------")
    lc = AlgorithmFactory.create("lcqaoa", G, p)
    lc.run(multistart_iter=100)
    opt_value = maxcut_value(G, lc.best_bitstring)
    ratio = opt_value / exact_value
    print(f"{lc.best_bitstring=}\n",
          f"{lc.history=}\n",
          f"{ratio=}")
    plot_max_cut(G=G, best_bitstring=lc.best_bitstring, filename=rp.fig(category=Category.MAX_CUT, index="_lcqaoa"))    
    
def run_example_regular_graph(costH):
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
    for i, G in enumerate(graphs):
        L = LightCone(G, 0, 1, p, costH=costH)
        el.compute(fun=L.expectation)
        el.save(filename=rp.log(category=Category.ENERGY_LANDSCAPE , index=i))
        gammas, betas, energies2d = el.grid()
        plot_energy_landscape(gammas=gammas, betas=betas, E=energies2d, save_fig=True, filename=rp.fig(category=Category.ENERGY_LANDSCAPE, index=i))

# def load_pre_computed_energies(from_gml_graph=False, from_opt_angles=False):
#     p = 1
#     run_name = "optimize_angles_increasing_n_nodes_fixed_gamma"
#     rp = RunPaths(run_name)
#     if from_gml_graph:
#         graphs = load_generated_graphs(rp.dirs["graphs"])
#         for i, G in enumerate(graphs):
#             light_cones = AlgorithmFactory.create("lcqaoa", G, p)
#             energy_to_csv(fun=light_cones.expectation, filename=rp.log(category=Category.ENERGY_LANDSCAPE , index=i))
#             gammas, betas, E = load_energy_from_csv(filename=rp.log(category=Category.ENERGY_LANDSCAPE , index=i))
#             plot_energy_landscape(gammas=gammas, betas=betas, E=E, filename=rp.fig(category=Category.ENERGY_LANDSCAPE , index=i), save_fig=True)
#             print("Processed graph ", i)
#     elif from_opt_angles:
#         gammas_lst = []
#         betas_lst = []
#         fig_dir = rp.dirs["fig"]
#         for i in range(5):
#             gammas, betas = load_optimized_angles(filename=rp.log(category=Category.OPTIMIZED_ANGLES, index=i))
#             gammas_lst.append(gammas)
#             betas_lst.append(betas)
#         cluster_zoom_in(betas_lst, gammas_lst, [ 30,  50,  70,  90, 110], fig_dir)
