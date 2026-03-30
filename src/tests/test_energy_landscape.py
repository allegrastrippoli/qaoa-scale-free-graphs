from optimization.energylandscape import EnergyLandscape
from algorithms.algofactory import AlgorithmFactory
from utils.plots import plot_full_graph, plot_top_n_subgraphs, plot_energy_landscape
from utils.generate import create_graph, generate_bounded_scale_free_graph
from utils.utils import top_n_max_neighborhood_size, get_colors
import networkx as nx
from paths import *

def compute_energy(el: EnergyLandscape, rp: RunPaths, name: str, G: nx.Graph, p: int, index: int):
        q = AlgorithmFactory.create(name=name, G=G, p=p)
        el.compute(fun=q.expectation)
        el.save(filename=rp.log(category=Category.ENERGY_LANDSCAPE, index=index))
        gammas, betas, energies2d = el.grid()
        q.run(multistart_iter=100)
        # points = [(q.angles[0], q.angles[1])]
        # plot_energy_landscape(gammas=gammas, betas=betas, E=energies2d, save_fig=True, filename=rp.fig(category=Category.ENERGY_LANDSCAPE, index=index), points=points)
        plot_energy_landscape(gammas=gammas, betas=betas, E=energies2d, save_fig=True, filename=rp.fig(category=Category.ENERGY_LANDSCAPE, index=index))
        
# 1. generate a graph
# 2. plot the degree distribution
# 4. plot the graph
# 3. if top_n > 0 plot the energy landscape ONLY for the top n nodes with the highest degree 
# 4. else plot the energy landscape for the full graph
def compute_energy_landscape(top_n=0, **kwargs):
    p = 1
    run_name="test_example_scale_free_graph"
    rp = RunPaths(run_name)
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)])
    # G = create_graph(rp=rp, fun=generate_bounded_scale_free_graph, n_nodes=kwargs.get("n_nodes", None), gamma=kwargs.get("gamma", None))
    el = EnergyLandscape()
    if top_n > 0:
        top_n_edges = top_n_max_neighborhood_size(G, top_n)
        light_cones = AlgorithmFactory.create(name="lcqaoa", G=G, p=p, edges_subset=top_n_edges)
        energies = {}
        for i, lc in enumerate(light_cones.light_cones):
            el.compute(fun=lc.expectation)
            el.save(filename=rp.log(category=Category.ENERGY_LANDSCAPE , index=i))
            gammas, betas, energies2d = el.grid()
            energies[(lc.u, lc.v)] = [gammas, betas, energies2d]
        edge_color_map, edge_colors, _, node_colors = get_colors(G, top_n, top_n_edges)
        plot_full_graph(G=G, node_colors=node_colors, edge_colors=edge_colors, filename=rp.fig(category=Category.FULL_GRAPH))
        plot_top_n_subgraphs(G=G, energies=energies, edge_color_map=edge_color_map, filename=rp.fig(category=Category.ENERGY_LANDSCAPE))
    else:
        compute_energy(el=el, rp=rp, name="qaoa", G=G, p=p, index=0 )
        compute_energy(el=el, rp=rp, name="lcqaoa", G=G, p=p, index=1)
        compute_energy(el=el, rp=rp, name="aqaoa", G=G, p=p, index=2)
        


        
 
