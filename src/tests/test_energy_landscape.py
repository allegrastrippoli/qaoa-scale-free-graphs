from experiments.energylandscape import EnergyLandscape
from algorithms.algofactory import AlgorithmFactory
from utils.plots import plot_full_graph, plot_top_n_subgraphs, plot_energy_landscape
from utils.generate import create_graph, generate_bounded_scale_free_graph
from utils.utils import top_n_max_neighborhood_size, get_colors
from paths import *

# 1. generate a scale free graph
# 2. plot the degree distribution
# 4. plot the graph
# 3. plot the energy landscape ONLY for the nodes with the highest degree 
def run_example_scale_free_graph(n_nodes, gamma, top_n=0):
    p = 1
    run_name="test_example_scale_free_graph"
    rp = RunPaths(run_name)
    G = create_graph(rp=rp, fun=generate_bounded_scale_free_graph, n_nodes=n_nodes, gamma=gamma)
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
        light_cones = AlgorithmFactory.create(name="lcqaoa", G=G, p=p)
        el.compute(fun=light_cones.expectation)
        el.save(filename=rp.log(category=Category.ENERGY_LANDSCAPE))
        gammas, betas, energies2d = el.grid()
        plot_energy_landscape(gammas=gammas, betas=betas, E=energies2d, save_fig=True, filename=rp.fig(category=Category.ENERGY_LANDSCAPE))
