from optimization.energylandscape import EnergyLandscape
from algorithms.algofactory import AlgorithmFactory
from utils.plots import plot_full_graph, plot_top_n_subgraphs, plot_energy_landscape
from utils.utils import top_n_max_neighborhood_size, get_colors
import networkx as nx
import numpy as np
from paths import *

def compute_energy_landscape(G: nx.Graph, p=1, index=0, algo="qaoa", gamma_start=0, gamma_end=2*np.pi, beta_start=0, beta_end=np.pi/2, **kwargs):
        p = 1
        run_name="test_energy_landscape"
        rp = RunPaths(run_name)
        el = EnergyLandscape()
        q = AlgorithmFactory.create(algo=algo, G=G, p=p)
        el.compute(fun=q.expectation, gamma_start=gamma_start, gamma_end=gamma_end, beta_start=beta_start, beta_end=beta_end)
        el.save(filename=rp.log(category=Category.ENERGY_LANDSCAPE, index=index))
        gammas, betas, energies2d = el.grid()
        plot_energy_landscape(gammas=gammas, betas=betas, E=energies2d, save_fig=True, filename=rp.fig(category=Category.ENERGY_LANDSCAPE, index=index))

def compute_top_n_subgraph_energy_landscape(G:nx.Graph, top_n=3):
    p = 1
    run_name="test_top_n_subgraph_energy_landscape"
    rp = RunPaths(run_name)
    el = EnergyLandscape()
    if top_n > 0:
        top_n_edges = top_n_max_neighborhood_size(G, top_n)
        light_cones = AlgorithmFactory.create(algo="lcqaoa", G=G, p=p, edges_subset=top_n_edges)
        energies = {}
        for i, lc in enumerate(light_cones.light_cones):
            el.compute(fun=lc.expectation)
            el.save(filename=rp.log(category=Category.ENERGY_LANDSCAPE , index=i))
            gammas, betas, energies2d = el.grid()
            energies[(lc.u, lc.v)] = [gammas, betas, energies2d]
        edge_color_map, edge_colors, _, node_colors = get_colors(G, top_n, top_n_edges)
        plot_full_graph(G=G, node_colors=node_colors, edge_colors=edge_colors, filename=rp.fig(category=Category.FULL_GRAPH))
        plot_top_n_subgraphs(G=G, energies=energies, edge_color_map=edge_color_map, filename=rp.fig(category=Category.ENERGY_LANDSCAPE))
