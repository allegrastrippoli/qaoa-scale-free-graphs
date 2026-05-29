from optimization.energylandscape import EnergyLandscape
from utils.plots import plot_full_graph, plot_top_n_subgraphs, plot_energy_landscape
from algorithms.algofactory import AlgorithmFactory
from utils.utils import top_n_max_neighborhood_size, get_colors
import networkx as nx
from paths import *

# run_name="test_energy_landscape"
def compute_energy_landscape(run_name: str, G: nx.Graph, p=1, index=0, algo="qaoa",  opt_gammas_lst=None, opt_betas_lst=None, n_nodes_lst=None, **kwargs):
        rp = RunPaths(run_name)
        el = EnergyLandscape()
        q = AlgorithmFactory.create(algo=algo, G=G, p=p, **kwargs)
        el.compute(fun=q.expectation, **kwargs)
        el.save(filename=rp.log(category=Category.ENERGY_LANDSCAPE, index=index))
        gammas, betas, energies2d = el.grid()
        plot_energy_landscape(gammas=gammas, betas=betas, E=energies2d, save_fig=True, opt_gammas_lst=opt_gammas_lst, opt_betas_lst=opt_betas_lst, n_nodes_lst=n_nodes_lst, filename=rp.fig(category=Category.ENERGY_LANDSCAPE, index=index))
        
def run_example_energy_landscape(**kwargs):
    run_name="test_energy_landscape"
    G = nx.Graph()
    G.add_nodes_from(range(5))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)])
    compute_energy_landscape(run_name=run_name, G=G, algo="qaoa", **kwargs)
    compute_energy_landscape(run_name=run_name,G=G, algo="aqaoa", index=1)
