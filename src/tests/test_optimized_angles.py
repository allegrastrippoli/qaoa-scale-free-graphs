from utils.plots import  plot_optimized_angles, plot_optimized_angles_fixed_clusters
from optimization.optimizedangles import OptimizedAngles
from utils.generate import generate_scale_free_graph, generate_bounded_scale_free_graph, create_graph
from tests.test_energy_landscape import compute_energy_landscape
from paths import *
import numpy as np

# run_name = "optimize_angles_fixed_n_nodes_fixed_gamma"
def optimize_angles_fixed_n_nodes_fixed_gamma(run_name: str, n_nodes: int, gamma: float, multistart_iter: int, n_graphs: int, algo="aqaoa", p=1):
    graphs = []
    rp = RunPaths(run_name)
    for i in range(n_graphs):
        G = create_graph(rp=rp, fun=generate_scale_free_graph, n_nodes=n_nodes, gamma=gamma, index=i)
        graphs.append(G)
    oa = OptimizedAngles()
    oa.compute(algo_name=algo, graphs=graphs, p=p, multistart_iter=multistart_iter, history_filename=rp.log(category=Category.HISTORY))
    oa.save(filename=rp.log(category=Category.OPTIMIZED_ANGLES))
    gammas, betas = oa.opt_angles()
    plot_optimized_angles(x=betas, y=gammas, filename=rp.fig(category=Category.OPTIMIZED_ANGLES))
    
# run_name = "optimize_angles_increasing_n_nodes_fixed_gamma"
def optimize_angles_increasing_n_nodes_fixed_gamma(run_name, start_n_nodes, end_n_nodes, gamma, multistart_iter, n_graphs, algo="lcqaoa", p=1):
    rp = RunPaths(run_name)
    n_nodes_lst = np.arange(start_n_nodes, end_n_nodes, 20)
    print(f"{n_nodes_lst=}")
    gammas_lst = []
    betas_lst = []
    for n_node in n_nodes_lst:
        graphs = []
        for j in range(n_graphs):
            G = create_graph(rp=rp, fun=generate_scale_free_graph, n_nodes=n_node, gamma=gamma, index=f"_{n_node}_nodes{j}")
            graphs.append(G)
        print("Optimization Start... 😙")
        oa = OptimizedAngles()
        oa.compute(algo_name=algo, graphs=graphs, p=p, multistart_iter=multistart_iter, history_filename=rp.log(category=Category.HISTORY))
        oa.save(filename=rp.log(category=Category.OPTIMIZED_ANGLES, index=f"_{n_node}_nodes"))
        gammas, betas = oa.get_opt_angles()
        gammas_lst.append(gammas)
        betas_lst.append(betas)
    print("Optimization Done 🥵")
    plot_optimized_angles_fixed_clusters(betas_lst=betas_lst, gammas_lst=gammas_lst, n_nodes_lst=n_nodes_lst, filename= rp.fig(category=Category.OPTIMIZED_ANGLES))
    G0 = graphs[0]
    print("Compute Energy Landscape... 🏞️")
    compute_energy_landscape(run_name=run_name, G=G0, algo="aqaoa", opt_gammas_lst=gammas_lst, opt_betas_lst=betas_lst, n_nodes_lst=n_nodes_lst)
    print("Done!! 🤓")
    
