from utils.plots import  plot_optimized_angles, plot_optimized_angles_fixed_clusters
from experiments.optimizedangles import OptimizedAngles
from utils.generate import *
from paths import *

def optimize_angles_fixed_n_nodes_fixed_gamma(n_nodes, gamma, multistart_iter, n_graphs):
    p = 1 
    graphs = []
    run_name = "optimize_angles_fixed_n_nodes_fixed_gamma"
    rp = RunPaths(run_name)
    for i in range(n_graphs):
        G = create_graph(rp=rp, fun=generate_bounded_scale_free_graph, n_nodes=n_nodes, gamma=gamma, index=i)
        graphs.append(G)
    oa = OptimizedAngles()
    oa.compute(algo_name="lcqaoa", graphs=graphs, p=p, multistart_iter=multistart_iter, history_filename=rp.log(category=Category.HISTORY))
    oa.save(filename=rp.log(category=Category.OPTIMIZED_ANGLES))
    gammas, betas = oa.opt_angles()
    plot_optimized_angles(x=betas, y=gammas, filename=rp.fig(category=Category.OPTIMIZED_ANGLES))
    
def optimize_angles_increasing_n_nodes_fixed_gamma(start_n_nodes, end_n_nodes, gamma, multistart_iter, n_graphs):
    p = 1 
    run_name = "optimize_angles_increasing_n_nodes_fixed_gamma"
    rp = RunPaths(run_name)
    n_nodes_lst = np.arange(start_n_nodes, end_n_nodes, 20)
    gammas_lst = []
    betas_lst = []
    for i, n_node in enumerate(n_nodes_lst):
        graphs = []
        for _ in range(n_graphs):
            G = create_graph(rp=rp, fun=generate_bounded_scale_free_graph, n_nodes=n_node, gamma=gamma, index=i)
            graphs.append(G)
        oa = OptimizedAngles()
        oa.compute(algo_name="lcqaoa", graphs=graphs, p=p, multistart_iter=multistart_iter, history_filename=rp.log(category=Category.HISTORY))
        oa.save(filename=rp.log(category=Category.OPTIMIZED_ANGLES), index=i)
        gammas, betas = oa.opt_angles()
        gammas_lst.append(gammas)
        betas_lst.append(betas)
    plot_optimized_angles_fixed_clusters(betas_lst=betas_lst, gammas_lst=gammas_lst, n_nodes_lst=n_nodes_lst, filename= rp.fig(category=Category.OPTIMIZED_ANGLES))
