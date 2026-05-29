from utils.plots import plot_gamma_beta_g_energy_nodes, plot_triangles
from optimization.optimizedangles import OptimizedAngles
from utils.file_utils import graph_info
from utils.plots import plot_degree_distribution
from paths import *
import numpy as np
import networkx as nx 

def create_graph(rp, fun, n, g, *args, index=None, **kwargs):
    if n <= 0:
        raise ValueError("Number of nodes must be > 0")
    G = fun(n=n, gamma=g, *args, **kwargs)
    triangles_per_node = nx.triangles(G)
    triangles = sum(triangles_per_node.values()) // 3
    graph_info(G=G, graphs_info_filename=rp.log(category=Category.GRAPHS_INFO), graph_filename=rp.graphs(category=Category.GRAPH, index=index))
    plot_degree_distribution(G=G, filename=rp.fig(category=Category.DEGREE_DISTRIBUTION, index=index))
    return G, triangles

def generate_dataset(rp, fun, n_nodes_lst, scaling_values, n_graphs, *args, **kwargs):
    triangles = []
    graphs = []
    g_values = []
    for n in n_nodes_lst:
        for g in scaling_values:
            for j in range(n_graphs):
                G, n_triangles = create_graph(rp=rp, fun=fun, g=g, n=n, index=f"_{n}_nodes{j}", *args, **kwargs)
                graphs.append(G)
                g_values.append(g)
                triangles.append(n_triangles)
    return triangles, graphs, g_values

# run_name = "optimize_angles_increasing_n_nodes"
def optimize_angles_increasing_n_nodes(run_name, start_n, end_n,*args, fun=nx.barabasi_albert_graph, scaling_values=[3], n_iter=100, n_graphs=4, algo_name="aqaoa", p=1, step=50, **kwargs):
    rp = RunPaths(run_name)
    n_nodes_lst = np.arange(start_n, end_n, step)
    print(f"{n_nodes_lst=}")
    gammas_lst = []
    betas_lst = []
    rows = []
    
    print("Generate Dataset... 👾")
    triangles, graphs, g_values = generate_dataset(rp=rp, fun=fun, n_nodes_lst=n_nodes_lst, scaling_values=scaling_values, n_graphs=n_graphs, *args, **kwargs)
    
    print("Optimization Start... 😙")
    oa = OptimizedAngles()
    for G, g in zip(graphs, g_values):
        row = oa.compute(G=G, algo_name=algo_name, p=p, iter=n_iter, g=g, *args, **kwargs)
        rows.append(row)
    
    print("Store data... ✅")
    oa.build_dataframe(rows)
    oa.save(filename=rp.log(category=Category.OPTIMIZED_ANGLES))
    
    print("Plot results... 🎨")
    gammas, betas = oa.get_opt_angles()
    gammas_lst.append(gammas)
    betas_lst.append(betas)
    filename = rp.log(category=Category.OPTIMIZED_ANGLES)
    plot_gamma_beta_g_energy_nodes(rp=rp, filename=filename)
    plot_triangles(triangles)
    print("Done 🥵")
