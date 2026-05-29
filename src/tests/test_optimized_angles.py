from utils.plots import  plot_optimized_angles, plot_optimized_angles_fixed_clusters, plot_gamma_beta_g_energy_nodes, plot_triangles
from optimization.optimizedangles import OptimizedAngles
from utils.generate import create_graph
from tests.test_energy_landscape import compute_energy_landscape
from itertools import combinations
from algorithms.algofactory import AlgorithmFactory
from paths import *
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt
import pandas as pd

# run_name = "optimize_angles_increasing_n_nodes"
def optimize_angles_increasing_n_nodes(run_name, start_n, end_n,*args, fun=nx.barabasi_albert_graph, scaling_values=[3], n_iter=100, n_graphs=4, algo_name="aqaoa", p=1, step=50, **kwargs):
    rp = RunPaths(run_name)
    n_nodes_lst = np.arange(start_n, end_n, step)
    print(f"{n_nodes_lst=}")
    triangles = []
    gammas_lst = []
    betas_lst = []
    for n in n_nodes_lst:
        for g in scaling_values:
            graphs = []
            g_values = []
            rows = []
            for j in range(n_graphs):
                G, n_triangles = create_graph(rp=rp, fun=fun, g=g, n=n, index=f"_{n}_nodes{j}", *args, **kwargs)
                graphs.append(G)
                g_values.append(g)
                triangles.append(n_triangles)
            print("Optimization Start... 😙")
            oa = OptimizedAngles()
            for G, g in zip(graphs, g_values):
                row = oa.compute(G=G, algo_name=algo_name, p=p, iter=n_iter, g=g, *args, **kwargs)
                rows.append(row)
            oa.build_dataframe(rows)
            oa.save(filename=rp.log(category=Category.OPTIMIZED_ANGLES))
            gammas, betas = oa.get_opt_angles()
            gammas_lst.append(gammas)
            betas_lst.append(betas)
    print("Optimization Done 🥵")
    filename = rp.log(category=Category.OPTIMIZED_ANGLES)
    plot_gamma_beta_g_energy_nodes(run_name=run_name, filename=filename)
    plot_triangles(triangles)


         
            


        
        
