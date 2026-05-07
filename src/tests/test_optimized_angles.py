from utils.plots import  plot_optimized_angles, plot_optimized_angles_fixed_clusters
from optimization.optimizedangles import OptimizedAngles
from utils.generate import create_graph, generate_scale_free_graph
from tests.test_energy_landscape import compute_energy_landscape
from itertools import combinations
from algorithms.algofactory import AlgorithmFactory
from paths import *
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

# run_name = "optimize_angles_increasing_n_nodes_fixed_gamma"
def optimize_angles_increasing_n_nodes_fixed_gamma(run_name, start_n, end_n,  *args,  iter=100, n_graphs=10, algo_name="aqaoa", p=1, step=50, **kwargs):
    rp = RunPaths(run_name)
    n_nodes_lst = np.arange(start_n, end_n, step)
    print(f"{n_nodes_lst=}")
    gammas_lst = []
    betas_lst = []
    for n in n_nodes_lst:
        graphs = []
        for j in range(n_graphs):
            G = create_graph(rp=rp, fun=nx.barabasi_albert_graph, n_nodes=n, index=f"_{n}_nodes{j}", *args,  **kwargs)
            graphs.append(G)
        print("Optimization Start... 😙")
        oa = OptimizedAngles()
        oa.compute(algo_name=algo_name, graphs=graphs, p=p, iter=iter, history_filename=rp.log(category=Category.HISTORY))
        oa.save(filename=rp.log(category=Category.OPTIMIZED_ANGLES, index=f"_{n}_nodes"))
        gammas, betas = oa.get_opt_angles()
        gammas_lst.append(gammas)
        betas_lst.append(betas)
    print("Optimization Done 🥵")
    plot_optimized_angles_fixed_clusters(betas_lst=betas_lst, gammas_lst=gammas_lst, n_nodes_lst=n_nodes_lst, filename= rp.fig(category=Category.OPTIMIZED_ANGLES))
    G0 = graphs[0]
    print("Compute Energy Landscape... 🏞️")
    compute_energy_landscape(run_name=run_name, G=G0, algo="aqaoa", opt_gammas_lst=gammas_lst, opt_betas_lst=betas_lst, n_nodes_lst=n_nodes_lst)
    print("Done!! 🤓")
    
def build_star_graph(G, stars):
    S = nx.Graph()
    for center, leaves in stars:
        S.add_node(center)
        for leaf in leaves:
            S.add_edge(center, leaf)
    return S

def find_star_centers(G):
    stars = []
    for u in G.nodes():
        neighbors = list(G.neighbors(u))
        k = len(neighbors)
        if k < 2:
            continue 
        is_star = True
        for v, w in combinations(neighbors, 2):
            if G.has_edge(v, w):
                is_star = False
                break
        if is_star:
            stars.append((u, neighbors))
    return stars

# run_name = "star_components_in_scale_free_graphs"
def star_components_in_scale_free_graphs(run_name, start_n_nodes, end_n_nodes, m, iterations, n_graphs, algo_name="aqaoa", p=1, step=50):
    rp = RunPaths(run_name)
    n_nodes_lst = np.arange(start_n_nodes, end_n_nodes, step)
    print(f"{n_nodes_lst=}")
    ratios = []
    gamma_vals = []
    gamma_x = []
    gamma_labels = [] 
    instance_idx = 0
    
    for n_node in n_nodes_lst:
        for j in range(n_graphs):
            G = create_graph(rp=rp, fun=nx.barabasi_albert_graph, n=n_node, m=m, index=f"_{n_node}_nodes{j}")
            stars = find_star_centers(G)
            S = build_star_graph(G, stars)
            print(f"{len(G.nodes)=}")
            print(f"{len(S.nodes)=}")
            
            pos = nx.spring_layout(G)
            nx.draw(G, pos=pos)
            plt.show()
            
            poss = nx.spring_layout(S)
            nx.draw(S, pos=poss)
            plt.show()
            
            algo_sf = AlgorithmFactory.create(algo=algo_name, G=G, p=p)
            algo_sf.run(iter=iterations)
            gamma_sf, beta_sf = algo_sf.angles
            energy_sf = algo_sf.energy
            
            algo_star = AlgorithmFactory.create(algo=algo_name, G=S, p=p)
            algo_star.run(iter=iterations)
            gamma_star, beta_star = algo_star.angles
            energy_star = algo_star.energy
            
            print(gamma_sf, beta_sf, gamma_star, beta_star)

            ratio = energy_star / energy_sf
            ratios.append(ratio)

            gamma_vals.extend([gamma_sf, gamma_star])
            gamma_x.extend([instance_idx, instance_idx])
            gamma_labels.extend(["G", "S"])

            instance_idx += 1
            
    x_ratio = np.arange(len(ratios))

    plt.figure()
    plt.scatter(x_ratio, ratios)
    plt.axhline(1, linestyle='--')  
    plt.xlabel("Instance index")
    plt.ylabel("Energy ratio (star / scale-free)")
    plt.title("Energy Ratio per Instance")
    plt.grid()
    plt.show()
 
    gamma_vals = np.array(gamma_vals)
    gamma_x = np.array(gamma_x)
    gamma_labels = np.array(gamma_labels) 

    plt.figure()

    mask_G = gamma_labels == "G"
    mask_S = gamma_labels == "S"

    plt.scatter(gamma_x[mask_G], gamma_vals[mask_G], label="Scale-free (G)")
    plt.scatter(gamma_x[mask_S], gamma_vals[mask_S], label="Star (S)")

    plt.xlabel("Instance index")
    plt.ylabel("Gamma values")
    plt.title("Gamma Comparison per Instance")
    plt.legend()
    plt.show()
    

         
            


        
        
