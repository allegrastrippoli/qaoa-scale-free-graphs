from test_example import *
from utils import max_neighborhood_size, generate_scale_free
from plots import plot_analytical_vs_numerical, heat_map_energy_landscape
from algorithms.algofactory import AlgorithmFactory
from plots import plot_degree_distribution, heat_map_energy_landscape, plot_metrics, plot_degree_distribution, plot_triangles_distribution
import networkx as nx
import pandas as pd
import os 

def graph_info(G, alpha, k_min, filename):
    degrees = [G.degree(n) for n in G.nodes()]
    max_ns, max_edge = max_neighborhood_size(G)
    nx.write_gml(G, filename)
    triangles_per_node = nx.triangles(G)
    triangles = sum(triangles_per_node.values()) // 3
    row = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "alpha" : alpha,   
        "k_min" : k_min, 
        "connected": nx.is_connected(G),
        "max_degree": max(degrees),
        "min_degree": min(degrees),
        "avg_degree": np.mean(degrees),
        "max_neighborhood_size": max_ns,
        "triangles": triangles,
        "gamma" : pd.NA,
        "beta" : pd.NA,
        "energy" : pd.NA}
    return row


def grid(df):
    gammas = np.sort(df["gamma"].unique())
    betas = np.sort(df["beta"].unique())
    energies2d = df["energy"].values.reshape(len(gammas), len(betas))
    return gammas, betas, energies2d

def compute(fun, n_points=100):  
    gamma_start = -np.pi/2
    gamma_end = np.pi/2
    beta_start = 0
    beta_end = np.pi/2
    gammas = np.linspace(gamma_start, gamma_end, n_points)
    betas = np.linspace(beta_start, beta_end, n_points)
    data = []
    for gamma in gammas:
        for beta in betas:
            exp = fun([gamma, beta])
            data.append((gamma, beta, exp))
    return pd.DataFrame(data, columns=["gamma", "beta", "energy"])
    
def compute_energy_landscape(rp: RunPaths, G: nx.Graph, p=1, index=0, algo="aqaoa"):
    q = AlgorithmFactory.create(algo=algo, G=G, p=p)
    df = compute(fun=q.expectation)
    filename=rp.log(OutputFile.ENERGY_LANDSCAPE, index=index)
    df.to_csv(filename, index=False)
    gammas, betas, energies2d = grid(df)
    return gammas, betas, energies2d
        
def generate_dataset(rp, n_nodes_lst, scaling_values, min_degrees, n_graphs):
    graphs = []
    rows = []
    for n in n_nodes_lst:
        for alpha in scaling_values:
            for k_min in min_degrees:
                for j in range(n_graphs):
                    if n <= 0:
                        raise ValueError("Number of nodes must be > 0")
                    graph_name = f"{j}_{n}nodes_{alpha}alpha_{k_min}kmin"
                    G = generate_scale_free(n=n, alpha=alpha, k_min=k_min)
                    row = graph_info(G=G, alpha=alpha, k_min=k_min, filename=rp.graphs(OutputFile.GRAPH, index=graph_name))
                    rows.append(row)
                    plot_degree_distribution(G=G, filename=rp.fig(OutputFile.DEGREE_DISTRIBUTION, index=graph_name))
                    graphs.append(G)
    df = pd.DataFrame(rows)
    return df, graphs


if __name__ == "__main__":
    # test_qaoa()
    # test_aqaoa()
    # test_lcqaoa()
    # test_max_cut()
    # test_regular_graphs()
    # run_example_energy_landscape()
    run_name = "test_optimized_angles"
    rp = RunPaths(run_name)
    start_n=50
    end_n=101
    step=50
    n_graphs=3
    p=1
    n_iter=10
    algo_name="aqaoa"
    initial_angles=None
    scaling_values=[2.25, 2.50]
    min_degrees=np.arange(1,3)
    n_nodes_lst = np.arange(start_n, end_n, step)
    print(f"{n_nodes_lst=}")
    rows = []
    print("Generate Dataset... 👾")
    df, graphs= generate_dataset(rp=rp, n_nodes_lst=n_nodes_lst, scaling_values=scaling_values, min_degrees=min_degrees, n_graphs=n_graphs)
    print("Optimization Start... 😙") 
    for i, G in enumerate(graphs):
        algo = AlgorithmFactory.create(algo=algo_name, G=G, p=p)
        algo.run(iter=n_iter, initial_angles=initial_angles)
        gamma,  beta = algo.angles
        df.loc[i, ["gamma", "beta", "energy"]] = [gamma, beta, algo.energy]
    df = df.reset_index()
    print(df.head)
    print("Store data... ✅")
    filename = rp.log(OutputFile.GRAPHS_INFO)
    header = not os.path.exists(filename)
    df.to_csv(filename, header=header, index=False)
    print("Plot results... 🎨")
    plot_metrics(rp=rp, filename=filename)
    print("Done 🥵")
    G = graphs[0]
    gammas, betas, energies2d = compute_energy_landscape(rp=rp, G=G)
    heat_map_energy_landscape(gammas=gammas, betas=betas, E=energies2d, df=df, filename=rp.fig(OutputFile.ENERGY_LANDSCAPE))  
    filename = rp.log(OutputFile.GRAPHS_INFO)
    plot_analytical_vs_numerical(rp=rp, filename=filename)
    
    # run_name = "test_triangles_distribution"
    # rp = RunPaths(run_name)
    # df, graphs= generate_dataset(rp=rp, n_nodes_lst=n_nodes_lst, scaling_values=scaling_values, min_degrees=min_degrees, n_graphs=n_graphs)
    # filename = rp.log(OutputFile.GRAPHS_INFO)
    # header = not os.path.exists(filename)
    # df.to_csv(filename, header=header, index=False)
    # filename = rp.log(OutputFile.GRAPHS_INFO)
    # plot_triangles_distribution(rp=rp, filename=filename)
    
    # run_example_scale_free(fun=generate_scale_free, n=1000, g=2.5, k_min=2)
    
    