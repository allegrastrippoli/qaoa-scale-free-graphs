from test_example import *
from utils import max_neighborhood_size, generate_scale_free, generate_bipartite_ring_network
from plots import plot_analytical_vs_numerical, heat_map_energy_landscape,approx_ratio
from algorithms.algofactory import AlgorithmFactory
from plots import plot_degree_distribution, heat_map_energy_landscape, plot_metrics, plot_degree_distribution, plot_triangles_distribution
from classic_maxcut import maxcut_gurobi
import networkx as nx
import gurobipy as gb
import pandas as pd
import os 

def graph_info(graph_id, G, alpha, k_min, filename):
    degrees = [G.degree(n) for n in G.nodes()]
    max_ns, max_edge = max_neighborhood_size(G)
    nx.write_gml(G, filename)
    triangles_per_node = nx.triangles(G)
    triangles = sum(triangles_per_node.values()) // 3
    row = {
        "graph_id": graph_id,
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
        "gamma" : np.nan,
        "beta" : np.nan,
        "energy" : np.nan,
        "gurobi_energy" : np.nan, 
        "qaoa_ratio" : np.nan,
        "randomcut_ratio" : np.nan}
    return row

def grid(df):
    gammas = np.sort(df["gamma"].unique())
    betas = np.sort(df["beta"].unique())
    energies2d = df["energy"].values.reshape(len(gammas), len(betas))
    return gammas, betas, energies2d

def compute(fun, n_points=100):  
    # gamma_start = 0
    # gamma_end = 2*np.pi
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
    
def compute_energy_landscape(rp: RunPaths, G: nx.Graph, p=1, index=0, algo="aqaoa", **kwargs):
    q = AlgorithmFactory.create(algo=algo, G=G, p=p, **kwargs)
    df = compute(fun=q.expectation)
    filename=rp.log(OutputFile.ENERGY_LANDSCAPE, index=index)
    df.to_csv(filename, index=False)
    gammas, betas, energies2d = grid(df)
    return gammas, betas, energies2d
        
def generate_dataset(rp, n_nodes_lst, scaling_values, min_degrees, n_graphs):
    graphs = {}
    graph_id = 0
    rows = []
    for n in n_nodes_lst:
        for alpha in scaling_values:
            for k_min in min_degrees:
                for j in range(n_graphs):
                    if n <= 0:
                        raise ValueError("Number of nodes must be > 0")
                    graph_name = f"{j}_{n}nodes_{alpha}alpha_{k_min}kmin"
                    G = generate_scale_free(n=n, alpha=alpha, k_min=k_min)
                    row = graph_info(graph_id=graph_id, G=G, alpha=alpha, k_min=k_min, filename=rp.graphs(OutputFile.GRAPH, index=graph_name))
                    rows.append(row)
                    graphs[graph_id] = G
                    graph_id += 1 
                    # plot_degree_distribution(G=G, filename=rp.fig(OutputFile.DEGREE_DISTRIBUTION, index=graph_name))
    df = pd.DataFrame(rows).set_index("graph_id", drop=False)
    return df, graphs

def run_example_energy_landscape():
    run_name="test_energy_landscape"
    rp = RunPaths(run_name)
    G = nx.Graph()
    G.add_nodes_from(range(5))
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)])
    gammas, betas, energies2d = compute_energy_landscape(rp=rp, G=G, algo="aqaoa", index=0)
    heat_map_energy_landscape(gammas=gammas, betas=betas, E=energies2d, filename=rp.fig(OutputFile.ENERGY_LANDSCAPE))
    
def random_cut_value(G, n_samples=200, seed=None):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    cut_values = []
    for _ in range(n_samples):
        assignment = rng.integers(0, 2, size=n)
        node_to_side = dict(zip(nodes, assignment))
        cut = sum(1 for u, v in G.edges() if node_to_side[u] != node_to_side[v])
        cut_values.append(cut)
    return -np.mean(cut_values), np.std(cut_values)

def run_example_scale_free(n_nodes=1000, alpha=2.5, k_min=3):
    run_name="run_example_scale_free"
    rp = RunPaths(run_name)
    G = generate_scale_free(n=n_nodes, alpha=alpha, k_min=k_min)
    start_time = time.time()
    gammas, betas, energies2d = compute_energy_landscape(rp=rp, G=G, algo="sfqaoa", index=0, k_min=k_min , alpha=alpha)
    print(time.time()-start_time)
    heat_map_energy_landscape(gammas=gammas, betas=betas, E=energies2d, filename=rp.fig(OutputFile.ENERGY_LANDSCAPE, index="Our"))
    start_time = time.time()
    gammas, betas, energies2d = compute_energy_landscape(rp=rp, G=G, algo="aqaoa", index=1)
    print(time.time()-start_time)
    heat_map_energy_landscape(gammas=gammas, betas=betas, E=energies2d, filename=rp.fig(OutputFile.ENERGY_LANDSCAPE, index="Wang"))
    
def test_approx_ratio(): 
    algo_name = "aqaoa"
    n_iter_optimizer=100
    initial_angles=None
    p=1
    n = 1000
    m = 5
    G3 = nx.barabasi_albert_graph(n, m)
    # algo = AlgorithmFactory.create(algo=algo_name, G=G3, p=p)
    # algo.run(n_iter=n_iter_optimizer, initial_angles=initial_angles)
    # _, gurobi_energy = maxcut_gurobi(G3)
    # print(algo.energy / gurobi_energy)
    
    algo_name = "sfqaoa"
    alpha = 3
    degrees = [d for _, d in G3.degree()]
    algo = AlgorithmFactory.create(algo=algo_name, G=G3, p=p, k_min=m, alpha=alpha)
    algo.run(n_iter=n_iter_optimizer, initial_angles=initial_angles)
    _, gurobi_energy = maxcut_gurobi(G3)
    print("Scale-Free QAOA Approximation Ratio ", algo.energy / gurobi_energy)
    half_edges = -len(G3.edges) / 2
    print("Half Edges Cut Approximation Ratio ", half_edges / gurobi_energy)
    
    # random_cut_energy = random_cut_value(G3)[0]
    # print("Random cut Approximation Ratio", random_cut_energy / gurobi_energy)
    


if __name__ == "__main__":
    # ==================== TESTS ====================
    # test_qaoa()
    # test_aqaoa()
    # test_lcqaoa()
    # test_max_cut()
    # run_example_energy_landscape()
    # run_example_scale_free()
    # test_approx_ratio()
    # ==================== RUN INFO ====================
    run_name = "test_optimized_angles"
    rp = RunPaths(run_name)
    # ==================== GRAPH GENERATION ====================
    start_n=10
    end_n=51
    step=10
    n_nodes_lst = np.arange(start_n, end_n, step)
    print(f"{n_nodes_lst=}")
    n_graphs=3
    # ==================== SCALING VALUES ====================
    scaling_values=[2.25, 2.75]
    # ==================== MINIMUM DEGREE ====================
    min_degrees=np.arange(2,6, 2)
    print(min_degrees)
    # ==================== ALGORITHM ====================
    algo_name="sfqaoa"
    p=1
    n_iter_optimizer=100
    initial_angles=None
    rows = []
    print("Generate Dataset... 👾")
    df, graphs_by_id = generate_dataset(rp=rp, n_nodes_lst=n_nodes_lst, scaling_values=scaling_values, min_degrees=min_degrees, n_graphs=n_graphs)
    print(f"Number of generated graphs, expected: {n_graphs * len(scaling_values) * len(min_degrees) * len(n_nodes_lst)}, got: {len(graphs_by_id)}")
    print("Optimization Start... 😙") 
    for graph_id, G in graphs_by_id.items():
        # algo = AlgorithmFactory.create(algo=algo_name, G=G, p=p)
        k_min=df.at[graph_id, "k_min"]
        alpha=df.at[graph_id, "alpha"]
        algo = AlgorithmFactory.create(algo=algo_name, G=G, p=p, k_min=k_min, alpha=alpha)
        algo.run(n_iter=n_iter_optimizer, initial_angles=initial_angles)
        _, gurobi_energy = maxcut_gurobi(G)
        gamma,  beta = algo.angles
        df.loc[graph_id, ["gamma", "beta", "energy",  "gurobi_energy"]] = [gamma, beta, algo.energy, gurobi_energy]
    print(df.head())
    print("Store data... ✅")
    filename = rp.log(OutputFile.GRAPHS_INFO)
    header = not os.path.exists(filename)
    df.to_csv(filename, header=header, index=False)
    print("Plot results... 🎨")
    approx_ratio(rp, filename)
    plot_metrics(rp=rp, filename=filename)
    G = graphs_by_id[0]
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
    
    # print("Done 🥵")
