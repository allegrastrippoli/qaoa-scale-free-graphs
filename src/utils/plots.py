from collections import Counter
from utils.utils import *
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import networkx as nx
import numpy as np
import pandas as pd

def plot_energy_landscape(gammas, betas, E, ax=None,
                          save_fig=False,  opt_gammas_lst=None, opt_betas_lst=None, n_nodes_lst=None, filename=""):
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(E, extent=[betas.min(), betas.max(), gammas.min(), gammas.max()], origin="lower", cmap="magma", aspect="auto")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")
    plt.colorbar(im, ax=ax, label="Energy")
    cmap = plt.get_cmap("tab10", len(n_nodes_lst))
    for i, (betas, gammas) in enumerate(zip(opt_betas_lst, opt_gammas_lst)):
        plt.scatter(betas, gammas, color=cmap(i), alpha=0.3, label=f"{n_nodes_lst[i]} nodes")
        ax.legend()
    if save_fig:
        plt.savefig(filename, dpi=300)
        plt.close()


def plot_degree_distribution(G: nx.Graph, gamma: float, filename):
    degrees = [G.degree(n) for n in G.nodes()]
    degree_counts = Counter(degrees)
    ks = sorted(degree_counts.keys())
    counts = [degree_counts[k] for k in ks]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(ks, counts, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Degree (k)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Degree Distribution (Linear) - γ={gamma:.2f}')
    ks_positive = [k for k in ks if k > 0]
    counts_positive = [degree_counts[k] for k in ks_positive]
    axes[1].scatter(ks_positive, counts_positive, color='green', alpha=0.7)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Degree (k)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Degree Distribution (Log-Log) - γ={gamma:.2f}')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
def plot_optimized_angles_fixed_clusters(betas_lst, gammas_lst, n_nodes_lst, filename):
    cmap = plt.get_cmap("tab10", len(n_nodes_lst))
    plt.figure(figsize=(8, 6))
    for i, (betas, gammas) in enumerate(zip(betas_lst, gammas_lst)):
        plt.scatter(betas, gammas, color=cmap(i), alpha=0.3, label=f"{n_nodes_lst[i]} nodes")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\gamma$")
    plt.title("Optimized Angles")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()
    
def cluster_zoom_in(betas_lst, gammas_lst, n_nodes_lst, fig_dir):
    cmap = plt.get_cmap("tab10", len(n_nodes_lst))
    df = pd.DataFrame(columns=["beta", "alpha", "color", "n_nodes"])
    for i, (betas, gammas) in enumerate(zip(betas_lst, gammas_lst)):
        for b, g in zip(betas, gammas):
            df.loc[len(df)] = [b, g, cmap(i), n_nodes_lst[i]]
    X = df[["beta", "alpha"]].values
    db = DBSCAN(eps=0.05, min_samples=5)
    labels = db.fit_predict(X)
    df["cluster"] = labels
    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            continue
        cluster_df = df[df["cluster"] == cluster_id]
        plt.figure()
        for n_nodes, group in cluster_df.groupby("n_nodes"):
            plt.scatter(
                group["beta"],
                group["alpha"],
                c=[group["color"].iloc[0]],
                alpha=0.3,
                label=f"{n_nodes} nodes")
        plt.title(f"Cluster {cluster_id}")
        plt.xlabel("beta")
        plt.ylabel("alpha")
        plt.legend(title="Graph size")
        plt.savefig(f"{fig_dir}/cluster_{cluster_id}.png", dpi=300, bbox_inches="tight")
        plt.close()
        
def plot_optimized_angles(x, y, filename, eps=0.1, min_samples=5):
    plt.figure(figsize=(8, 6))
    data = np.column_stack((x, y))
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(data)
    unique_clusters = set(labels)
    plt.scatter(x, y, c=labels, cmap="tab10", alpha=0.3)
    for cluster in unique_clusters:
        if cluster == -1:
            continue 
        cluster_points = data[labels == cluster]
        x_vals = cluster_points[:, 0]
        y_vals = cluster_points[:, 1]
        x_ave = np.mean(x_vals)
        x_std = np.std(x_vals)
        y_ave = np.mean(y_vals)
        y_std = np.std(y_vals)
        plt.errorbar(x_ave, y_ave, xerr=x_std, yerr=y_std, fmt='o', capsize=5, elinewidth=2, label=f'Cluster {cluster}')
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\gamma$")
    plt.title("Optimized Angles (DBSCAN Clusters)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_full_graph(G, filename, node_colors=None, edge_colors=None):
    pos = nx.spring_layout(G)    
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos=pos, node_color=node_colors,edge_color=edge_colors, node_size=200, with_labels=True)
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_top_n_subgraphs(G, energies, edge_color_map, filename):
    # edge_color_map, edge_colors, node_color_map, node_colors = get_colors(G, top_n, top_n_edges) 
    num = len(edge_color_map)
    cols = 3
    rows = (num + cols - 1) // cols
    fig = plt.figure(figsize=(5 * cols, 8 * rows))
    outer_gs = fig.add_gridspec(rows, cols)
    for idx, ((edge, color)) in enumerate(edge_color_map.items()):
        row = idx // cols
        col = idx % cols
        inner_gs = outer_gs[row, col].subgridspec(2, 1, height_ratios=[1, 1.2])
        ax_graph = fig.add_subplot(inner_gs[0])
        ax_energy = fig.add_subplot(inner_gs[1])
        G_sub = edge_neighborhood_subgraph(G, edge)
        pos = nx.spring_layout(G_sub, seed=42)
        nx.draw(G_sub, pos=pos, ax=ax_graph, node_color=[color], edge_color=color, node_size=180, with_labels=True)
        ax_graph.set_title(f"Edge {edge}")
        plot_energy_landscape(energies[edge][0], energies[edge][1], energies[edge][2], save_fig=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
           
def plot_max_cut(G, best_bitstring, filename):
    node_colors = []
    for bit in best_bitstring:
        if bit == "0":
            node_colors.append("red")
        elif bit == "1":
            node_colors.append("green")
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(8, 8)) 
    nx.draw(G, pos=pos, node_color=node_colors, node_size=200, with_labels=True, ax=ax)
    ax.set_title(f"Max cut: {best_bitstring}")
    plt.savefig(filename, dpi=300)
    plt.close(fig)
