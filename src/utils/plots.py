from matplotlib.colors import Normalize
from collections import Counter
import matplotlib.pyplot as plt
from utils.utils import *
import networkx as nx
import pandas as pd
from paths import *

def plot_metrics(rp, filename):
    df = pd.read_csv(filename, sep=",")
    df.columns = df.columns.str.strip()
    output_path = rp.dirs["metrics"]
    output_path.mkdir(parents=True, exist_ok=True)
    metrics = ["gamma", "beta", "energy", "triangles"]
    g_norm = Normalize(vmin=df["g"].min(), vmax=df["g"].max())
    nodes_norm = Normalize(vmin=df["nodes"].min(), vmax=df["nodes"].max())
    cmap = plt.cm.coolwarm
    for metric in metrics:
        plt.figure(figsize=(7, 5))
        sc = plt.scatter(df["nodes"], df[metric], c=df["g"], cmap=cmap, norm=g_norm, s=60)
        plt.xlabel("nodes")
        plt.ylabel(metric)
        plt.title(f"{metric} vs nodes")
        cbar = plt.colorbar(sc)
        cbar.set_label("g")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / f"{metric}_vs_nodes.png", dpi=300, bbox_inches="tight")
        plt.close()
    for metric in metrics:
        plt.figure(figsize=(7, 5))
        sc = plt.scatter(df["g"],df[metric], c=df["nodes"],cmap=cmap,norm=nodes_norm, s=60)
        plt.xlabel("g")
        plt.ylabel(metric)
        plt.title(f"{metric} vs g")
        cbar = plt.colorbar(sc)
        cbar.set_label("nodes")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / f"{metric}_vs_g.png", dpi=300, bbox_inches="tight")
        plt.close()
    plt.figure(figsize=(7, 5))
    
    sc = plt.scatter(df["gamma"],df["triangles"], c=df["g"],cmap=cmap,norm=g_norm, s=60)
    plt.xlabel("gamma")
    plt.ylabel("triangles")
    plt.title(f"triangles vs gamma")
    cbar = plt.colorbar(sc)
    cbar.set_label("g")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / f"triangles_vs_gamma.png", dpi=300, bbox_inches="tight")
    plt.close()
        
def plot_full_graph(G, filename, node_colors=None, edge_colors=None):
    pos = nx.spring_layout(G)    
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos=pos, node_color=node_colors,edge_color=edge_colors, node_size=200, with_labels=True)
    plt.savefig(filename, dpi=300)
    plt.close()

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

def plot_degree_distribution(G, filename):
    degrees = [d for _, d in G.degree()]
    degree_counts = Counter(degrees)
    deg, freq = zip(*sorted(degree_counts.items()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(deg, freq)
    axes[0].set_title("Degree Distribution (Linear Scale)")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Frequency")
    axes[1].scatter(deg, freq)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_title("Degree Distribution (Log-Log Scale)")
    axes[1].set_xlabel("Degree")
    axes[1].set_ylabel("Frequency")
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_optimized_angles_fixed_clusters(oa, filename):
    opt_gamma, opt_beta = oa.get_opt_angles()
    n_nodes = oa.get_number_of_nodes()
    cmap = plt.get_cmap("tab10", len(n_nodes))
    plt.figure(figsize=(8, 6))
    for i, (b, g) in enumerate(zip(opt_beta, opt_gamma)):
        plt.scatter(b, g, color=cmap(i), alpha=0.3, label=f"{n_nodes[i]} nodes")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\gamma$")
    plt.title("Optimized Angles")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()
    
def  plot_energy_landscape(gammas, betas, E, oa=None, ax=None, save_fig=True, filename=""):
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(E, extent=[betas.min(), betas.max(), gammas.min(), gammas.max()], origin="lower", cmap="magma", aspect="auto")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")
    plt.colorbar(im, ax=ax, label="Energy")
    if oa is not None:
        opt_gamma, opt_beta = oa.get_opt_angles()
        n_nodes = oa.get_number_of_nodes()
        cmap = plt.get_cmap("tab10", len(n_nodes))
        for i, (b, g) in enumerate(zip(opt_beta, opt_gamma)):
            plt.scatter(b, g, color=cmap(i), alpha=0.3, label=f"{n_nodes[i]} nodes")
            ax.legend()
    if save_fig:
        plt.savefig(filename, dpi=300)
        plt.close()
