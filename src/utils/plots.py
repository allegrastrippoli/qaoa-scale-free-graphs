from matplotlib.colors import Normalize
from collections import Counter
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from utils.utils import *
import networkx as nx
import pandas as pd
from paths import *

def _prepare_grouped(df):
    grouped = (
        df.groupby(["k_min", "g"])["gamma"]
        .agg(mean="mean", std="std")
        .reset_index())
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped

def plot_analytical_vs_numerical(rp, filename):
    data = [
        {"k_min": 1, "g": 2,    "gamma": 0.52},
        {"k_min": 1, "g": 2.25, "gamma": 0.64},
        {"k_min": 1, "g": 2.5,  "gamma": 0.72},
        {"k_min": 1, "g": 2.75, "gamma": 0.79},
        {"k_min": 1, "g": 3,    "gamma": 0.84},
        {"k_min": 2, "g": 2,    "gamma": 0.34},
        {"k_min": 2, "g": 2.25, "gamma": 0.41},
        {"k_min": 2, "g": 2.5,  "gamma": 0.46},
        {"k_min": 2, "g": 2.75, "gamma": 0.50},
        {"k_min": 2, "g": 3,    "gamma": 0.53},
        {"k_min": 3, "g": 2,    "gamma": 0.27},
        {"k_min": 3, "g": 2.25, "gamma": 0.33},
        {"k_min": 3, "g": 2.5,  "gamma": 0.37},
        {"k_min": 3, "g": 2.75, "gamma": 0.40},
        {"k_min": 3, "g": 3,    "gamma": 0.43},
        {"k_min": 4, "g": 2,    "gamma": 0.23},
        {"k_min": 4, "g": 2.25, "gamma": 0.28},
        {"k_min": 4, "g": 2.5,  "gamma": 0.32},
        {"k_min": 4, "g": 2.75, "gamma": 0.34},
        {"k_min": 4, "g": 3,    "gamma": 0.36},
        {"k_min": 5, "g": 2,    "gamma": 0.20},
        {"k_min": 5, "g": 2.25, "gamma": 0.25},
        {"k_min": 5, "g": 2.5,  "gamma": 0.28},
        {"k_min": 5, "g": 2.75, "gamma": 0.30},
        {"k_min": 5, "g": 3,    "gamma": 0.32}
    ]
    analytical_df = pd.DataFrame(data)
    df = pd.read_csv(filename, sep=",")
    df.columns = df.columns.str.strip()
    output_path = rp.dirs["metrics"]
    output_path.mkdir(parents=True, exist_ok=True)
    grouped_num = _prepare_grouped(df)
    grouped_ana = _prepare_grouped(analytical_df)
    all_k = sorted(set(grouped_num["k_min"]).union(grouped_ana["k_min"]))
    cmap = plt.cm.get_cmap("tab10", len(all_k)) 
    color_map = {k: cmap(i) for i, k in enumerate(all_k)}
    fig, ax = plt.subplots(figsize=(8, 6))
    def plot_grouped_numerical(grouped):
        for k_val, subdf in grouped.groupby("k_min"):
            subdf = subdf.sort_values("g")
            ax.errorbar(
                subdf["g"],
                subdf["mean"],
                yerr=subdf["std"],
                fmt="o-",
                color=color_map[k_val],
                capsize=3,
                linewidth=1.5,
                markersize=5,
                label=f"min_deg={k_val}")

    def plot_grouped_analytical(grouped):
        for k_val, subdf in grouped.groupby("k_min"):
            subdf = subdf.sort_values("g")
            ax.plot(
                subdf["g"],
                subdf["mean"],
                linestyle="--",
                marker="o",
                color=color_map[k_val],
                linewidth=1.5,
                markersize=5,
                label=f"min_deg={k_val}")
    plot_grouped_numerical(grouped_num)
    plot_grouped_analytical(grouped_ana)
    ax.set_xlabel("g")
    ax.set_ylabel(r"$\gamma$")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / "gamma_vs_g_by_kmin.png", dpi=300)
    plt.close()

def plot_triangles_distribution(rp, filename, index):
    df = pd.read_csv(filename, sep=",")
    df.columns = df.columns.str.strip()
    output_path = rp.dirs["metrics"]
    output_path.mkdir(parents=True, exist_ok=True)
    grouped = (df.groupby(["g", "nodes"]).agg(triangles_mean=("triangles", "mean"), triangles_std=("triangles", "std")).reset_index())
    grouped["triangles_std"] = grouped["triangles_std"].fillna(0)
    cmap = plt.cm.coolwarm
    norm = Normalize(vmin=df["g"].min(), vmax=df["g"].max())
    fig, ax = plt.subplots(figsize=(8, 6))
    for g_value, group in grouped.groupby("g"):
        group = group.sort_values("nodes")
        color = cmap(norm(g_value))
        ax.errorbar(
            group["nodes"],
            group["triangles_mean"],
            yerr=group["triangles_std"],
            fmt="o-",
            color=color,
            capsize=3,
            label=f"g={g_value}")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="g")
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Mean number of triangles")
    ax.set_title("Triangles vs Nodes for different g")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f"triangles_distribution{index}.png", dpi=300)
    plt.close()

def plot_metrics(rp, filename):
    df = pd.read_csv(filename, sep=",")
    df.columns = df.columns.str.strip()
    output_path = rp.dirs["metrics"]
    output_path.mkdir(parents=True, exist_ok=True)
    g_norm = Normalize(vmin=df["g"].min(), vmax=df["g"].max())
    nodes_norm = Normalize(vmin=df["nodes"].min(), vmax=df["nodes"].max())
    cmap = plt.cm.coolwarm
    metrics = ["gamma", "beta", "energy", "triangles"]
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
    metrics = ["energy", "triangles"]
    for metric in metrics:
        plt.figure(figsize=(7, 5))
        sc = plt.scatter(df["gamma"],df[metric], c=df["g"],cmap=cmap,norm=g_norm, s=60)
        plt.xlabel("gamma")
        plt.ylabel(metric)
        plt.title(f"{metric} vs gamma")
        cbar = plt.colorbar(sc)
        cbar.set_label("g")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / f"{metric}_vs_gamma.png", dpi=300, bbox_inches="tight")
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
