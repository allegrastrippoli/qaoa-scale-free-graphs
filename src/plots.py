from matplotlib.colors import Normalize
from collections import Counter
import matplotlib.pyplot as plt
from utils import *
import networkx as nx
import pandas as pd
from paths import *

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
    
def heat_map_energy_landscape(gammas, betas, E, df=None, ax=None, save_fig=True, filename=""):
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(E, extent=[betas.min(), betas.max(), gammas.min(), gammas.max()], origin="lower", cmap="magma", aspect="auto")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")
    plt.colorbar(im, ax=ax, label="Energy")
    if df is not None:
        opt_gamma = df["gamma"].to_numpy()
        opt_beta = df["beta"].to_numpy()
        n_nodes = df["nodes"]
        unique_nodes = sorted(set(n_nodes))
        cmap = plt.get_cmap("tab10", len(unique_nodes))
        color_map = {n: cmap(i) for i, n in enumerate(unique_nodes)}
        for b, g, n in zip(opt_beta, opt_gamma, n_nodes):
            plt.scatter(b, g, color=color_map[n], alpha=0.3)
        handles = [plt.Line2D([], [], marker='o', linestyle='', color=color_map[n], label=f"{n} nodes") for n in unique_nodes]
        ax.legend(handles=handles)
    if save_fig:
        plt.savefig(filename, dpi=300)
        plt.close()

def _prepare_grouped(df):
    grouped = (
        df.groupby(["k_min", "alpha"])["gamma"]
        .agg(mean="mean", std="std")
        .reset_index())
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped

def plot_analytical_vs_numerical(rp, filename):
    data = [
        {"k_min": 1, "alpha": 2,    "gamma": 0.52},
        {"k_min": 1, "alpha": 2.25, "gamma": 0.64},
        {"k_min": 1, "alpha": 2.5,  "gamma": 0.72},
        {"k_min": 1, "alpha": 2.75, "gamma": 0.79},
        {"k_min": 1, "alpha": 3,    "gamma": 0.84},
        {"k_min": 2, "alpha": 2,    "gamma": 0.34},
        {"k_min": 2, "alpha": 2.25, "gamma": 0.41},
        {"k_min": 2, "alpha": 2.5,  "gamma": 0.46},
        {"k_min": 2, "alpha": 2.75, "gamma": 0.50},
        {"k_min": 2, "alpha": 3,    "gamma": 0.53},
        {"k_min": 3, "alpha": 2,    "gamma": 0.27},
        {"k_min": 3, "alpha": 2.25, "gamma": 0.33},
        {"k_min": 3, "alpha": 2.5,  "gamma": 0.37},
        {"k_min": 3, "alpha": 2.75, "gamma": 0.40},
        {"k_min": 3, "alpha": 3,    "gamma": 0.43},
        {"k_min": 4, "alpha": 2,    "gamma": 0.23},
        {"k_min": 4, "alpha": 2.25, "gamma": 0.28},
        {"k_min": 4, "alpha": 2.5,  "gamma": 0.32},
        {"k_min": 4, "alpha": 2.75, "gamma": 0.34},
        {"k_min": 4, "alpha": 3,    "gamma": 0.36},
        {"k_min": 5, "alpha": 2,    "gamma": 0.20},
        {"k_min": 5, "alpha": 2.25, "gamma": 0.25},
        {"k_min": 5, "alpha": 2.5,  "gamma": 0.28},
        {"k_min": 5, "alpha": 2.75, "gamma": 0.30},
        {"k_min": 5, "alpha": 3,    "gamma": 0.32}
    ]
    analytical_df = pd.DataFrame(data)
    df = pd.read_csv(filename, sep=",")
    df.columns = df.columns.str.strip()
    output_path = rp.dirs["metrics"]
    output_path.mkdir(parents=True, exist_ok=True)
    grouped_num = _prepare_grouped(df)
    grouped_ana = _prepare_grouped(analytical_df)
    all_k = sorted(set(grouped_num["k_min"]).union(grouped_ana["k_min"]))
    cmap = plt.get_cmap("tab10", len(all_k)) 
    color_map = {k: cmap(i) for i, k in enumerate(all_k)}
    fig, ax = plt.subplots(figsize=(8, 6))
    def plot_grouped_numerical(grouped):
        for k_val, subdf in grouped.groupby("k_min"):
            subdf = subdf.sort_values("alpha")
            ax.errorbar(
                subdf["alpha"],
                subdf["mean"],
                yerr=subdf["std"],
                fmt="o-",
                color=color_map[k_val],
                capsize=3,
                linewidth=1.5,
                markersize=5,
                label=f"min degree={k_val}")
    def plot_grouped_analytical(grouped):
        for k_val, subdf in grouped.groupby("k_min"):
            subdf = subdf.sort_values("alpha")
            ax.plot(
                subdf["alpha"],
                subdf["mean"],
                linestyle="--",
                marker="o",
                color=color_map[k_val],
                linewidth=1.5,
                markersize=5,
                label=f"min degree={k_val}")
    plot_grouped_numerical(grouped_num)
    plot_grouped_analytical(grouped_ana)
    ax.set_xlabel("alpha")
    ax.set_ylabel(r"$\gamma$")
    ax.legend()
    plt.title("Analytical and Numerical comparison")
    plt.tight_layout()
    plt.savefig(output_path / "analytical_numerical_comparison.png", dpi=300)
    plt.close()

def plot_triangles_distribution(rp, filename, index=0):
    df = pd.read_csv(filename, sep=",")
    df.columns = df.columns.str.strip()
    output_path = rp.dirs["metrics"]
    output_path.mkdir(parents=True, exist_ok=True)
    grouped = (df.groupby(["alpha", "nodes"]).agg(triangles_mean=("triangles", "mean"), triangles_std=("triangles", "std")).reset_index())
    grouped["triangles_std"] = grouped["triangles_std"].fillna(0)
    cmap = plt.cm.coolwarm
    norm = Normalize(vmin=df["alpha"].min(), vmax=df["alpha"].max())
    fig, ax = plt.subplots(figsize=(8, 6))
    for g_value, group in grouped.groupby("alpha"):
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
    fig.colorbar(sm, ax=ax, label="alpha")
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Number of triangles")
    ax.set_title("Triangles vs Nodes")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f"triangles_distribution{index}.png", dpi=300)
    plt.close()

def plot_metrics(rp, filename):
    df = pd.read_csv(filename, sep=",")
    df.columns = df.columns.str.strip()
    output_path = rp.dirs["fig"]
    g_norm = Normalize(vmin=df["alpha"].min(), vmax=df["alpha"].max())
    nodes_norm = Normalize(vmin=df["nodes"].min(), vmax=df["nodes"].max())
    cmap = plt.cm.coolwarm
    metrics = ["gamma", "beta", "energy", "triangles"]
    for metric in metrics:
        plt.figure(figsize=(7, 5))
        sc = plt.scatter(df["nodes"], df[metric], c=df["alpha"], cmap=cmap, norm=g_norm, s=60)
        plt.xlabel("nodes")
        if metric == "gamma":
            plt.ylabel(r"$\gamma$")
            plt.title(r"$\gamma$ vs nodes")
        elif metric == "beta":
            plt.ylabel(r"$\beta$")
            plt.title(r"$\beta$ vs nodes")
        else:
            plt.ylabel(metric)
            plt.title(f"{metric} vs nodes")
        cbar = plt.colorbar(sc)
        cbar.set_label("alpha")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / f"{metric}_vs_nodes.png", dpi=300, bbox_inches="tight")
        plt.close()
    for metric in metrics:
        plt.figure(figsize=(7, 5))
        sc = plt.scatter(df["alpha"],df[metric], c=df["nodes"],cmap=cmap,norm=nodes_norm, s=60)
        plt.xlabel("alpha")
        if metric == "gamma":
            plt.ylabel(r"$\gamma$")
            plt.title(r"$\gamma$ vs alpha")
        elif metric == "beta":
            plt.ylabel(r"$\beta$")
            plt.title(r"$\beta$ vs alpha")
        else:
            plt.ylabel(metric)
            plt.title(f"{metric} vs alpha")
        cbar = plt.colorbar(sc)
        cbar.set_label("nodes")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / f"{metric}_vs_g.png", dpi=300, bbox_inches="tight")
        plt.close()
    metrics = ["energy", "triangles"]
    for metric in metrics:
        plt.figure(figsize=(7, 5))
        sc = plt.scatter(df["gamma"],df[metric], c=df["alpha"],cmap=cmap,norm=g_norm, s=60)
        plt.xlabel(r"$\gamma$")
        plt.ylabel(metric)
        plt.title(fr"{metric} vs $\gamma$")
        cbar = plt.colorbar(sc)
        cbar.set_label("alpha")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / f"{metric}_vs_gamma.png", dpi=300, bbox_inches="tight")
        plt.close()     
        
def approx_ratio(rp, filename):
    df = pd.read_csv(filename, sep=",")
    df.columns = df.columns.str.strip()
    df["ratio"] = df["energy"] / df["gurobi_energy"]
    plt.figure(figsize=(7, 5))
    sc = plt.plot([i for i in range(len(df["ratio"]))],df["ratio"], color="pink")
    plt.xlabel(r"Graphs")
    plt.ylabel("Approximation Ratio")
    plt.show()
    plt.tight_layout()
    plt.close()     
    
    
def approx_ratio(rp, filename):
    df = pd.read_csv(filename, sep=",")
    df.columns = df.columns.str.strip()
    df["qaoa_ratio"] = df["energy"] / df["gurobi_energy"]
    df["randomcut_ratio"] = (-(df["edges"] / 2)) / df["gurobi_energy"]
    plt.figure(figsize=(7, 5))
    for (k_min, alpha), group in df.groupby(["k_min", "alpha"]):
        stats = (group.groupby("nodes")
            .agg(
                qaoa_mean=("qaoa_ratio", "mean"),
                qaoa_std=("qaoa_ratio", "std"),
                random_mean=("randomcut_ratio", "mean"),
                random_std=("randomcut_ratio", "std"),)
            .reset_index()
            .sort_values("nodes"))
        label_qaoa = f"QAOA (k={k_min}, α={alpha})"
        label_rand = f"Random (k={k_min}, α={alpha})"
        plt.errorbar(
            stats["nodes"],
            stats["qaoa_mean"],
            yerr=stats["qaoa_std"],
            marker="o",
            capsize=4,
            label=label_qaoa,)
        plt.errorbar(
            stats["nodes"],
            stats["random_mean"],
            yerr=stats["random_std"],
            marker="s",
            linestyle="--",
            capsize=4,
            label=label_rand,)
    plt.xlabel("Number of nodes")
    plt.ylabel("Approximation Ratio")
    plt.title("Approximation Ratio vs. Random Cut Baseline")
    plt.legend()
    plt.tight_layout()
    filename = rp.fig(OutputFile.APPROX_RATIO)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
