from collections import Counter
from utils.utils import *
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Given a csv file, plots the energy landscape 
def plot_energy_landscape(gammas, betas, E, filename, ax=None, save_fig=False):
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(E, extent=[betas.min(), betas.max(), gammas.min(), gammas.max()], origin="lower", cmap="magma", aspect="auto")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")
    plt.colorbar(im, ax=ax, label="Energy")
    if save_fig:
        plt.savefig(filename, dpi=300)
        plt.close()
    return im

# Given a scale free graphs, plots the degree distribution, bot linear and log-log 
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

# Input: two numpy arrays 
# Output: a scatterplot that also shows average and standard deviation 
def plot_optimized_angles(x, y, filename):
    plt.figure(figsize=(8, 6))
    x_ave = np.average(x)
    x_std = np.std(x)
    y_ave = np.average(y)
    y_std = np.std(y)
    plt.scatter(x, y)
    plt.errorbar(x_ave, y_ave, yerr=y_std, xerr=x_std, fmt='o', color='red',
                ecolor='red', capsize=5, elinewidth=2, label='Data with Std Dev')
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\gamma$")
    plt.title('Optimized Angles')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300)

# def plot_energy_landscape(fun, ax=None, save_fig=False, index=""):
#     if ax is None:
#         ax = plt.gca()
#     fig = ax.figure 
#     n_ticks = 5
#     n_points = 100
#     gammas = np.linspace(0, 2*np.pi, n_points)
#     betas = np.linspace(0, np.pi/2, n_points)
#     E = np.zeros((n_points, n_points))
#     for i, gamma in enumerate(gammas):
#         for j, beta in enumerate(betas):
#             E[i, j] = fun([gamma, beta])
#     im = ax.imshow(E, extent=[0, n_points - 1, 0, n_points - 1], origin="lower", cmap="magma", aspect="auto")
#     x_ticks = np.linspace(0, n_points - 1, n_ticks)
#     y_ticks = np.linspace(0, n_points - 1, n_ticks)
#     ax.set_xticks(x_ticks)
#     ax.set_yticks(y_ticks)
#     ax.set_xticklabels(np.round(np.linspace(betas[0], betas[-1], n_ticks), 2))
#     ax.set_yticklabels(np.round(np.linspace(gammas[0], gammas[-1], n_ticks), 2))
#     ax.set_xlabel(r"$\beta$")
#     ax.set_ylabel(r"$\gamma$")
#     fig.colorbar(im, ax=ax, label="Energy")
#     if save_fig:
#         plt.savefig(f"./utils/figures/energy_landscape_{index}.png", dpi=300)
#         plt.close()
#     return im
    
def plot_edge_subgraph(G_sub, edge, color):
    pos = nx.spring_layout(G_sub)
    edge_colors = []
    widths = []
    for e in G_sub.edges():
        if e == edge or (e[1], e[0]) == edge:
            edge_colors.append(color)
            widths.append(3.0)   
        else:
            edge_colors.append(color)
            widths.append(1.5)
    nx.draw(G_sub, pos=pos, node_color=color, edge_color=edge_colors, node_size=180, width=widths, with_labels=True)

def plot_full_graph(G, node_colors, edge_colors, filename):
    pos = nx.spring_layout(G)    
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos=pos, node_color=node_colors,edge_color=edge_colors, node_size=200, with_labels=True,)
    plt.savefig(filename, dpi=300)

# def plot_top_n_subgraphs(G, edge_color_map):
#     num = len(edge_color_map)
#     cols = 3
#     rows = (num + cols - 1) // cols
#     fig = plt.figure(figsize=(5 * cols, 8 * rows))
#     outer_gs = fig.add_gridspec(rows, cols)
#     for idx, ((edge, color)) in enumerate(edge_color_map.items()):
#         row = idx // cols
#         col = idx % cols
#         inner_gs = outer_gs[row, col].subgridspec(2, 1, height_ratios=[1, 1.2])
#         ax_graph = fig.add_subplot(inner_gs[0])
#         ax_energy = fig.add_subplot(inner_gs[1])
#         G_sub = edge_neighborhood_subgraph(G, edge)
#         pos = nx.spring_layout(G_sub, seed=42)
#         nx.draw(G_sub, pos=pos, ax=ax_graph, node_color=[color], edge_color=color, node_size=180, with_labels=True)
#         ax_graph.set_title(f"Edge {edge}")
#         u, v = edge
#         # for lc in light_cones.light_cones:
#         #     if (lc.u, lc.v) == (u, v) or (lc.v, lc.u) == (u, v):
#         #         plot_energy_landscape(lc.expectation)
#     plt.tight_layout()
#     plt.savefig("./utils/figures/top_n_subgraphs.png", dpi=300)
    
def plot_graph_subgraphs(G, top_n, top_n_edges, full_graph_filename, subgraphs_filename):
    edge_color_map, edge_colors, node_color_map, node_colors = get_colors(G, top_n, top_n_edges)
    plot_full_graph(G, node_colors, edge_colors, full_graph_filename)
    # plot_top_n_subgraphs(G, edge_color_map, subgraphs_filename) 
