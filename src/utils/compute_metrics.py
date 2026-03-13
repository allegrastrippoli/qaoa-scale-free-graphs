from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd
from qmodels.rqaoa import RQAOA
from qmodels.qaoa import QAOA
from qmodels.lightcones import Simulation
from utils.hamiltonians import graph_to_hamiltonian
import csv 

def energy_to_csv(fun, filename="./utils/csv/energy_landscape.csv"):
    n_points = 100
    gammas = np.linspace(0, 2*np.pi, n_points)
    betas = np.linspace(0, np.pi/2, n_points)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma", "beta", "energy"])
        for gamma in gammas:
            for beta in betas:
                energy = fun([gamma, beta])
                writer.writerow([gamma, beta, energy])

def run_simulation(G, p):
    S = Simulation(G, p)
    S.run()
    return S
 
def run_qaoa(G, p):
    H = graph_to_hamiltonian(nx.to_numpy_array(G), len(G.nodes))
    Q = QAOA(p, H)
    Q.run()
    return Q
  
def run_rqaoa(G, p):
    H = graph_to_hamiltonian(nx.to_numpy_array(G), len(G.nodes))
    Q = QAOA(p, H)
    rq = RQAOA(p, H, Q, G)
    rq.run()
    return rq

def optimized_angles_to_csv(graphs, runner, filename="./utils/csv/optimized_angles.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "gamma", "beta"])
        for i, G in enumerate(graphs):
            algo = runner(G)
            gamma, beta = algo.angles 
            writer.writerow([i, gamma, beta])
              
def load_generated_graphs(directory="./utils/graphs"):
    graphs = []
    for gml_file in Path(directory).glob("*.gml"):
        G = nx.read_gml(gml_file)
        graphs.append(G)
    return graphs

def load_optimized_angles(csv_file="./utils/csv/optimized_angles.csv"):
    df = pd.read_csv(csv_file)
    gammas = df["gamma"].to_numpy()
    betas = df["beta"].to_numpy()
    return gammas, betas
