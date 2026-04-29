from algorithms.algofactory import AlgorithmFactory
from utils.file_utils import history_to_csv
import numpy as np
import pandas as pd

class OptimizedAngles:
    def __init__(self, df=None, algo=None):
        self.df = df
        
    def compute(self, graphs, p, algo_name, history_filename=None, initial_angles=None, iter=0):
        data = []
        for i, G in enumerate(graphs):
            algo = AlgorithmFactory.create(algo=algo_name, G=G, p=p)
            algo.run(iter=iter, initial_angles=initial_angles)
            gamma, beta = algo.angles
            data.append([i, gamma, beta, algo.energy])
            # if hasattr(algo, "history") and algo.history:
            #     history_to_csv(algo_name=algo_name, best_bitstring=algo.best_bitstring, history=algo.history, filename=history_filename)
        self.df = pd.DataFrame(data, columns=["graph_id", "gamma", "beta", "energy"])
            
    def save(self, filename):
         self.df.to_csv(filename, index=False)
         
    def load(self, filename):
        self.df = pd.read_csv(filename)
        
    def get_opt_angles(self):
         gammas = self.df["gamma"].to_numpy()
         betas = self.df["beta"].to_numpy()
         return gammas, betas
     
    def get_min_energies(self):
        return self.df["energy"].to_numpy()
