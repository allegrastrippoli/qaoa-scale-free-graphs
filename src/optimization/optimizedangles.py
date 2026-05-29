from algorithms.algofactory import AlgorithmFactory
import numpy as np
import pandas as pd
import os 

class OptimizedAngles:
    def __init__(self, df=None, algo=None):
        self.df = df
        
    def compute(self, G, p, algo_name, *args, initial_angles=None, iter=0,  **kwargs):
        algo = AlgorithmFactory.create(algo=algo_name, G=G, p=p)
        algo.run(iter=iter, initial_angles=initial_angles)
        gamma,  beta = algo.angles
        row = {
            "nodes": len(G.nodes),
            "edges": len(G.edges),
            "gamma": gamma,
            "beta": beta,
            "energy": algo.energy,
        }
        for i, value in enumerate(args):
            row[f"arg_{i}"] = value
        row.update(kwargs)
        return row         
            
    def build_dataframe(self, rows):
        self.df = pd.DataFrame(rows)
        
    def save(self, filename):
        header = not os.path.exists(filename)
        self.df.to_csv(filename, mode='a', header=header, index=False)
         
    def load(self, filename):
        self.df = pd.read_csv(filename)
        
    def get_opt_angles(self):
         gammas = self.df["gamma"].to_numpy()
         betas = self.df["beta"].to_numpy()
         return gammas, betas
     
    def get_min_energies(self):
        return self.df["energy"].to_numpy()



