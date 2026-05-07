from algorithms.algofactory import AlgorithmFactory
import numpy as np
import pandas as pd

class OptimizedAngles:
    def __init__(self, df=None, algo=None):
        self.df = df
        
    def compute(self, G, p, algo_name, initial_angles=None, iter=0):
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
        return row
            
    def build_dataframe(self, rows):
        self.df = pd.DataFrame(rows)
        
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



