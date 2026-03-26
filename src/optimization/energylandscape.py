import numpy as np
import pandas as pd

class EnergyLandscape:
    def __init__(self, df=None):
        self.df = df
       
    def compute(self, fun, gamma_start=0, gamma_end=np.pi, beta_start=0, beta_end=np.pi/2, n_points=100):
        self.n_points = n_points
        gammas = np.linspace(gamma_start, gamma_end, n_points)
        betas = np.linspace(beta_start, beta_end, n_points)
        data = []
        for gamma in gammas:
            for beta in betas:
                data.append((gamma, beta, fun([gamma, beta])))
        self.df = pd.DataFrame(data, columns=["gamma", "beta", "energy"])
        
    def save(self, filename):
         self.df.to_csv(filename, index=False)
         
    def load(self, filename):
        self.df = pd.read_csv(filename)
        
    def grid(self):
        unique_gammas = np.sort(self.df["gamma"].unique())
        unique_betas = np.sort(self.df["beta"].unique())
        energies2d = self.df["energy"].values.reshape(len(unique_gammas), len(unique_betas))
        return unique_gammas, unique_betas, energies2d
        