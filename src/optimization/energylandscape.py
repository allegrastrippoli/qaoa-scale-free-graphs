import numpy as np
import pandas as pd

class EnergyLandscape:
    def __init__(self, df=None):
        self.df = df
    
    def compute(self, fun, n_points=100, **kwargs):
        gamma_start = kwargs.get("gamma_start", 0)
        gamma_end = kwargs.get("gamma_end", 2*np.pi)
        beta_start = kwargs.get("beta_start", 0)
        beta_end = kwargs.get("beta_end", np.pi/2)
        gammas = np.linspace(gamma_start, gamma_end, n_points)
        betas = np.linspace(beta_start, beta_end, n_points)
        data = []
        for gamma in gammas:
            for beta in betas:
                exp = fun([gamma, beta])
                data.append((gamma, beta, exp))
        self.df = pd.DataFrame(data, columns=["gamma", "beta", "energy"])
        
    def save(self, filename):
         self.df.to_csv(filename, index=False)
         
    def load(self, filename):
        self.df = pd.read_csv(filename)
        
    def grid(self):
        gammas = np.sort(self.df["gamma"].unique())
        betas = np.sort(self.df["beta"].unique())
        energies2d = self.df["energy"].values.reshape(len(gammas), len(betas))
        return gammas, betas, energies2d
