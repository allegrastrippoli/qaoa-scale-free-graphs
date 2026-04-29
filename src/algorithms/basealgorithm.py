from abc import ABC, abstractmethod
from scipy.optimize import minimize
import numpy as np

class BaseAlgorithm(ABC):
    def __init__(self, p):
        self.p = p
        self.angles = None
        self.energy = None
        
    def run(self, iter=1, initial_angles=None):
        bounds = self._bounds()
        if iter > 0:
            best_val = np.inf
            for _ in range(iter):
                initial_angles = self.initialize_angles()
                res = minimize(self.expectation, initial_angles, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
                val = res.fun
                if val < best_val:
                    best_val = val
                    self.angles = res.x     
                    self.energy = res.fun 
            self._postprocess(res)
        else:
            raise ValueError("iter must be > 0")

    def _bounds(self):
        return [(0, np.pi)]*self.p + [(0,np.pi/2)]*self.p
    
    def initialize_angles(self):
        gammas = np.random.uniform(0, np.pi, size=self.p)
        betas = np.random.uniform(0, np.pi / 2, size=self.p)
        return np.concatenate([gammas, betas])

    @abstractmethod
    def expectation(self, angles):
        pass
