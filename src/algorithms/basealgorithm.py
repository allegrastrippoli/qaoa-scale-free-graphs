from abc import ABC, abstractmethod
from scipy.optimize import minimize
import numpy as np

class BaseAlgorithm(ABC):
    def __init__(self, p):
        self.p = p
        self.angles = None
        self.energy = None
        
    def run(self, multistart_iter=0, initial_angles=None):
        bounds = self._bounds()
        if multistart_iter > 0:
            best_val = np.inf
            for _ in range(multistart_iter):
                initial_angles = self.initialize_angles()
                res = minimize(self.expectation, initial_angles, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
                val = res.fun
                if val < best_val:
                    best_val = val
        else:
            if initial_angles is None:
                initial_angles = self.initialize_angles()
            res = minimize(self.expectation, initial_angles, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
        self._postprocess(res)

    def _bounds(self):
        return [(0, np.pi)]*self.p + [(0,np.pi/2)]*self.p
    
    def initialize_angles(self):
        gammas = np.random.uniform(0, np.pi, size=self.p)
        betas = np.random.uniform(0, np.pi / 2, size=self.p)
        return np.concatenate([gammas, betas])

    @abstractmethod
    def expectation(self, angles):
        pass
