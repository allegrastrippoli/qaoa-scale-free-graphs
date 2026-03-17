from abc import ABC, abstractmethod
from scipy.optimize import minimize
from utils.utils import initialize_angles
import numpy as np

class BaseAlgorithm(ABC):
    def run(self, n_iter=0, multistart=False, initial_angles=None):
        bounds = self._bounds()
        if multistart and n_iter > 0:
            print(f"Multistart init, {n_iter} iterations")
            best_val = np.inf
            for _ in range(n_iter):
                initial_angles = initialize_angles(self.p)
                res = minimize(self.expectation, initial_angles, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
                val = res.fun
                if val < best_val:
                    best_val = val
                    self.angles = res.x
        else:
            if initial_angles is None:
                initial_angles = initialize_angles(self.p)
            res = minimize(self.expectation, initial_angles, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
            self.angles = res.x
        self._postprocess(res)

    def _bounds(self):
        return [(0,2*np.pi+0.1)]*self.p + [(0,np.pi+0.1)]*self.p

    @abstractmethod
    def expectation(self, angles):
        pass
