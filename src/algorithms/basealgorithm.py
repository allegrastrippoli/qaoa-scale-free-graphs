from abc import ABC, abstractmethod
from scipy.optimize import minimize
from utils.utils import initialize_angles
import numpy as np

class BaseAlgorithm(ABC): 
    def run(self, initial_angles=[]):
        if not initial_angles:
            initial_angles = initialize_angles(self.p)
        bounds = self._bounds()
        res = minimize(self.expectation, initial_angles, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
        self.angles = res.x
        self._postprocess(res)

    def _bounds(self):
        return [(0,2*np.pi+0.1)]*self.p + [(0,np.pi+0.1)]*self.p

    @abstractmethod
    def expectation(self, angles):
        pass
