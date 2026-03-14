from abc import ABC, abstractmethod
from scipy.optimize import minimize
import random
import numpy as np

class BaseAlgorithm(ABC): 
    def run(self):
        initial_angles = self._initialize_angles()
        bounds = self._bounds()
        res = minimize(self.expectation, initial_angles, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
        self.angles = res.x
        self._postprocess(res)

    def _initialize_angles(self):
        angles = []
        for i in range(2*self.p):
            if i < self.p:
                angles.append(random.uniform(0,2*np.pi))
            else:
                angles.append(random.uniform(0,np.pi))
        return angles

    def _bounds(self):
        return [(0,2*np.pi+0.1)]*self.p + [(0,np.pi+0.1)]*self.p


    @abstractmethod
    def expectation(self, angles):
        pass
