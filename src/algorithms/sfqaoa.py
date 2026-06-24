from algorithms.basealgorithm import BaseAlgorithm
from scipy.integrate import quad
import numpy as np

class ScaleFreeQAOA(BaseAlgorithm):
     def __init__(self, G, p, k_min, alpha):
        super().__init__(p)
        self.G = G   
        self.k_min = k_min
        self.alpha = alpha
        
     def expectation(self, angles):# -> Any:
        exp = 0.0
        gamma = angles[0]
        beta = angles[1]
        z = -np.log(np.cos(gamma))
        
        f = lambda D: np.exp(-z*D) * D**(1-self.alpha)
        I, err = quad(f, self.k_min, np.inf)    
        
        t1 = (
            np.sin(4*beta)
            * np.tan(gamma) 
            * len(self.G.nodes) 
            * (self.alpha - 1)
            * (self.k_min**(self.alpha-1)))
        exp = len(self.G.edges) * 0.5 + 0.25 * t1 * I
        return -exp # -exp if we want to minimize min(-f(x))
