from algorithms.basealgorithm import BaseAlgorithm
import numpy as np

class AnalyticalQAOA(BaseAlgorithm):
    def __init__(self, G, p):
        self.G = G
        self.p = p
        
    def expectation(self, angles):
        exp = 0.0
        for u,v in self.G.edges:
            exp += self.expectation_edge(u, v, angles)
        return exp
            
    def compute_triangles_containing_edge(self, u, v):
        neighbors_u = set(self.G.neighbors(u))
        neighbors_v = set(self.G.neighbors(v))
        return len(neighbors_u & neighbors_v)

    def expectation_edge(self, u, v, angles):
        gamma = angles[0]
        beta = angles[1]
        deg_u = self.G.degree(u) - 1
        deg_v = self.G.degree(v) - 1
        lambda_uv = self.compute_triangles_containing_edge(u, v)
        cos_du = np.cos(gamma)**deg_u
        cos_dv = np.cos(gamma)**deg_v
        t1 = (np.sin(4*beta)*np.sin(gamma)) * (cos_du+cos_dv)
        sin_squared = np.sin(2*beta)**2
        cos_dudv = np.cos(gamma)**(deg_u+deg_v-2*lambda_uv) 
        t2 = (sin_squared * cos_dudv) * (1 - np.cos(2*gamma)**lambda_uv)
        exp = 0.5 + 0.25*t1 - 0.25*t2
        return exp
        
    def _postprocess(self, res):
        pass
