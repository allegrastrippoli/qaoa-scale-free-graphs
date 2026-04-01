from algorithms.operators import graph_to_hamiltonian
from algorithms.lcqaoa import LightConesQAOA
from algorithms.rqaoa import RecursiveQAOA
from algorithms.aqaoa import AnalyticalQAOA
from algorithms.qaoa import QAOA
import networkx as nx

class AlgorithmFactory:
    registry = {}

    @classmethod
    def register(cls, algo):
        def decorator(builder):
            cls.registry[algo] = builder
            return builder
        return decorator

    @classmethod
    def create(cls, algo, *args, **kwargs):
        if algo not in cls.registry:
            raise ValueError(f"Unknown algorithm {algo}")
        return cls.registry[algo](*args, **kwargs)
    
@AlgorithmFactory.register("qaoa")
def build_qaoa(G, p, **kwargs):
    H = graph_to_hamiltonian(nx.to_numpy_array(G), len(G.nodes))
    return QAOA(depth=p, H=H)

@AlgorithmFactory.register("rqaoa")
def build_rqaoa(G, p, **kwargs):
    H = graph_to_hamiltonian(nx.to_numpy_array(G), len(G.nodes))
    Q = QAOA(depth=p, H=H)
    return RecursiveQAOA(depth=p, H=H, Q=Q, G=G)

@AlgorithmFactory.register("lcqaoa")
def build_lcqaoa(G, p, **kwargs):
    return LightConesQAOA(G=G, p=p, **kwargs)

@AlgorithmFactory.register("aqaoa")
def build_lcqaoa(G, p, **kwargs):
    return AnalyticalQAOA(G=G, p=p, **kwargs)
