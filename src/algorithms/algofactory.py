from algorithms.operators import graph_to_hamiltonian
from algorithms.lcqaoa import LCQAOA
from algorithms.rqaoa import RQAOA
from algorithms.qaoa import QAOA
import networkx as nx

class AlgorithmFactory:
    registry = {}

    @classmethod
    def register(cls, name):
        def decorator(builder):
            cls.registry[name] = builder
            return builder
        return decorator

    @classmethod
    def create(cls, name, *args, **kwargs):
        if name not in cls.registry:
            raise ValueError(f"Unknown algorithm {name}")
        return cls.registry[name](*args, **kwargs)
    
@AlgorithmFactory.register("qaoa")
def build_qaoa(G, p, **kwargs):
    H = graph_to_hamiltonian(nx.to_numpy_array(G), len(G.nodes))
    return QAOA(depth=p, H=H)

@AlgorithmFactory.register("rqaoa")
def build_rqaoa(G, p, **kwargs):
    H = graph_to_hamiltonian(nx.to_numpy_array(G), len(G.nodes))
    Q = QAOA(depth=p, H=H)
    return RQAOA(depth=p, H=H, Q=Q, G=G)

@AlgorithmFactory.register("lcqaoa")
def build_lcqaoa(G, p, **kwargs):
    return LCQAOA(G=G, p=p, **kwargs)
