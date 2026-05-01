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
    _validate_inputs(G, p)
    H = graph_to_hamiltonian(nx.to_numpy_array(G), len(G.nodes))
    return QAOA(p=p, H=H)

@AlgorithmFactory.register("rqaoa")
def build_rqaoa(G, p, **kwargs):
    _validate_inputs(G, p)
    H = graph_to_hamiltonian(nx.to_numpy_array(G), len(G.nodes))
    Q = QAOA(p=p, H=H)
    return RecursiveQAOA(p=p, H=H, Q=Q, G=G)

@AlgorithmFactory.register("lcqaoa")
def build_lcqaoa(G, p, **kwargs):
    _validate_inputs(G, p)
    return LightConesQAOA(G=G, p=p, **kwargs)

@AlgorithmFactory.register("aqaoa")
def build_aqaoa(G, p, **kwargs):
    _validate_inputs(G, p)
    return AnalyticalQAOA(G=G, p=p, **kwargs)

def _validate_inputs(G, p):
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx.Graph")

    if G.number_of_nodes() < 2:
        raise ValueError("G must have at least two nodes")

    if G.number_of_edges() < 1:
        raise ValueError("G must have at least one edge")

    if not isinstance(p, int) or p <= 0:
        raise ValueError("p must be a positive integer")
