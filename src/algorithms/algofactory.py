from algorithms.operators import graph_to_hamiltonian
from algorithms.lcqaoa import LightConesQAOA
from algorithms.rqaoa import RecursiveQAOA
from algorithms.aqaoa import AnalyticalQAOA
from algorithms.sfqaoa import ScaleFreeQAOA
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
    def create(cls, algo, G, p, **kwargs):
        if algo not in cls.registry:
            raise ValueError(f"Unknown algorithm {algo}")
        return cls.registry[algo](G, p, **kwargs)
    
@AlgorithmFactory.register("qaoa")
def build_qaoa(G, p):
    _validate_inputs(G, p)
    H = graph_to_hamiltonian(nx.to_numpy_array(G), len(G.nodes))
    return QAOA(p, H)

@AlgorithmFactory.register("rqaoa")
def build_rqaoa(G, p):
    _validate_inputs(G, p)
    H = graph_to_hamiltonian(nx.to_numpy_array(G), len(G.nodes))
    Q = QAOA(p, H)
    return RecursiveQAOA(p, H, Q, G)

@AlgorithmFactory.register("lcqaoa")
def build_lcqaoa(G, p, edges_subset=None):
    _validate_inputs(G, p)
    return LightConesQAOA(G, p, edges_subset)

@AlgorithmFactory.register("aqaoa")
def build_aqaoa(G, p):
    _validate_inputs(G, p)
    return AnalyticalQAOA(G, p)

@AlgorithmFactory.register("sfqaoa")
def build_sfqaoa(G, p, k_min, alpha):
    _validate_inputs(G, p)
    return ScaleFreeQAOA(G, p, k_min, alpha)

def _validate_inputs(G, p):
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx.Graph")
    if G.number_of_nodes() < 2:
        raise ValueError("G must have at least two nodes")
    if G.number_of_edges() < 1:
        raise ValueError("G must have at least one edge")
    if not isinstance(p, int) or p <= 0:
        raise ValueError("p must be a positive integer")
