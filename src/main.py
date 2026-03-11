from utils.utils import *
from utils.generate import *
from utils.plots import  *
from tests.test_qmodels import test_energy_landscape_regular_graph
from qmodels.rqaoa import RQAOA
from qmodels.qaoa import QAOA
import networkx as nx
import numpy as np

if __name__ == "__main__":
    test_energy_landscape_regular_graph(True)
    test_energy_landscape_regular_graph(False)
    

