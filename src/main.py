from utils.utils import *
from utils.plots import  *
from utils.generate import *
from tests.test_qmodels import *
from qmodels.rqaoa import RQAOA
from qmodels.qaoa import QAOA

if __name__ == "__main__":
    # test_energy_landscape_regular_graph(True)
    # test_energy_landscape_regular_graph(False)
    # test_scale_free_graph()
    generate_graphs(30, 50, 2.4)
    p = 1
    graphs = load_all_graphs()
    compute_optimized_angles(graphs, p)
    gammas, betas = load_gamma_beta()
    plot_optimized_angles(gammas, betas)
        

