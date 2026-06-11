from tests.test_example import *
from utils.generate import generate_scale_free
from utils.plots import plot_analytical_vs_numerical
import json

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    # test_qaoa()
    # test_lcqaoa()
    # test_regular_graphs(**config)
    # run_example_energy_landscape(**config)
    
    run_name = "test_optimized_angles"
    rp = RunPaths(run_name)
    for i in range(1,6):
         test_optimized_angles(rp=rp, start_n=800, end_n=1001, step=100, fun=generate_scale_free, scaling_values=[2, 2.25, 2.50, 2.75, 3], k_min=i, strictlyEnforceMinimumDegree=True, index=i)
    filename = rp.log(category=Category.OPTIMIZED_ANGLES)
    plot_analytical_vs_numerical(rp=rp, filename=filename)
    
    # run_name = "test_triangles_distribution"
    # rp = RunPaths(run_name)
    # for i in range(1,6):
    #     run_triangle_example(rp=rp, start_n=800, end_n=1001, step=20, fun=generate_scale_free, scaling_values=[2, 2.25, 2.50, 2.75, 3], k_min=i, index=i, strictlyEnforceMinimumDegree=True)
