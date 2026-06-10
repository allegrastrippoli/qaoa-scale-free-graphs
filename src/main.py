from tests.test_example import *
from utils.generate import generate_scale_free
import json

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    # test_qaoa()
    # test_lcqaoa()
    # test_regular_graphs(**config)
    # run_example_energy_landscape(**config)
    
    # scaling_values=[2, 2.25, 2.50, 2.75, 3, 3.25, 3.50, 3.75, 4]

    run_name = "test_optimized_angles"
    test_optimized_angles(run_name=run_name, start_n=80, end_n=101, step=100, fun=generate_scale_free, scaling_values=[2, 2.25, 2.50, 2.75, 3], k_min=4, strictlyEnforceMinimumDegree=True)

    # run_name = "test_triangles_distribution"
    # rp = RunPaths(run_name)
    # for i in range(1,6):
    #     run_triangle_example(rp=rp, start_n=800, end_n=1001, step=20, fun=generate_scale_free, scaling_values=[2, 2.25, 2.50, 2.75, 3], k_min=i, index=i, strictlyEnforceMinimumDegree=True)
