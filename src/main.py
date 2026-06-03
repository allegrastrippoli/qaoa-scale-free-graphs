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
    test_optimized_angles(run_name=run_name, start_n=100, end_n=200, step=50, fun=generate_scale_free, scaling_values=[ 3, 4], k_min=5, strictlyEnforceMinimumDegree=True)
