from pathlib import Path
from enum import Enum

class Category(str, Enum):
    GRAPHS_INFO = "graphs_info"
    ENERGY_LANDSCAPE = "energy_landscape"
    OPTIMIZED_ANGLES = "optimized_angles"
    HISTORY = "history"
    DEGREE_DISTRIBUTION = "degree_distribution"
    FULL_GRAPH = "full_graph"
    MAX_CUT = "max_cut"
    GRAPH = "graph"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

class RunPaths:
    def __init__(self, run_name):
        self.base = OUTPUT_DIR / run_name
        self.dirs = {
            "log": self.base / "log",
            "fig": self.base / "figures",
            "graphs": self.base / "graphs",
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
    def _name(self, category: Category, index=None):
        if not isinstance(category, Category):
            raise TypeError("category must be a Category enum")
        return f"{category.value}{index}" if index is not None else category.value

    def log(self, category: Category, index=None):
        return self.dirs["log"] / f"{self._name(category, index)}.csv"

    def fig(self, category: Category, index=None):
        return self.dirs["fig"] / f"{self._name(category, index)}.png"

    def graphs(self, category: Category, index):
        return self.dirs["graphs"] / f"{self._name(category, index)}.gml"
