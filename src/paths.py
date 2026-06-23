from pathlib import Path
from enum import Enum
import shutil

class OutputFile(str, Enum):
    GRAPHS_INFO = "graphs_info"
    ENERGY_LANDSCAPE = "energy_landscape"
    DEGREE_DISTRIBUTION = "degree_distribution/degree_distribution"
    MAX_CUT = "max_cut"
    GRAPH = "graph"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

class RunPaths:
    def __init__(self, run_name):
        self.base = OUTPUT_DIR / run_name
        if self.base.exists():
            shutil.rmtree(self.base)
        self.base.mkdir(parents=True, exist_ok=True)
        self.dirs = {
            "log": self.base / "log",
            "fig": self.base / "figures",
            "graphs": self.base / "graphs",
            "metrics": self.base / "metrics"
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        (self.base / "figures" / "degree_distribution").mkdir(parents=True, exist_ok=True)  
            
    def _name(self, output_file: OutputFile, index=None):
        if not isinstance(output_file, OutputFile):
            raise TypeError("OutputFile must be a OutputFile enum")
        return (f"{output_file.value}{index}" if index is not None else output_file.value)

    def log(self, output_file: OutputFile, index=None):
        return self.dirs["log"] / f"{self._name(output_file, index)}.csv"

    def fig(self, output_file: OutputFile, index=None):
        return self.dirs["fig"] / f"{self._name(output_file, index)}.png"

    def graphs(self, output_file: OutputFile, index):
        return self.dirs["graphs"] / f"{self._name(output_file, index)}.gml"
