from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"
CSV_DIR = OUTPUT_DIR / "csv"
FIG_DIR = OUTPUT_DIR / "figures"
GRAPHS_DIR =  OUTPUT_DIR / "graphs"

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def csv_energy_landscape_path(name):
    return CSV_DIR / "energy_landscape" / f"{name}.csv"

def csv_optimized_angles_path(name):
    return CSV_DIR / "optimized_angles" / f"{name}.csv"

def fig_energy_landscape_path(name):
    return FIG_DIR / "energy_landscape" / f"{name}.png"

def fig_optimized_angles_path(name):
    return FIG_DIR / "optimized_angles" / f"{name}.png"

def graphs_path(name):
    return GRAPHS_DIR / f"{name}.gml"
