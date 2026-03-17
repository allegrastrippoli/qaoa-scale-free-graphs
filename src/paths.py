from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"

def get_run_dirs(run_name):
    run_dir = OUTPUT_DIR / run_name
    csv_dir = run_dir / "csv"
    fig_dir = run_dir / "figures"
    graphs_dir = run_dir / "graphs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, fig_dir, graphs_dir

def csv_graphs_info_path(run_name, index):
    csv_dir, _, _ = get_run_dirs(run_name)
    return csv_dir / f"graphs_info{index}.csv"

def csv_energy_landscape_path(run_name, index):
    csv_dir, _, _ = get_run_dirs(run_name)
    return csv_dir / f"energy_landscape{index}.csv"

def csv_optimized_angles_path(run_name, index):
    csv_dir, _, _ = get_run_dirs(run_name)
    return csv_dir / f"optimized_angles{index}.csv"

def csv_history_path(run_name, index):
    csv_dir, _, _ = get_run_dirs(run_name)
    return csv_dir / f"history{index}.csv"

def fig_energy_landscape_path(run_name, index):
    _, fig_dir, _ = get_run_dirs(run_name)
    return fig_dir / f"energy_landscape{index}.png"

def fig_degree_distribution_path(run_name, index):
    _, fig_dir, _ = get_run_dirs(run_name)
    return fig_dir / f"degree_distribution{index}.png"

def fig_optimized_angles_path(run_name, index):
    _, fig_dir, _ = get_run_dirs(run_name)
    return fig_dir / f"optimized_angles{index}.png"

def fig_full_graph(run_name, index):
    _, fig_dir, _ = get_run_dirs(run_name)
    return fig_dir / f"full_graph{index}.png"

def fig_max_cut(run_name, index):
    _, fig_dir, _ = get_run_dirs(run_name)
    return fig_dir / f"max_cut{index}.png"

def graphs_path(run_name, index):
    _, _, graphs_dir = get_run_dirs(run_name)
    return graphs_dir / f"graph{index}.gml"
