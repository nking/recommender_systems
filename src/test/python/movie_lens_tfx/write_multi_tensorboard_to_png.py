import glob
import json
import math
import os
import unittest
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util

from find_pushed_trainer_uri import get_pushed_trainer_uri
from helper import *

class WriteMultiTensorboardToPng(unittest.TestCase):
    """
    Generates single-page multi-run comparison PNGs for *ndcg*, *recal*, and *loss* metrics,
    plus a unified `hyperparameters_table.png` comparing numeric hyperparameters across all runs.
    """
    
    output_dir = os.path.join(get_bin_dir(), "multi_tensorboard_pngs")
    
    direcs = ["../TMP40", "../TMP41", "../TMP42", "../TMP43",     "../TMP44",
        "../TMP45"]
    titles = ["DNN[8,4]", "DNN[16,8]", "DNN[32,16]", "DNN[64, 32]", "DNN[128,64]",
        "DNN[256,128]"]
    
    #direcs = ["../TMP02", "../TMP01", "../TMP03"]
    #titles = ["DNN[16,8]", "DNN[64, 32]", "DNN[128,64]"]
    
    runs_list: List[Dict[str, str]] = []
    
    for i, direc in enumerate(direcs):
        rel_path = os.path.join(get_project_dir(), direc)
        direc = os.path.realpath(rel_path)
        direc = os.path.join(direc, "bin/rs_pipeline")
        os.environ["PIPELINE_DATA_DIR"] = direc
        saved_model_dir, trainer_uri = get_pushed_trainer_uri()
        if trainer_uri.find("model_run") == -1:
            trainer_uri = trainer_uri.replace("model", 'model_run')
        logdir = trainer_uri
        
        runs_list.append({
            "saved_model_dir": saved_model_dir,
            "logdir": logdir,
            "title": titles[i]
        })

    def test_write_multi_tensorboard_to_png(self):
        if not self.runs_list:
            print("WriteMultiTensorboardToPng.runs_list is empty. Nothing to plot.")
            return
        self.write_multi_runs_to_png(self.runs_list, self.output_dir)

    @classmethod
    def write_multi_runs_to_png(
        cls,
        runs: List[Dict[str, str]],
        outdir: str,
        smoothing_weight: float = 0.6,
        panel_size: Tuple[float, float] = (1.95, 1.95),
        max_cols: int = 3,
        runs_as_rows_in_hyperparams_table: bool = False,
    ) -> None:
        """
        Takes a list of dictionaries where each dictionary has:
            - 'saved_model_dir': path to SavedModel directory
            - 'logdir': path to TensorBoard log directory (containing train/validation/test)
            - 'title': display title for the run
        Generates:
            1. `hyperparameters_table.png` (and `.json`) comparing all numeric hyperparameters
               ordered by `runs`.
            2. One PNG page per metric (filtered to *ndcg*, *recal*, *loss*).
        """
        os.makedirs(outdir, exist_ok=True)

        prepared_runs = []
        runs_hyperparams: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        all_metrics = []

        for item in runs:
            saved_model_dir = item["saved_model_dir"]
            logdir = item.get("logdir") or item.get("log_dir")
            title = item["title"]

            train_dir = os.path.join(logdir, "train")
            val_dir = os.path.join(logdir, "validation")
            test_dir = os.path.join(logdir, "test")

            hyperparams_dict = cls.get_hyperparams_dict(saved_model_dir)
            runs_hyperparams.append((title, hyperparams_dict))

            irred_err_dict = cls.get_irred_error_dict(saved_model_dir)

            random_ndcg = None
            random_recall = None
            random_loss = None
            if hyperparams_dict is not None:
                k = hyperparams_dict["k"]
                n_movies = hyperparams_dict["n_movies"]
                batch_size = hyperparams_dict["BATCH_SIZE"]
                random_ndcg = cls.calc_random_ndcg(k=k, movie_catalog_size=n_movies)
                random_recall = cls.calc_random_recall(k=k, movie_catalog_size=n_movies)
                random_loss = cls.calc_random_inbatch_softmax_loss(batch_size)

            run_metrics = list_tfevents_metrics(train_dir)
            for m in run_metrics:
                if is_target_metric(m) and m not in all_metrics:
                    all_metrics.append(m)

            prepared_runs.append(
                {
                    "title": title,
                    "saved_model_dir": saved_model_dir,
                    "logdir": logdir,
                    "train_dir": train_dir,
                    "val_dir": val_dir,
                    "test_dir": test_dir,
                    "irred_err_dict": irred_err_dict,
                    "random_ndcg": random_ndcg,
                    "random_recall": random_recall,
                    "random_loss": random_loss,
                }
            )

        # 1. Generate the unified Numeric Hyperparameters Table PNG in outdir
        hp_table_outfile = os.path.join(outdir, "hyperparameters_table.png")
        generate_hyperparams_table_png(
            runs_hyperparams=runs_hyperparams,
            output_path=hp_table_outfile,
            runs_as_rows=runs_as_rows_in_hyperparams_table,
        )
        print(f"Wrote hyperparameters comparison table to {hp_table_outfile}", flush=True)

        # 2. Generate per-metric multi-run comparison pages
        print(f"Filtered target metrics (*ndcg*, *recal*, *loss*): {all_metrics}", flush=True)

        for metric in all_metrics:
            outfile = os.path.join(outdir, f"{metric}.png")
            runs_for_metric = []

            for r in prepared_runs:
                irr_dict = cls.find_irred_dict(r["irred_err_dict"], metric)
                random_metric = None
                m_lower = metric.lower()

                if "recal" in m_lower:
                    irr_dict = None
                    random_metric = r["random_recall"]
                elif "ndcg" in m_lower:
                    random_metric = r["random_ndcg"]
                elif "loss" in m_lower:
                    random_metric = r["random_loss"]

                runs_for_metric.append(
                    {
                        "title": r["title"],
                        "train_dir": r["train_dir"],
                        "val_dir": r["val_dir"],
                        "test_dir": r["test_dir"],
                        "irred_err_dict": irr_dict,
                        "random_metric": random_metric,
                    }
                )

            generate_multi_run_metric_page(
                runs_plot_data=runs_for_metric,
                scalar_name=metric,
                output_path=outfile,
                smoothing_weight=smoothing_weight,
                panel_size=panel_size,
                max_cols=max_cols,
            )

        print(f"Wrote multi-run metric PNG pages to {outdir}", flush=True)

    @staticmethod
    def get_irred_error_dict(saved_model_dir: str) -> Optional[Dict]:
        file_path = os.path.join(
            saved_model_dir, "assets.extra", "irreducible_error.json"
        )
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return None

    @staticmethod
    def get_hyperparams_dict(saved_model_dir: str) -> Optional[Dict]:
        file_path = os.path.join(
            saved_model_dir, "assets.extra", "hyperparameters.json"
        )
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                h_dict = json.load(f)
                return h_dict["values"]
        return None

    @staticmethod
    def calc_random_inbatch_softmax_loss(batch_size: int) -> float:
        return float(np.log(batch_size))

    @staticmethod
    def calc_random_ndcg(k: int, movie_catalog_size: int) -> float:
        s = np.sum([1.0 / np.log2(r + 1) for r in range(1, k + 1)])
        return (s / movie_catalog_size).item()

    @staticmethod
    def calc_random_recall(k: int, movie_catalog_size: int) -> float:
        return k / movie_catalog_size

    @staticmethod
    def calc_random_precision(k: int, movie_catalog_size: int) -> float:
        return 1.0 / movie_catalog_size

    @staticmethod
    def calc_random_mrr(k: int, movie_catalog_size: int) -> float:
        s = np.sum([1.0 / r for r in range(1, k + 1)])
        return (s / movie_catalog_size).item()

    @staticmethod
    def find_irred_dict(
        irred_err_dict: Optional[Dict], metric_name: str
    ) -> Optional[Dict]:
        if irred_err_dict is None:
            return None
        clean_name = metric_name.replace("epoch_", "")
        return irred_err_dict.get(clean_name, None)


def get_latest_scalars(
    log_dir: str, scalar_name: str
) -> Tuple[Optional[List[int]], Optional[List[float]], Optional[List[float]]]:
    """
    Extracts steps, values, and wall_times for a given scalar from TensorBoard logs.
    Compatible with TensorBoard 2.21.0 (supports both TF1 scalars and TF2 rank-0 tensors).
    """
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None, None, None

    event_file = max(event_files, key=os.path.getctime)

    try:
        ea = event_accumulator.EventAccumulator(
            event_file,
            size_guidance={"scalars": 0, "tensors": 0},
        )
        ea.Reload()
    except Exception as e:
        print(f"Error loading event file {event_file}: {e}")
        return None, None, None

    available_tags = ea.Tags()
    steps, values, wall_times = [], [], []

    if scalar_name in available_tags.get("scalars", []):
        events = ea.Scalars(scalar_name)
        steps = [e.step for e in events]
        values = [float(e.value) for e in events]
        wall_times = [e.wall_time for e in events]
    elif scalar_name in available_tags.get("tensors", []):
        events = ea.Tensors(scalar_name)
        steps = [e.step for e in events]
        values = [float(tensor_util.make_ndarray(e.tensor_proto)) for e in events]
        wall_times = [e.wall_time for e in events]
    else:
        print(f"Metric '{scalar_name}' not found in {log_dir}.")
        return None, None, None

    return steps, values, wall_times


def exponential_smoothing(scalars: List[float], weight: float = 0.6) -> List[float]:
    """
    Applies exponential smoothing to a list of scalar values.
    """
    smoothed = []
    if not scalars:
        return smoothed
    last = scalars[0]
    smoothed.append(last)
    for current in scalars[1:]:
        smoothed_val = (1.0 - weight) * last + weight * current
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def list_tfevents_metrics(log_dir: str) -> List[str]:
    """
    Lists all scalar and tensor metric tags in a TensorBoard log directory.
    """
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            "scalars": 0,
            "tensors": 0,
            "images": 1,
            "audio": 1,
            "histograms": 1,
        },
    )
    ea.Reload()
    tags = ea.Tags()

    outlist = []
    for tag in tags.get("scalars", []):
        if tag not in outlist:
            outlist.append(tag)
    for tag in tags.get("tensors", []):
        if tag not in outlist:
            outlist.append(tag)

    return outlist


def is_target_metric(metric_name: str) -> bool:
    """
    Filters metrics to only include *ndcg*, *recal* (including recall), and *loss*.
    """
    m = metric_name.lower()
    return ("ndcg" in m) or ("recal" in m) or ("loss" in m)


def is_numeric_hyperparam(val: Any) -> bool:
    """
    Returns True if `val` is a numeric scalar (int/float, excluding bool)
    or a non-empty list/tuple of numeric scalars (e.g., layer sizes [64, 32]).
    Excludes strings, URLs, paths, dicts, None, and booleans.
    """
    if isinstance(val, bool) or val is None:
        return False
    if isinstance(val, (int, float, np.number)):
        return True
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return all(
            isinstance(x, (int, float, np.number)) and not isinstance(x, bool)
            for x in val
        )
    return False


def format_hyperparam_value(val: Any) -> str:
    """
    Formats a numeric hyperparameter cleanly for table display.
    """
    if val is None:
        return "-"
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(format_hyperparam_value(x) for x in val) + "]"
    if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
        return f"{int(val)}"
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return str(val)
        if val.is_integer() and abs(val) < 1e12:
            return f"{int(val)}"
        return f"{val:.6g}"
    return str(val)


def generate_hyperparams_table_png(
    runs_hyperparams: List[Tuple[str, Optional[Dict[str, Any]]]],
    output_path: str,
    runs_as_rows: bool = False,
) -> None:
    """
    Generates a single PNG table comparing all numeric hyperparameters across runs,
    ordered by the input `runs` list. Rows where hyperparameter values differ across
    runs are subtly highlighted for quick visual comparison.
    """
    numeric_keys: List[str] = []
    filtered_run_dicts: List[Tuple[str, Dict[str, Any]]] = []

    for title, h_dict in runs_hyperparams:
        clean_dict: Dict[str, Any] = {}
        if isinstance(h_dict, dict):
            for k, v in h_dict.items():
                if is_numeric_hyperparam(v):
                    clean_dict[k] = v
                    if k not in numeric_keys:
                        numeric_keys.append(k)
        filtered_run_dicts.append((title, clean_dict))

    if not numeric_keys or not filtered_run_dicts:
        print("No numeric hyperparameters found across runs. Skipping hyperparameters table.")
        return

    # Sort hyperparameter names alphabetically (case-insensitive)
    numeric_keys = sorted(numeric_keys, key=lambda s: s.lower())
    run_titles = [t for t, _ in filtered_run_dicts]

    if not runs_as_rows:
        # Default: Columns = Runs (in run order), Rows = Numeric Hyperparameters
        col_labels = run_titles
        row_labels = numeric_keys
        cell_text = []
        row_differs = []

        for k in numeric_keys:
            row_vals = [d.get(k, None) for _, d in filtered_run_dicts]
            formatted_row = [format_hyperparam_value(v) for v in row_vals]
            cell_text.append(formatted_row)
            row_differs.append(len(set(formatted_row)) > 1)

        n_rows = len(row_labels)
        n_cols = len(col_labels)
        fig_width = max(3.2, 1.6 + 1.15 * n_cols)
        fig_height = max(1.8, 0.55 + 0.24 * (n_rows + 1))
        bbox = [0.28, 0.0, 0.72, 0.92]
    else:
        # Alternative: Rows = Runs (in run order), Columns = Numeric Hyperparameters
        row_labels = run_titles
        col_labels = numeric_keys
        cell_text = []
        for _, d in filtered_run_dicts:
            cell_text.append(
                [format_hyperparam_value(d.get(k, None)) for k in numeric_keys]
            )
        row_differs = [False] * len(row_labels)

        n_rows = len(row_labels)
        n_cols = len(col_labels)
        fig_width = max(4.0, 1.2 + 0.95 * n_cols)
        fig_height = max(1.6, 0.55 + 0.26 * (n_rows + 1))
        bbox = [0.18, 0.0, 0.82, 0.90]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.axis("off")
    ax.set_title(
        "Numeric Hyperparameters Comparison",
        loc="left",
        fontsize=8.0,
        fontweight="bold",
        pad=6,
    )

    the_table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=bbox,
    )

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(5.5)

    for (i, j), cell in the_table.get_celld().items():
        cell.set_edgecolor("#e5e7eb")
        cell.set_linewidth(0.6)
        if i == 0:
            # Header row
            cell.set_text_props(weight="bold", color="#1f2937")
            cell.set_facecolor("#f3f4f6")
        elif j == -1:
            # Row label cell
            is_diff = row_differs[i - 1] if not runs_as_rows else False
            cell.set_text_props(
                ha="left",
                weight="bold" if is_diff else "medium",
                color="#111827" if is_diff else "#374151",
            )
            cell.set_facecolor("#fffbeb" if is_diff else "#f9fafb")
        else:
            is_diff = row_differs[i - 1] if not runs_as_rows else False
            if is_diff:
                # Subtle highlight for rows where hyperparameters vary across runs
                cell.set_facecolor("#fffbeb")
                cell.set_text_props(weight="bold", color="#111827")
            else:
                cell.set_facecolor("#ffffff" if i % 2 == 1 else "#f9fafb")
                cell.set_text_props(color="#374151")

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    output_json_path = output_path.replace(".png", ".json")
    json_out = {
        "runs_order": run_titles,
        "numeric_hyperparameters": {
            k: {title: d.get(k, None) for title, d in filtered_run_dicts}
            for k in numeric_keys
        },
    }
    with open(output_json_path, "w") as f:
        json.dump(json_out, f, indent=4)


def generate_multi_run_metric_page(
    runs_plot_data: List[Dict],
    scalar_name: str,
    output_path: str,
    smoothing_weight: float = 0.6,
    panel_size: Tuple[float, float] = (1.95, 1.95),
    max_cols: int = 3,
) -> None:
    """
    Generates a single PNG page for `scalar_name` containing:
      1. A smaller chart + summary table for each item in `runs_plot_data`,
         titled with the item's `title` and locked to the shared x/y axis ranges.
      2. An overplotted comparison chart on the same page with the exact same
         x-axis and y-axis ranges (solid lines for validation, dashed lines for train).
    """
    is_loss = "loss" in scalar_name.lower()
    bound_label = "irred error" if is_loss else "ceiling"

    def get_relative_times(times: List[float]) -> List[float]:
        if not times:
            return []
        start_time = times[0]
        return [(t - start_time) / 60.0 for t in times]

    valid_runs = []
    all_x = []
    all_y = []

    for item in runs_plot_data:
        train_dir = item["train_dir"]
        val_dir = item["val_dir"]
        test_dir = item["test_dir"]
        title = item["title"]

        train_steps, train_values, train_times = get_latest_scalars(train_dir, scalar_name)
        val_steps, val_values, val_times = get_latest_scalars(val_dir, scalar_name)
        test_steps, test_values, test_times = get_latest_scalars(
            test_dir, scalar_name.replace("epoch_", "")
        )

        if not train_steps or not val_steps or not test_steps:
            print(f"Skipping '{title}' for metric '{scalar_name}': missing train/val/test data.")
            continue

        train_smoothed = exponential_smoothing(train_values, smoothing_weight)
        val_smoothed = exponential_smoothing(val_values, smoothing_weight)
        test_smoothed = exponential_smoothing(test_values, smoothing_weight)

        run_entry = {
            "title": title,
            "train_steps": train_steps,
            "train_values": train_values,
            "train_smoothed": train_smoothed,
            "train_relative": get_relative_times(train_times),
            "val_steps": val_steps,
            "val_values": val_values,
            "val_smoothed": val_smoothed,
            "val_relative": get_relative_times(val_times),
            "test_steps": test_steps,
            "test_values": test_values,
            "test_smoothed": test_smoothed,
            "test_relative": get_relative_times(test_times),
            "irred_err_dict": item.get("irred_err_dict"),
            "random_metric": item.get("random_metric"),
        }
        valid_runs.append(run_entry)

        # Collect global x and y values across all runs for shared axes
        all_x.extend(train_steps)
        all_x.extend(val_steps)
        all_x.append(test_steps[-1])

        all_y.extend(train_values)
        all_y.extend(val_values)
        all_y.extend(train_smoothed)
        all_y.extend(val_smoothed)
        all_y.append(test_values[-1])

    if not valid_runs:
        print(f"No valid runs found for metric '{scalar_name}'. Skipping page generation.")
        return

    # Compute shared x-axis and y-axis ranges across all runs
    x_min, x_max = min(all_x), max(all_x)
    x_padding = max((x_max - x_min) * 0.04, 0.5)
    shared_xlim = (x_min - x_padding, x_max + x_padding)

    y_min, y_max = min(all_y), max(all_y)
    y_padding = max((y_max - y_min) * 0.08, 0.01)
    shared_ylim = (y_min - y_padding, y_max + y_padding)

    # Total cells on page = N individual run panels + 1 overplotted panel
    total_panels = len(valid_runs) + 1
    ncols = min(total_panels, max_cols)
    nrows = math.ceil(total_panels / ncols)

    fig_width = ncols * panel_size[0]
    fig_height = nrows * panel_size[1] + 0.35
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)

    titles_summary = " | ".join(r["title"] for r in valid_runs)
    fig.suptitle(
        f"{scalar_name}  ({titles_summary})",
        fontsize=7.5,
        fontweight="bold",
        y=0.99,
    )

    outer_gs = GridSpec(
        nrows,
        ncols,
        figure=fig,
        wspace=0.30,
        hspace=0.32,
        top=0.91,
        bottom=0.03,
        left=0.04,
        right=0.98,
    )

    # Colors for individual panels (matching original generate_tensorboard_chart)
    train_color = "#1f77b4"        # Blue
    train_color_faded = "#aec7e8"  # Faded Blue
    val_color = "#ff7f0e"          # Orange
    val_color_faded = "#ffbb78"    # Faded Orange
    test_color = "#d62728"         # Red
    bound_color = "#2ca02c"        # Green
    random_color = "#000000"       # Black

    out_summary_dict = {
        "metric": scalar_name,
        "shared_xlim": list(shared_xlim),
        "shared_ylim": list(shared_ylim),
        "runs": {},
    }

    # 1. Plot each individual run panel (smaller size, shared xlim & ylim)
    for idx, r in enumerate(valid_runs):
        row_idx = idx // ncols
        col_idx = idx % ncols
        inner_gs = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer_gs[row_idx, col_idx],
            height_ratios=[2.0, 1.2],
            hspace=0.25,
        )

        ax = fig.add_subplot(inner_gs[0])
        ax_table = fig.add_subplot(inner_gs[1])
        ax_table.axis("off")

        irred_err_dict = r["irred_err_dict"]
        random_metric = r["random_metric"]

        bound_val = None
        bound_moe = None
        if irred_err_dict is not None:
            bound_val = (
                irred_err_dict.get("irred_error")
                if is_loss
                else irred_err_dict.get("ceiling")
            )
            bound_moe = irred_err_dict.get("margin_of_error_on_irred_err")

        # Raw (faded) and smoothed (solid) lines
        ax.plot(
            r["train_steps"],
            r["train_values"],
            color=train_color_faded,
            alpha=0.7,
            linewidth=0.5,
            linestyle="-",
        )
        ax.plot(
            r["val_steps"],
            r["val_values"],
            color=val_color_faded,
            alpha=0.7,
            linewidth=0.5,
            linestyle="-",
        )
        ax.plot(
            r["train_steps"],
            r["train_smoothed"],
            color=train_color,
            linewidth=1.0,
            linestyle="-",
        )
        ax.plot(
            r["val_steps"],
            r["val_smoothed"],
            color=val_color,
            linewidth=1.0,
            linestyle="-",
        )

        # Latest Test point
        latest_test_step = r["test_steps"][-1]
        latest_test_value = r["test_values"][-1]
        latest_test_smoothed = r["test_smoothed"][-1]
        latest_test_relative = r["test_relative"][-1]

        ax.plot(
            latest_test_step,
            latest_test_value,
            marker="o",
            markersize=3.5,
            color=test_color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=5,
        )

        # Irreducible Error / Ceiling
        if bound_val is not None:
            ax.axhline(
                y=bound_val,
                color=bound_color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                zorder=2,
            )
            if bound_moe is not None:
                ax.axhspan(
                    bound_val - bound_moe,
                    bound_val + bound_moe,
                    color=bound_color,
                    alpha=0.15,
                    zorder=1,
                )

        # Random Baseline
        if random_metric is not None:
            ax.axhline(
                y=random_metric,
                color=random_color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.85,
                zorder=2,
            )

        # Title is the item's title
        ax.set_title(
            r["title"], loc="left", fontsize=6.5, fontweight="bold", pad=3
        )
        ax.set_xlim(shared_xlim)
        ax.set_ylim(shared_ylim)
        ax.tick_params(axis="x", labelsize=4.8, pad=1)
        ax.tick_params(axis="y", labelsize=4.8, pad=1)

        ax.grid(
            True,
            which="major",
            axis="both",
            linestyle="-",
            linewidth=0.8,
            alpha=0.75,
            color="#cbd5e1",
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Summary Table for this item
        latest_train_step = r["train_steps"][-1]
        latest_train_value = r["train_values"][-1]
        latest_train_smoothed = r["train_smoothed"][-1]
        latest_train_relative = r["train_relative"][-1]

        latest_val_step = r["val_steps"][-1]
        latest_val_value = r["val_values"][-1]
        latest_val_smoothed = r["val_smoothed"][-1]
        latest_val_relative = r["val_relative"][-1]

        if bound_val is not None:
            bound_val_str = f"{bound_val:.4f}"
            if bound_moe is not None:
                bound_val_str += f" ±{bound_moe:.4f}"
        else:
            bound_val_str = "-"

        cell_text = [
            [
                f"{latest_train_smoothed:.4f}",
                f"{latest_train_value:.4f}",
                f"{latest_train_step}",
                f"{latest_train_relative:.2f}m",
            ],
            [
                f"{latest_val_smoothed:.4f}",
                f"{latest_val_value:.4f}",
                f"{latest_val_step}",
                f"{latest_val_relative:.2f}m",
            ],
            [
                f"{latest_test_smoothed:.4f}",
                f"{latest_test_value:.4f}",
                f"{latest_test_step}",
                f"{latest_test_relative:.2f}m",
            ],
            ["-", bound_val_str, "-", "-"],
        ]
        row_labels = ["train", "validation", "test", bound_label]

        if random_metric is not None:
            cell_text.append(["-", f"{random_metric:.4f}", "-", "-"])
            row_labels.append("random")

        col_labels = ["Smoothed", "Value", "Step", "Relative"]

        the_table = ax_table.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
            bbox=[0.20, 0.0, 0.80, 1.0],
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(4.2)

        color_map = {
            "train": train_color,
            "validation": val_color,
            "test": test_color,
            bound_label: bound_color,
            "random": random_color,
        }

        for i, row_label in enumerate(row_labels):
            cell = the_table[i + 1, -1]
            cell.set_text_props(ha="left", weight="medium")
            color = color_map.get(row_label, "#000000")
            cell.get_text().set_text(f"● {row_label}")
            cell.get_text().set_color(color)

        for (i, j), cell in the_table.get_celld().items():
            cell.set_edgecolor("#f3f4f6")
            if i == 0:
                cell.set_text_props(weight="bold", color="#4b5563")
                cell.set_facecolor("#f9fafb")
            else:
                cell.set_facecolor("#ffffff")

        run_out = {
            "latest_train_step": latest_train_step,
            "latest_train_value": latest_train_value,
            "latest_train_smoothed": latest_train_smoothed,
            "latest_val_step": latest_val_step,
            "latest_val_value": latest_val_value,
            "latest_val_smoothed": latest_val_smoothed,
            "latest_test_step": latest_test_step,
            "latest_test_value": latest_test_value,
            "latest_test_smoothed": latest_test_smoothed,
            "random_metric": random_metric,
        }
        if is_loss:
            run_out["irreducible_error"] = bound_val
        else:
            run_out["ceiling"] = bound_val
        out_summary_dict["runs"][r["title"]] = run_out

    # 2. Plot Overplotted Panel on the same page with identical xlim and ylim
    over_idx = len(valid_runs)
    over_row = over_idx // ncols
    over_col = over_idx % ncols
    inner_gs_over = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer_gs[over_row, over_col],
        height_ratios=[2.0, 1.2],
        hspace=0.20,
    )
    ax_over = fig.add_subplot(inner_gs_over[0])
    ax_legend = fig.add_subplot(inner_gs_over[1])
    ax_legend.axis("off")

    cmap = plt.get_cmap("tab10")
    legend_handles = []
    legend_labels = []

    for idx, r in enumerate(valid_runs):
        run_color = cmap(idx % 10)
        # Solid for validation, dashed for train
        (line_val,) = ax_over.plot(
            r["val_steps"],
            r["val_smoothed"],
            color=run_color,
            linewidth=1.1,
            linestyle="-",
            label=f"{r['title']} (val)",
        )
        (line_train,) = ax_over.plot(
            r["train_steps"],
            r["train_smoothed"],
            color=run_color,
            linewidth=0.9,
            linestyle="--",
            alpha=0.85,
            label=f"{r['title']} (train)",
        )
        legend_handles.extend([line_val, line_train])
        legend_labels.extend([f"{r['title']} (val —)", f"{r['title']} (train --)"])

    # Include reference lines (ceiling/irreducible error and random) if present
    first_irred = valid_runs[0].get("irred_err_dict")
    first_random = valid_runs[0].get("random_metric")
    if first_irred is not None:
        b_val = (
            first_irred.get("irred_error")
            if is_loss
            else first_irred.get("ceiling")
        )
        if b_val is not None:
            ax_over.axhline(
                y=b_val,
                color=bound_color,
                linestyle="--",
                linewidth=0.9,
                alpha=0.6,
                zorder=1,
            )
    if first_random is not None:
        ax_over.axhline(
            y=first_random,
            color=random_color,
            linestyle=":",
            linewidth=0.9,
            alpha=0.7,
            zorder=1,
        )

    ax_over.set_title(
        "Overplotted (Solid=Val, Dashed=Train)",
        loc="left",
        fontsize=6.5,
        fontweight="bold",
        pad=3,
    )
    ax_over.set_xlim(shared_xlim)
    ax_over.set_ylim(shared_ylim)
    ax_over.tick_params(axis="x", labelsize=4.8, pad=1)
    ax_over.tick_params(axis="y", labelsize=4.8, pad=1)

    ax_over.grid(
        True,
        which="major",
        axis="both",
        linestyle="-",
        linewidth=0.8,
        alpha=0.75,
        color="#cbd5e1",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax_legend.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="center",
        fontsize=4.3,
        frameon=True,
        facecolor="#f9fafb",
        edgecolor="#e5e7eb",
        ncol=1,
    )

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    output_json_path = output_path.replace(".png", ".json")
    with open(output_json_path, "w") as f:
        json.dump(out_summary_dict, f, indent=4)


if __name__ == "__main__":
    unittest.main()
