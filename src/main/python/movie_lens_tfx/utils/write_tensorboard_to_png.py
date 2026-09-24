import json
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util
from typing import List

import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def get_latest_scalars(log_dir, scalar_name):
    """
    Extracts steps, values, and wall_times for a given scalar from TensorBoard logs.
    """
    # Find the event file
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None, None, None
    # Use the most recent event file
    event_file = max(event_files, key=os.path.getctime)
    
    # Initialize accumulator
    try:
        ea = event_accumulator.EventAccumulator(
            event_file,
            size_guidance={'scalars': 0, 'tensors': 0}  # Keep all data
        )
        ea.Reload()
    except Exception as e:
        print(f"Error loading event file {event_file}: {e}")
        return None, None, None
    
    # Check for scalars and tensors
    available_tags = ea.Tags()
    steps, values, wall_times = [], [], []
    
    if scalar_name in available_tags.get('scalars', []):
        events = ea.Scalars(scalar_name)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        wall_times = [e.wall_time for e in events]
    elif scalar_name in available_tags.get('tensors', []):
        events = ea.Tensors(scalar_name)
        steps = [e.step for e in events]
        values = [float(tensor_util.make_ndarray(e.tensor_proto)) for e in
            events]
        wall_times = [e.wall_time for e in events]
    else:
        print(f"Metric '{scalar_name}' not found in {log_dir}.")
        print(f"Available scalars: {available_tags.get('scalars', [])}")
        print(f"Available tensors: {available_tags.get('tensors', [])}")
        return None, None, None
    
    return steps, values, wall_times


def exponential_smoothing(scalars, weight=0.6):
    """
    Applies exponential smoothing to a list of scalar values.
    """
    smoothed = []
    if not scalars:
        return smoothed
    last = scalars[0]  # First value is the initial smoothed value
    smoothed.append(last)
    for current in scalars[1:]:
        # EMA = (1 - weight) * previous_EMA + weight * current_value
        # Note: TensorBoard's default weight is roughly 0.6
        smoothed_val = (1 - weight) * last + weight * current
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def generate_tensorboard_chart(train_dir, val_dir, test_dir, irred_err_dict,
        scalar_name, output_path, smoothing_weight=0.6):
    """
    Generates a single PNG chart (~3.5x4 inches) with overplotted train/val curves,
    faded raw lines, smoothed lines, and a data summary table.
    """
    train_steps, train_values, train_times = get_latest_scalars(train_dir,
        scalar_name)
    val_steps, val_values, val_times = get_latest_scalars(val_dir, scalar_name)
    test_steps, test_values, test_times = get_latest_scalars(test_dir,
        scalar_name.replace("epoch_", ""))
    
    if train_steps is None or val_steps is None or test_steps is None:
        print("Could not load data for runs. Exiting.")
        return
    
    is_loss = scalar_name.find("loss") > -1
    
    train_smoothed = exponential_smoothing(train_values, smoothing_weight)
    val_smoothed = exponential_smoothing(val_values, smoothing_weight)
    test_smoothed = exponential_smoothing(test_values, smoothing_weight)
    
    def get_relative_times(times):
        if not times: return []
        start_time = times[0]
        return [(t - start_time) / 60.0 for t in times]
    
    train_relative = get_relative_times(train_times)
    val_relative = get_relative_times(val_times)
    test_relative = get_relative_times(test_times)
    
    # --- Dynamic Bounds Logic ---
    bound_val = None
    bound_moe = None
    bound_label = 'irred error' if is_loss else 'ceiling'
    
    if irred_err_dict is not None:
        if is_loss:
            # Safely check both potential keys just in case
            bound_val = irred_err_dict.get('irreducible_error',
                irred_err_dict.get('irred_error'))
        else:
            bound_val = irred_err_dict.get('ceiling')
        
        # Support both margin-of-error naming conventions
        bound_moe = irred_err_dict.get('margin_of_error_on_irred_err',
            irred_err_dict.get('irred_error_margin_of_error'))
    
    # Extract latest test stats to plot as a point
    latest_test_step = test_steps[-1]
    latest_test_value = test_values[-1]
    latest_test_smoothed = test_smoothed[-1]
    latest_test_relative = test_relative[-1]
    
    # Adjust height_ratios slightly to give more room for the newly added table rows
    fig = plt.figure(figsize=(2.5, 2.5), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.2], hspace=0.25)
    
    # Plot Axis
    ax = fig.add_subplot(gs[0])
    
    # High-contrast color scheme
    train_color = '#1f77b4'  # Blue
    train_color_faded = '#aec7e8'  # Faded Blue
    val_color = '#ff7f0e'  # Orange
    val_color_faded = '#ffbb78'  # Faded Orange
    test_color = '#d62728'  # Red
    bound_color = '#2ca02c'  # Green
    
    # Plot raw (faded) and smoothed (solid) lines
    ax.plot(train_steps, train_values, color=train_color_faded, alpha=0.7,
        linewidth=0.6, linestyle='-')
    ax.plot(val_steps, val_values, color=val_color_faded, alpha=0.7,
        linewidth=0.6, linestyle='-')
    ax.plot(train_steps, train_smoothed, color=train_color, linewidth=1.2,
        linestyle='-')
    ax.plot(val_steps, val_smoothed, color=val_color, linewidth=1.2,
        linestyle='-')
    
    # Plot Test Point on the right hand side
    ax.plot(latest_test_step, latest_test_value, marker='o', markersize=4,
        color=test_color, markeredgecolor='white', markeredgewidth=0.5,
        zorder=5)
    
    # Plot Irreducible Error / Ceiling (Horizontal Line + Shaded Margin of Error)
    if bound_val is not None:
        ax.axhline(y=bound_val, color=bound_color, linestyle='--',
            linewidth=1.2, alpha=0.8, zorder=2)
        
        # Plot the shaded area for Margin of Error if it exists
        if bound_moe is not None:
            ax.axhspan(bound_val - bound_moe,
                bound_val + bound_moe,
                color=bound_color, alpha=0.15, zorder=1)
    
    # Typography & layout scaling
    ax.set_title(scalar_name, loc='left', fontsize=7.5, fontweight='bold',
        pad=4)
    ax.tick_params(axis='x', labelsize=5.5, pad=1)
    ax.tick_params(axis='y', labelsize=5.5, pad=1)
    
    # High-contrast grid guidelines
    ax.grid(True, which='major', axis='both', linestyle='-', linewidth=1.0,
        alpha=0.75, color='#cbd5e1')
    
    # Remove bounding box spines for clean look
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Update ylim check to account for the new plotted values and shaded region
    all_values = train_values + val_values + train_smoothed + val_smoothed + [
        latest_test_value]
    if bound_val is not None:
        all_values.append(bound_val)
        if bound_moe is not None:
            # Add bounds of the margin of error so they don't get clipped off the chart
            all_values.extend([bound_val - bound_moe, bound_val + bound_moe])
    
    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        y_padding = max((y_max - y_min) * 0.08, 0.01)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Table Axis
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    
    latest_train_step = train_steps[-1]
    latest_train_value = train_values[-1]
    latest_train_smoothed = train_smoothed[-1]
    latest_train_relative = train_relative[-1]
    
    latest_val_step = val_steps[-1]
    latest_val_value = val_values[-1]
    latest_val_smoothed = val_smoothed[-1]
    latest_val_relative = val_relative[-1]
    
    # Safely format bounds for the table
    if bound_val is not None:
        bound_val_str = f"{bound_val:.4f}"
        if bound_moe is not None:
            bound_val_str += f" ±{bound_moe:.4f}"
    else:
        bound_val_str = "-"
    
    # Table Content
    cell_text = [
        [f"{latest_train_smoothed:.4f}", f"{latest_train_value:.4f}",
            f"{latest_train_step}", f"{latest_train_relative:.2f}m"],
        [f"{latest_val_smoothed:.4f}", f"{latest_val_value:.4f}",
            f"{latest_val_step}", f"{latest_val_relative:.2f}m"],
        [f"{latest_test_smoothed:.4f}", f"{latest_test_value:.4f}",
            f"{latest_test_step}", f"{latest_test_relative:.2f}m"],
        ["-", bound_val_str, "-", "-"]
    ]
    
    # Dynamically label the last row based on metric type
    row_labels = ['train', 'validation', 'test', bound_label]
    col_labels = ['Smoothed', 'Value', 'Step', 'Relative']
    
    the_table = ax_table.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]
    )
    
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(5.0)
    
    # Custom Row Label Colors & Styling Map
    color_map = {
        'train': train_color,
        'validation': val_color,
        'test': test_color,
        bound_label: bound_color
        # Color dynamically maps to either 'irred error' or 'ceiling'
    }
    
    for i, row_label in enumerate(row_labels):
        cell = the_table[i + 1, -1]
        cell.set_text_props(ha='left', weight='medium')
        color = color_map.get(row_label, '#000000')
        cell.get_text().set_text(f'● {row_label}')
        cell.get_text().set_color(color)
    
    # Style Table Header and Cells
    for (i, j), cell in the_table.get_celld().items():
        cell.set_edgecolor('#f3f4f6')
        if i == 0:
            cell.set_text_props(weight='bold', color='#4b5563')
            cell.set_facecolor('#f9fafb')
        else:
            cell.set_facecolor('#ffffff')
    
    # Save output chart
    os.makedirs(os.path.dirname(output_path),
        exist_ok=True) if os.path.dirname(output_path) else None
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Update dictionary structure
    out_dict = {
        "latest_train_step": latest_train_step,
        "latest_train_value": latest_train_value,
        "latest_train_smoothed": latest_train_smoothed,
        "latest_val_step": latest_val_step,
        "latest_val_value": latest_val_value,
        "latest_val_smoothed": latest_val_smoothed,
        "latest_test_step": latest_test_step,
        "latest_test_value": latest_test_value,
        "latest_test_smoothed": latest_test_smoothed,
    }
    
    # Dynamically inject the proper key for JSON
    if is_loss:
        out_dict["irreducible_error"] = bound_val
    else:
        out_dict["ceiling"] = bound_val
    
    output_file_path = output_path.replace(".png", ".json")
    with open(output_file_path, "w") as f:
        json.dump(out_dict, f, indent=4)
    #print(f"Successfully saved chart with high-contrast grids to {output_path}")

def export_scalars_to_png(log_dir, output_png_path, scalar_name='loss'):
    # OPTIMIZATION: Prevent memory bloat and truncation
    # '0' means "keep all" for scalars/tensors (default truncates at 10k steps)
    # '1' is the minimum allowed, preventing massive RAM usage for heavy assets
    size_guidance = {
        'scalars': 0,
        'tensors': 0,
        'images': 1,
        'audio': 1,
        'histograms': 1,
    }
    
    # Initialize accumulator and load data
    ea = event_accumulator.EventAccumulator(log_dir,
        size_guidance=size_guidance)
    ea.Reload()
    
    steps, values = [], []
    available_tags = ea.Tags()
    scalar_keys = available_tags.get('scalars', [])
    tensor_keys = available_tags.get('tensors', [])
    
    # Extract data (Accounts for differences between TF1 and TF2 logging)
    if scalar_name in scalar_keys:
        events = ea.Scalars(scalar_name)
        steps = [e.step for e in events]
        values = [e.value for e in events]
    elif scalar_name in tensor_keys:
        # TensorFlow 2.x often logs scalars as rank-0 tensors
        events = ea.Tensors(scalar_name)
        steps = [e.step for e in events]
        values = [float(tensor_util.make_ndarray(e.tensor_proto)) for e in
            events]
    else:
        print(f"Metric '{scalar_name}' not found.")
        print(f"Available scalars: {scalar_keys}")
        print(f"Available tensors: {tensor_keys}")
        return
        
        # Plot using standard matplotlib
    plt.figure(figsize=(2.5, 2.5))
    plt.plot(steps, values, label=scalar_name, color='#ff7043', linewidth=1.5)
    plt.xlabel('Steps')
    plt.ylabel(scalar_name.capitalize())
    plt.title(f'{scalar_name.capitalize()} over Time')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Save as PNG
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved chart to {output_png_path}")
    
    # Plot using standard matplotlib
    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, label=scalar_name, color='#ff7043', linewidth=1.5)
    plt.xlabel('Steps')
    plt.ylabel(scalar_name.capitalize())
    plt.title(f'{scalar_name.capitalize()} over Time')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Save as PNG
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def list_tfevents_metrics(log_dir) -> List[str]:
    # Load the event accumulator with zero-guidance to ensure all tags are read
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            'scalars': 0,
            'tensors': 0,
            'images': 1,
            'audio': 1,
            'histograms': 1,
        }
    )
    ea.Reload()
    
    # ea.Tags() returns a dictionary of all data categories found
    tags = ea.Tags()
    
    print(f"Inspecting log directory: {log_dir}\n")
    
    outlist = []
    
    # 1. Standard Scalar Metrics (TF1 style / basic logging)
    scalars = tags.get('scalars', [])
    print(f"📊 Scalars ({len(scalars)} found):")
    for tag in scalars:
        print(f"    - {tag}")
        outlist.append(tag)
    
    # 2Tensor Metrics (TF2.x style scalar logs often live here)
    tensors = tags.get('tensors', [])
    print(f"\n📦 Tensors ({len(tensors)} found):")
    for tag in tensors:
        print(f"    - {tag}")
        outlist.append(tag)
    
    #  Other asset types (Optional check)
    for category in ['images', 'histograms', 'audio']:
        items = tags.get(category, [])
        if items:
            print(f"\n🖼️ {category.capitalize()} ({len(items)} found):")
            for tag in items:
                print(f"    - {tag}")


    return outlist