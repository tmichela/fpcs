import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from time import perf_counter

from fpcs import downsample
from lttbc import downsample as lttbc_downsample


DATA_FILE = "./FaultDetectionA_TEST_SUBSET.npz"


def load_npz_file(path):
    data = np.load(path)
    if "data_x" in data:
        data_x = data["data_x"]
    elif "x" in data:
        data_x = data["x"]
    else:
        raise ValueError(f"NPZ file {path} missing 'data_x' array")

    data_y = data["data_y"] if "data_y" in data else None
    return data_x, data_y


def plot_stacked_comparison(x, y, algorithms_results, title, save_path=None):
    """
    Create a vertically stacked set of plots, one for each downsampling algorithm.
    """
    n_algos = len(algorithms_results)
    fig, axes = plt.subplots(n_algos, 1, figsize=(12, 4 * n_algos), sharex=True)

    if n_algos == 1:
        axes = [axes]

    for i, (name, (x_down, y_down)) in enumerate(algorithms_results.items()):
        ax = axes[i]
        # Original data in blue background
        ax.plot(x, y, color='tab:blue', label='Original', alpha=1.0, linewidth=0.8)
        # Downsampled data in orange
        ax.plot(x_down, y_down, color='tab:orange', marker='o', markersize=2, linestyle='-', 
                linewidth=1, alpha=0.7, label=name)

        ax.set_title(f"Algorithm: {name}", fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_ylabel('Value')

    plt.xlabel('Time')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downsampling comparison.')
    parser.add_argument('--data', type=str, default=DATA_FILE, help='Path to data file')
    parser.add_argument('--n_out', type=int, default=5000, help='Target number of output points (default: 5,000)')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Data file not found at {args.data}")
        exit(1)

    print(f"Loading {args.data}...")
    data_x, data_y = load_npz_file(args.data)

    # Flatten all cases into a single continuous series
    y = np.asarray(data_x, dtype=np.float64).flatten()
    x = np.arange(len(y), dtype=np.float64)

    input_size = len(x)
    n_out = args.n_out

    results = {}
    stats = []

    print(f"-- Processing {input_size:,} points...")

    # 1. FPCS
    t0 = perf_counter()
    # Dynamic R to match target n_out roughly
    r_factor = max(1, int(input_size / (0.8 * n_out)))
    x_fpcs, y_fpcs = downsample(x, y, r_factor)
    dt_fpcs = perf_counter() - t0
    results['FPCS'] = (x_fpcs, y_fpcs)
    stats.append({
        'algo': 'FPCS',
        'in': input_size,
        'out': len(x_fpcs),
        'time_ms': dt_fpcs * 1000
    })

    # 2. LTTB (C implementation)
    t0 = perf_counter()
    x_lttb, y_lttb = lttbc_downsample(x, y, n_out)
    dt_lttb = perf_counter() - t0
    results['LTTB'] = (x_lttb, y_lttb)
    stats.append({
        'algo': 'LTTB',
        'in': input_size,
        'out': len(x_lttb),
        'time_ms': dt_lttb * 1000
    })

    # Visualize
    plot_stacked_comparison(x, y, results, 
                           title=f"Downsampling Comparison ({args.data})", 
                           save_path="downsampling_comparison.png")

    # Final Summary Table
    print(f"\n### Performance Comparison: {args.data}")
    print(f"| Algorithm | Input Points | Output Points | Time (ms) | Throughput (M pts/s) |")
    print(f"| :--- | ---: | ---: | ---: | ---: |")
    
    # Calculate relative speed comparison
    min_time = min(s['time_ms'] for s in stats)
    
    for s in stats:
        throughput = (s['in'] / s['time_ms'] / 1000)
        ratio = (s['out'] / s['in']) * 100
        
        # Format output points with ratio
        out_str = f"{s['out']:,} ({ratio:.2f}%)"
        
        # Format time with speed comparison
        speed_comp = s['time_ms'] / min_time
        time_str = f"{s['time_ms']:.3f} ({'1' if speed_comp == 1 else f'{speed_comp:.2f}x'})"
        
        print(f"| {s['algo']} | {s['in']:,} | {out_str} | {time_str} | {throughput:.2f} |")
    print()
