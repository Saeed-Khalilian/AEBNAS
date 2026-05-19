import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from os.path import basename

def _plot_column_into_axes(df, axes_col, x_col, x_label, column_title=None):
    # Plot the four subplots into an existing column of 4 axes
    # Filter valid data
    valid_acc = df.dropna(subset=['acc_rmse'])
    valid_compl = df.dropna(subset=['compl_rmse'])

    # 1. Optimization Progress (HV)
    ax1 = axes_col[0]
    color_hv = 'tab:blue'
    ax1.set_ylabel('Hypervolume (HV)', color=color_hv)
    ax1.plot(df[x_col], df['hv'], marker='o', color=color_hv, label='HV')
    ax1.tick_params(axis='y', labelcolor=color_hv)
    ax1.grid(True, linestyle='--', alpha=0.7)
    if x_col == 'iteration':
        ax1_twin = ax1.twinx()
        color_samples = 'tab:gray'
        ax1_twin.set_ylabel('Total Samples', color=color_samples)
        ax1_twin.plot(df[x_col], df['n_samples'], marker='s', linestyle='--', color=color_samples, label='Samples')
        ax1_twin.tick_params(axis='y', labelcolor=color_samples)
    if column_title:
        ax1.set_title(f'{column_title}\nOptimization Progress (vs {x_label})')
    else:
        ax1.set_title(f'Optimization Progress (vs {x_label})')

    # 2. Surrogate Model Accuracy (RMSE)
    ax2 = axes_col[1]
    ax2.plot(valid_acc[x_col], valid_acc['acc_rmse'], marker='o', label='Acc RMSE')
    ax2.plot(valid_compl[x_col], valid_compl['compl_rmse'], marker='x', label='Compl RMSE')
    ax2.set_ylabel('RMSE')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title('Surrogate Prediction Error (RMSE)')

    # 3. Accuracy Rank Correlation (Rho & Tau)
    ax3 = axes_col[2]
    ax3.plot(valid_acc[x_col], valid_acc['acc_rho'], marker='o', color='tab:blue', label='Spearman (Rho)')
    ax3.plot(valid_acc[x_col], valid_acc['acc_tau'], marker='s', color='tab:orange', label='Kendall (Tau)')
    ax3.set_ylabel('Correlation')
    ax3.legend(loc='lower right')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_title('Accuracy: Rank Correlation')

    # 4. Complexity Rank Correlation (Rho & Tau)
    ax4 = axes_col[3]
    ax4.plot(valid_compl[x_col], valid_compl['compl_rho'], marker='x', color='tab:green', label='Spearman (Rho)')
    ax4.plot(valid_compl[x_col], valid_compl['compl_tau'], marker='d', color='tab:red', label='Kendall (Tau)')
    ax4.set_ylabel('Correlation')
    ax4.set_xlabel(x_label)
    ax4.legend(loc='lower right')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.set_title('Complexity: Rank Correlation')

def extract_dataset_name(file_path):
    """Extract dataset name from file path.
    
    Expects paths like: ./plots/cifar10/gin/gin.csv
    Returns the dataset name (e.g., 'cifar10')
    """
    # Normalize path separators to forward slashes for consistency
    normalized_path = file_path.replace('\\', '/')
    path_parts = normalized_path.split('/')
    
    # Find 'plots' in the path and get the next part as dataset name
    if 'plots' in path_parts:
        plots_idx = path_parts.index('plots')
        if plots_idx + 1 < len(path_parts):
            return path_parts[plots_idx + 1]
    
    raise ValueError(f"Could not extract dataset name from path: {file_path}")

def validate_same_dataset(csv_paths):
    """Ensure all files are from the same dataset."""
    datasets = [extract_dataset_name(path) for path in csv_paths]
    if len(set(datasets)) > 1:
        raise ValueError(
            f"All input files must be from the same dataset. "
            f"Found datasets: {set(datasets)}"
        )
    return datasets[0]

def plot_history(csv_paths):
    # Accept either a single path string or a list of paths
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    csv_paths = list(csv_paths)
    n = len(csv_paths)

    # Single file: preserve original behavior and filenames
    if n == 1:
        df = pd.read_csv(csv_paths[0])
        output_dir = os.path.dirname(csv_paths[0]) or '.'
        plt.style.use('seaborn-v0_8-muted')
        fig, axes = plt.subplots(4, 1, figsize=(10, 18), sharex=True)

        _plot_column_into_axes(df, axes, 'iteration', 'Iteration')
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'nas_progress.png')
        plt.savefig(save_path, dpi=300)
        print(f"Visualization saved to: {save_path}")
        plt.close(fig)

        plt.style.use('seaborn-v0_8-muted')
        fig, axes = plt.subplots(4, 1, figsize=(10, 18), sharex=True)
        _plot_column_into_axes(df, axes, 'n_samples', 'Number of Samples')
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'nas_progress_samples.png')
        plt.savefig(save_path, dpi=300)
        print(f"Visualization saved to: {save_path}")
        plt.close(fig)
        return

    # Multiple files: validate same dataset and create comparison figures
    dataset_name = validate_same_dataset(csv_paths)
    
    # Determine output directory: dataset_folder/comparison
    # Get the plots folder path from the first file
    first_file_dir = os.path.dirname(csv_paths[0])
    plots_dir = first_file_dir
    # Navigate up to the dataset folder (plots/dataset_name)
    while plots_dir and os.path.basename(plots_dir) != dataset_name:
        plots_dir = os.path.dirname(plots_dir)
    
    if not plots_dir or os.path.basename(plots_dir) != dataset_name:
        raise ValueError(f"Could not find dataset folder for dataset: {dataset_name}")
    
    output_dir = os.path.join(plots_dir, 'comparison')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Comparison folder created/verified at: {output_dir}")
    
    plt.style.use('seaborn-v0_8-muted')

    for x_col, x_label, out_name in [
        ('iteration', 'Iteration', 'nas_progress_comparison_iteration.png'),
        ('n_samples', 'Number of Samples', 'nas_progress_comparison_samples.png')
    ]:
        fig, axes = plt.subplots(4, n, figsize=(6 * n, 18), sharex='col', sharey='row')
        # Ensure axes is a 2D array with shape (4, n)
        axes = np.atleast_2d(axes)
        if axes.shape[0] != 4:
            axes = axes.reshape(4, -1)

        for col_idx, csv_path in enumerate(csv_paths):
            df = pd.read_csv(csv_path)
            col_title = basename(csv_path)
            column_axes = axes[:, col_idx]
            _plot_column_into_axes(df, column_axes, x_col, x_label, column_title=col_title)

        plt.tight_layout()
        save_path = os.path.join(output_dir, out_name)
        plt.savefig(save_path, dpi=300)
        print(f"Visualization saved to: {save_path}")
        plt.close(fig)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize NAS history from history.csv')
    parser.add_argument('csv_paths', nargs='+', type=str, help='Path(s) to history.csv file(s)')
    args = parser.parse_args()

    # Validate provided paths
    missing = [p for p in args.csv_paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"Error: File not found at {p}")
        raise SystemExit(1)

    plot_history(args.csv_paths)
