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
    ax1.set_xlabel(x_label)
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
    ax2.set_xlabel(x_label)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title('Surrogate Prediction Error (RMSE)')

    # 3. Accuracy Rank Correlation (Rho & Tau)
    ax3 = axes_col[2]
    ax3.plot(valid_acc[x_col], valid_acc['acc_rho'], marker='o', color='tab:blue', label='Spearman (Rho)')
    ax3.plot(valid_acc[x_col], valid_acc['acc_tau'], marker='s', color='tab:orange', label='Kendall (Tau)')
    ax3.set_ylabel('Correlation')
    ax3.set_xlabel(x_label)
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

def get_surrogate_name(csv_filename):
    """Convert CSV filename to a pretty surrogate model name.
    
    Examples:
    - mlp.csv -> MLP Surrogate
    - gin.csv -> GNN Surrogate
    - gnn.csv -> GNN Surrogate
    - transformer.csv -> Transformer Surrogate
    """
    filename_lower = csv_filename.lower()
    
    if 'mlp' in filename_lower:
        return 'MLP Surrogate'
    elif 'gin' in filename_lower or 'gnn' in filename_lower:
        return 'GNN Surrogate'
    elif 'transformer' in filename_lower:
        return 'Transformer Surrogate'
    else:
        return csv_filename

def validate_same_dataset(csv_paths):
    """Ensure all files are from the same dataset."""
    datasets = [extract_dataset_name(path) for path in csv_paths]
    if len(set(datasets)) > 1:
        raise ValueError(
            f"All input files must be from the same dataset. "
            f"Found datasets: {set(datasets)}"
        )
    return datasets[0]

def plot_history(csv_paths, use_iterations=False):
    # Accept either a single path string or a list of paths
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    csv_paths = list(csv_paths)
    n = len(csv_paths)

    # Single file: create only samples by default, or iterations if flag is set
    if n == 1:
        df = pd.read_csv(csv_paths[0])
        output_dir = os.path.dirname(csv_paths[0]) or '.'
        plt.style.use('seaborn-v0_8-muted')
        
        # Determine which mode to use
        if use_iterations:
            x_col, x_label = 'iteration', 'Iteration'
        else:
            x_col, x_label = 'n_samples', 'Number of Samples'
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 18), sharex=True)
        _plot_column_into_axes(df, axes, x_col, x_label)
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'nas_progress.png')
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
    
    # Create subplots subfolder for individual row figures
    subplots_dir = os.path.join(output_dir, 'subplots')
    os.makedirs(subplots_dir, exist_ok=True)
    print(f"Subplots folder created/verified at: {subplots_dir}")
    
    plt.style.use('seaborn-v0_8-muted')

    # Determine which mode to use
    if use_iterations:
        x_col, x_label, out_name = 'iteration', 'Iteration', 'nas_progress_comparison_iteration.png'
    else:
        x_col, x_label, out_name = 'n_samples', 'Number of Samples', 'nas_progress_comparison_samples.png'
    
    # Create main comparison figure
    fig, axes = plt.subplots(4, n, figsize=(6 * n, 18), sharex='col', sharey='row')
    # Ensure axes is a 2D array with shape (4, n)
    axes = np.atleast_2d(axes)
    if axes.shape[0] != 4:
        axes = axes.reshape(4, -1)

    for col_idx, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        col_title = get_surrogate_name(basename(csv_path))
        column_axes = axes[:, col_idx]
        _plot_column_into_axes(df, column_axes, x_col, x_label, column_title=col_title)

    plt.tight_layout()
    save_path = os.path.join(output_dir, out_name)
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to: {save_path}")
    plt.close(fig)
    
    # Create individual subplot figures for each row
    row_titles = [
        'Optimization Progress',
        'Surrogate Prediction Error',
        'Accuracy: Rank Correlation',
        'Complexity: Rank Correlation'
    ]
    
    for row_idx in range(4):
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharex='col', sharey='row')
        # Ensure axes is 1D for consistency
        if n == 1:
            axes = np.array([axes])
        else:
            axes = np.atleast_1d(axes)
        
        for col_idx, csv_path in enumerate(csv_paths):
            df = pd.read_csv(csv_path)
            col_title = get_surrogate_name(basename(csv_path))
            
            # Extract the specific subplot for this row
            ax = axes[col_idx]
            
            if row_idx == 0:  # Optimization Progress (HV)
                color_hv = 'tab:blue'
                ax.set_ylabel('Hypervolume (HV)', color=color_hv)
                ax.plot(df[x_col], df['hv'], marker='o', color=color_hv, label='HV')
                ax.tick_params(axis='y', labelcolor=color_hv)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_xlabel(x_label)
                if x_col == 'iteration':
                    ax_twin = ax.twinx()
                    color_samples = 'tab:gray'
                    ax_twin.set_ylabel('Total Samples', color=color_samples)
                    ax_twin.plot(df[x_col], df['n_samples'], marker='s', linestyle='--', color=color_samples, label='Samples')
                    ax_twin.tick_params(axis='y', labelcolor=color_samples)
                ax.set_title(f'{col_title}\nOptimization Progress (vs {x_label})')
            
            elif row_idx == 1:  # Surrogate Model Accuracy (RMSE)
                valid_acc = df.dropna(subset=['acc_rmse'])
                valid_compl = df.dropna(subset=['compl_rmse'])
                ax.plot(valid_acc[x_col], valid_acc['acc_rmse'], marker='o', label='Acc RMSE')
                ax.plot(valid_compl[x_col], valid_compl['compl_rmse'], marker='x', label='Compl RMSE')
                ax.set_ylabel('RMSE')
                ax.set_xlabel(x_label)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_title(f'{col_title}\nSurrogate Prediction Error (RMSE)')
            
            elif row_idx == 2:  # Accuracy Rank Correlation
                valid_acc = df.dropna(subset=['acc_rmse'])
                ax.plot(valid_acc[x_col], valid_acc['acc_rho'], marker='o', color='tab:blue', label='Spearman (Rho)')
                ax.plot(valid_acc[x_col], valid_acc['acc_tau'], marker='s', color='tab:orange', label='Kendall (Tau)')
                ax.set_ylabel('Correlation')
                ax.set_xlabel(x_label)
                ax.legend(loc='lower right')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_title(f'{col_title}\nAccuracy: Rank Correlation')
            
            elif row_idx == 3:  # Complexity Rank Correlation
                valid_compl = df.dropna(subset=['compl_rmse'])
                ax.plot(valid_compl[x_col], valid_compl['compl_rho'], marker='x', color='tab:green', label='Spearman (Rho)')
                ax.plot(valid_compl[x_col], valid_compl['compl_tau'], marker='d', color='tab:red', label='Kendall (Tau)')
                ax.set_ylabel('Correlation')
                ax.set_xlabel(x_label)
                ax.legend(loc='lower right')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_title(f'{col_title}\nComplexity: Rank Correlation')
        
        plt.tight_layout()
        subplot_name = f'row_{row_idx}_{row_titles[row_idx].lower().replace(": ", "_").replace(" ", "_")}.png'
        save_path = os.path.join(subplots_dir, subplot_name)
        plt.savefig(save_path, dpi=300)
        print(f"Subplot saved to: {save_path}")
        plt.close(fig)
    
    # Create combined plots subfolder
    combined_dir = os.path.join(output_dir, 'combined_plots')
    os.makedirs(combined_dir, exist_ok=True)
    print(f"Combined plots folder created/verified at: {combined_dir}")
    
    # Define color palette for surrogates (blue, orange, green)
    #color_palette = ['blue', 'orange', 'green']
    color_palette = ['tab:blue', 'tab:orange', 'tab:green']
    colors = [color_palette[i % len(color_palette)] for i in range(n)]
    
    # Create combined plots for each row
    # Row 0: Hypervolume (1 subplot with all surrogates)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for col_idx, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        col_title = get_surrogate_name(basename(csv_path))
        ax.plot(df[x_col], df['hv'], marker='o', label=col_title, color=colors[col_idx], linewidth=2)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('Hypervolume (HV)')
    ax.set_title('Optimization Progress (Hypervolume)')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = os.path.join(combined_dir, 'row_0_optimization_progress.png')
    plt.savefig(save_path, dpi=300)
    print(f"Combined plot saved to: {save_path}")
    plt.close(fig)
    
    # Row 1: Surrogate Prediction Error (2 subplots: Acc RMSE and Compl RMSE)
    fig, (ax_acc, ax_compl) = plt.subplots(1, 2, figsize=(14, 5))
    
    for col_idx, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        valid_acc = df.dropna(subset=['acc_rmse'])
        valid_compl = df.dropna(subset=['compl_rmse'])
        col_title = get_surrogate_name(basename(csv_path))
        
        ax_acc.plot(valid_acc[x_col], valid_acc['acc_rmse'], marker='o', label=col_title, color=colors[col_idx], linewidth=2)
        ax_compl.plot(valid_compl[x_col], valid_compl['compl_rmse'], marker='o', label=col_title, color=colors[col_idx], linewidth=2)
    
    ax_acc.set_xlabel(x_label)
    ax_acc.set_ylabel('RMSE')
    ax_acc.set_title('Accuracy RMSE')
    ax_acc.legend(loc='best')
    ax_acc.grid(True, linestyle='--', alpha=0.7)
    
    ax_compl.set_xlabel(x_label)
    ax_compl.set_ylabel('RMSE')
    ax_compl.set_title('Complexity RMSE')
    ax_compl.legend(loc='best')
    ax_compl.grid(True, linestyle='--', alpha=0.7)
    
    # Sync y-axis limits
    all_rmse = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        all_rmse.extend(df.dropna(subset=['acc_rmse'])['acc_rmse'].tolist())
        all_rmse.extend(df.dropna(subset=['compl_rmse'])['compl_rmse'].tolist())
    y_min, y_max = min(all_rmse), max(all_rmse)
    margin = (y_max - y_min) * 0.1
    ax_acc.set_ylim(y_min - margin, y_max + margin)
    ax_compl.set_ylim(y_min - margin, y_max + margin)
    
    plt.tight_layout()
    save_path = os.path.join(combined_dir, 'row_1_surrogate_prediction_error.png')
    plt.savefig(save_path, dpi=300)
    print(f"Combined plot saved to: {save_path}")
    plt.close(fig)
    
    # Row 2: Accuracy Rank Correlation (2 subplots: Tau and Rho)
    fig, (ax_tau, ax_rho) = plt.subplots(1, 2, figsize=(14, 5))
    
    for col_idx, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        valid_acc = df.dropna(subset=['acc_rmse'])
        col_title = get_surrogate_name(basename(csv_path))
        
        ax_tau.plot(valid_acc[x_col], valid_acc['acc_tau'], marker='s', label=col_title, color=colors[col_idx], linewidth=2)
        ax_rho.plot(valid_acc[x_col], valid_acc['acc_rho'], marker='o', label=col_title, color=colors[col_idx], linewidth=2)
    
    ax_tau.set_xlabel(x_label)
    ax_tau.set_ylabel('Correlation')
    ax_tau.set_title('Accuracy: Kendall (Tau)')
    ax_tau.legend(loc='best')
    ax_tau.grid(True, linestyle='--', alpha=0.7)
    
    ax_rho.set_xlabel(x_label)
    ax_rho.set_ylabel('Correlation')
    ax_rho.set_title('Accuracy: Spearman (Rho)')
    ax_rho.legend(loc='best')
    ax_rho.grid(True, linestyle='--', alpha=0.7)
    
    # Sync y-axis limits
    all_corr = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        valid_acc = df.dropna(subset=['acc_rmse'])
        all_corr.extend(valid_acc['acc_tau'].tolist())
        all_corr.extend(valid_acc['acc_rho'].tolist())
    y_min, y_max = min(all_corr), max(all_corr)
    margin = (y_max - y_min) * 0.1
    ax_tau.set_ylim(y_min - margin, y_max + margin)
    ax_rho.set_ylim(y_min - margin, y_max + margin)
    
    plt.tight_layout()
    save_path = os.path.join(combined_dir, 'row_2_accuracy_rank_correlation.png')
    plt.savefig(save_path, dpi=300)
    print(f"Combined plot saved to: {save_path}")
    plt.close(fig)
    
    # Row 3: Complexity Rank Correlation (2 subplots: Tau and Rho)
    fig, (ax_tau, ax_rho) = plt.subplots(1, 2, figsize=(14, 5))
    
    for col_idx, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        valid_compl = df.dropna(subset=['compl_rmse'])
        col_title = get_surrogate_name(basename(csv_path))
        
        ax_tau.plot(valid_compl[x_col], valid_compl['compl_tau'], marker='d', label=col_title, color=colors[col_idx], linewidth=2)
        ax_rho.plot(valid_compl[x_col], valid_compl['compl_rho'], marker='x', label=col_title, color=colors[col_idx], linewidth=2)
    
    ax_tau.set_xlabel(x_label)
    ax_tau.set_ylabel('Correlation')
    ax_tau.set_title('Complexity: Kendall (Tau)')
    ax_tau.legend(loc='best')
    ax_tau.grid(True, linestyle='--', alpha=0.7)
    
    ax_rho.set_xlabel(x_label)
    ax_rho.set_ylabel('Correlation')
    ax_rho.set_title('Complexity: Spearman (Rho)')
    ax_rho.legend(loc='best')
    ax_rho.grid(True, linestyle='--', alpha=0.7)
    
    # Sync y-axis limits
    all_corr = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        valid_compl = df.dropna(subset=['compl_rmse'])
        all_corr.extend(valid_compl['compl_tau'].tolist())
        all_corr.extend(valid_compl['compl_rho'].tolist())
    y_min, y_max = min(all_corr), max(all_corr)
    margin = (y_max - y_min) * 0.1
    ax_tau.set_ylim(y_min - margin, y_max + margin)
    ax_rho.set_ylim(y_min - margin, y_max + margin)
    
    plt.tight_layout()
    save_path = os.path.join(combined_dir, 'row_3_complexity_rank_correlation.png')
    plt.savefig(save_path, dpi=300)
    print(f"Combined plot saved to: {save_path}")
    plt.close(fig)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize NAS history from history.csv')
    parser.add_argument('csv_paths', nargs='+', type=str, help='Path(s) to history.csv file(s)')
    parser.add_argument('--iterations', action='store_true', help='Use iterations on x-axis instead of samples (single file only)')
    args = parser.parse_args()

    # Validate provided paths
    missing = [p for p in args.csv_paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"Error: File not found at {p}")
        raise SystemExit(1)

    plot_history(args.csv_paths, use_iterations=args.iterations)
