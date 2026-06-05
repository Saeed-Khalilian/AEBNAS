import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def get_surrogate_name(path):
    """Convert path to a pretty surrogate model name based on keywords."""
    path_lower = path.lower()
    if 'mlp' in path_lower:
        return 'MLP Surrogate'
    elif 'gin' in path_lower or 'gnn' in path_lower:
        return 'GNN Surrogate'
    elif 'transformer' in path_lower:
        return 'Transformer Surrogate'
    return None

def get_non_dominated(costs):
    """
    Find the pareto-efficient points
    :param costs: list of [cost1, cost2]
    :return: list of booleans
    """
    is_efficient = [True] * len(costs)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            for j, other in enumerate(costs):
                if is_efficient[j] and i != j:
                    if c[0] <= other[0] and c[1] <= other[1] and (c[0] < other[0] or c[1] < other[1]):
                        is_efficient[j] = False
    return is_efficient

def plot_models(csv_paths, target_macs=None):
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
        
    n = len(csv_paths)
    fig, axes = plt.subplots(1, n, figsize=(6 * n + (2 if n == 1 else 0), 5), sharex=True, sharey=True, squeeze=False)
    axes = axes[0] # it's a 2D array of shape (1, n) because of squeeze=False
    
    plt.style.use('seaborn-v0_8-muted')
    
    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        
        # Find pareto front: Minimize top1_error and minimize MAEP to target_macs
        target_mac = target_macs[i] if target_macs and i < len(target_macs) else None
        if target_mac:
            df['maep'] = np.abs((df['avg_macs'] - target_mac) / target_mac) 
            adjusted_error = df['top1_error'] + 0.2 * df['maep']
            costs = np.column_stack((adjusted_error, df['maep'])).tolist()
        else:
            costs = df[['top1_error', 'avg_macs']].values.tolist()
            
        is_pareto = get_non_dominated(costs)
        df['is_pareto'] = is_pareto
        
        ax = axes[i]
        
        # All models scatter
        # Use viridis colormap to go from purple (0) to yellow (30)
        # The first 100 architectures: iteration 0. Then the next 8 map to iteration 1.
        indices = np.arange(len(df))
        iterations = np.where(indices < 100, 0, 1 + (indices - 100) // 8)

        x_col = 'maep' if target_mac else 'avg_macs'
        
        scatter = ax.scatter(df[x_col], df['top1_acc'], 
                             c=iterations, cmap='viridis', 
                             alpha=0.7, label='Evaluated Models', s=60, zorder=2)
        
        # Pareto front scatter
        pareto_df = df[df['is_pareto']]
        ax.scatter(pareto_df[x_col], pareto_df['top1_acc'], 
                   c='red', marker='*', s=250, edgecolors='black', 
                   label='Pareto Front', zorder=5)
        
        # Set titles and labels
        surrogate_name = get_surrogate_name(csv_path)
        
        if surrogate_name:
            title_str = f"Evaluated Architectures - {surrogate_name}"
        else:
            title_str = "Evaluated Architectures"
            
        ax.set_title(title_str)
            
        x_label = "MAEP Error to Target MACs (%)" if target_mac else "Average Number of MACs (M)"
        ax.set_xlabel(x_label)
        if i == 0:
            ax.set_ylabel("Top-1 Accuracy (%)")
        ax.tick_params(labelleft=True)
            
        ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
        
        # Plot target MACs if provided
        if target_mac:
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Target MACs', zorder=3)
            
        ax.legend(loc='lower right')
        
        # Add colorbar for each subplot to indicate iteration
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Iteration Progress')
        
    plt.tight_layout()
    
    # Save the figure
    if n == 1:
        out_dir = os.path.dirname(csv_paths[0]) or '.'
        out_path = os.path.join(out_dir, "models_found.png")
    else:
        out_dir = os.path.dirname(csv_paths[0]) or '.'
        out_path = os.path.join(out_dir, "comparison_models_found.png")
        
    plt.savefig(out_path, dpi=300)
    print(f"Visualization saved to: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize all evaluated architectures and pareto front')
    parser.add_argument('csv_paths', nargs='+', type=str, help='Path(s) to all_evaluated_architectures.csv file(s)')
    parser.add_argument('--target_macs', nargs='+', type=float, help='Target MACs for each CSV file')
    args = parser.parse_args()
    
    missing = [p for p in args.csv_paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"Error: File not found at {p}")
        raise SystemExit(1)
        
    if args.target_macs and len(args.target_macs) != len(args.csv_paths):
        print("Error: The number of --target_macs must match the number of CSV files.")
        raise SystemExit(1)
        
    plot_models(args.csv_paths, args.target_macs)
