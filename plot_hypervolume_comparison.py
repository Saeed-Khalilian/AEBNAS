import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def get_surrogate_name_from_col(col_name):
    """Convert column name to a pretty surrogate model name for the legend."""
    name_lower = col_name.lower()
    if 'mlp' in name_lower:
        return 'MLP Surrogate'
    elif 'gin' in name_lower or 'gnn' in name_lower:
        if 'deep' in name_lower:
            return 'GNN (Deep) Surrogate'
        elif 'shallow' in name_lower:
            return 'GNN (Shallow) Surrogate'
        return 'GNN Surrogate'
    elif 'transformer' in name_lower or 'trans' in name_lower:
        return 'Transformer Surrogate'
    else:
        # Fallback to cleaning up the column name directly
        return col_name.replace('_hv', '').replace('_', ' ').title()

def plot_hv_comparison(csv_path, output_path=None):
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Identify iteration column and hv columns
    if 'iteration' not in df.columns:
        print("Error: CSV must contain an 'iteration' column.")
        return
        
    hv_cols = [col for col in df.columns if col != 'iteration' and col.endswith('_hv')]
    
    if not hv_cols:
        print("Error: No columns ending with '_hv' found in the CSV.")
        return

    if output_path is None:
        output_dir = os.path.dirname(csv_path) or '.'
        output_path = os.path.join(output_dir, 'hypervolume_comparison.png')

    # Apply the styling inspired by visualizations.py
    plt.style.use('seaborn-v0_8-muted')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    color_palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    
    for i, col in enumerate(hv_cols):
        label = get_surrogate_name_from_col(col)
        color = color_palette[i % len(color_palette)]
        
        # Drop NaN values for this specific run so lines connect properly if iterations are missing
        valid_df = df.dropna(subset=[col])
        
        ax.plot(
            valid_df['iteration'], 
            valid_df[col], 
            marker='o', 
            label=label, 
            color=color, 
            linewidth=2
        )

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Hypervolume (HV)')
    ax.set_title('Optimization Progress (Hypervolume Comparison)')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Hypervolume comparison plot saved to: {output_path}")
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot hypervolume comparison from a combined CSV file.')
    parser.add_argument('--csv', type=str, required=True, help='Path to the combined hypervolume CSV file')
    parser.add_argument('--output', type=str, default=None, help='Path to save the output plot (optional)')
    
    args = parser.parse_args()
    plot_hv_comparison(args.csv, args.output)
