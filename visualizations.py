import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def create_plot(df, output_dir, x_col, filename, x_label):
    # Set style
    plt.style.use('seaborn-v0_8-muted')
    fig, axes = plt.subplots(4, 1, figsize=(10, 18), sharex=True)
    
    # Filter valid data
    valid_acc = df.dropna(subset=['acc_rmse'])
    valid_compl = df.dropna(subset=['compl_rmse'])

    # 1. Optimization Progress (HV)
    ax1 = axes[0]
    color_hv = 'tab:blue'
    ax1.set_ylabel('Hypervolume (HV)', color=color_hv)
    ax1.plot(df[x_col], df['hv'], marker='o', color=color_hv, label='HV')
    ax1.tick_params(axis='y', labelcolor=color_hv)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Only add sample count twin axis if iteration is the main X axis
    if x_col == 'iteration':
        ax1_twin = ax1.twinx()
        color_samples = 'tab:gray'
        ax1_twin.set_ylabel('Total Samples', color=color_samples)
        ax1_twin.plot(df[x_col], df['n_samples'], marker='s', linestyle='--', color=color_samples, label='Samples')
        ax1_twin.tick_params(axis='y', labelcolor=color_samples)
    
    ax1.set_title(f'Optimization Progress (vs {x_label})')
    
    # 2. Surrogate Model Accuracy (RMSE)
    ax2 = axes[1]
    ax2.plot(valid_acc[x_col], valid_acc['acc_rmse'], marker='o', label='Acc RMSE')
    ax2.plot(valid_compl[x_col], valid_compl['compl_rmse'], marker='x', label='Compl RMSE')
    ax2.set_ylabel('RMSE')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title('Surrogate Prediction Error (RMSE)')
    
    # 3. Accuracy Rank Correlation (Rho & Tau)
    ax3 = axes[2]
    ax3.plot(valid_acc[x_col], valid_acc['acc_rho'], marker='o', color='tab:blue', label='Spearman (Rho)')
    ax3.plot(valid_acc[x_col], valid_acc['acc_tau'], marker='s', color='tab:orange', label='Kendall (Tau)')
    ax3.set_ylabel('Correlation')
    ax3.legend(loc='lower right')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_title('Accuracy: Rank Correlation')
    
    # 4. Complexity Rank Correlation (Rho & Tau)
    ax4 = axes[3]
    ax4.plot(valid_compl[x_col], valid_compl['compl_rho'], marker='x', color='tab:green', label='Spearman (Rho)')
    ax4.plot(valid_compl[x_col], valid_compl['compl_tau'], marker='d', color='tab:red', label='Kendall (Tau)')
    ax4.set_ylabel('Correlation')
    ax4.set_xlabel(x_label)
    ax4.legend(loc='lower right')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.set_title('Complexity: Rank Correlation')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to: {save_path}")
    plt.close(fig)

def plot_history(csv_path):
    # Load data
    df = pd.read_csv(csv_path)
    output_dir = os.path.dirname(csv_path)
    
    # Generate both plots
    create_plot(df, output_dir, 'iteration', 'nas_progress.png', 'Iteration')
    create_plot(df, output_dir, 'n_samples', 'nas_progress_samples.png', 'Number of Samples')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize NAS history from history.csv')
    parser.add_argument('csv_path', type=str, help='Path to the history.csv file')
    args = parser.parse_args()
    
    if os.path.exists(args.csv_path):
        plot_history(args.csv_path)
    else:
        print(f"Error: File not found at {args.csv_path}")
