import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def get_surrogate_name(csv_filename):
    """Convert CSV filename to a pretty surrogate model name.
    
    Examples:
    - mlp.csv -> MLP Surrogate
    - gin.csv -> GNN Surrogate
    - gnn.csv -> GNN Surrogate
    - transformer.csv -> Transformer Surrogate
    """
    filename_lower = os.path.basename(csv_filename).lower()
    
    if 'mlp' in filename_lower:
        return 'MLP Surrogate'
    elif 'gin' in filename_lower or 'gnn' in filename_lower:
        return 'GNN Surrogate'
    elif 'transformer' in filename_lower:
        return 'Transformer Surrogate'
    else:
        return os.path.basename(csv_filename)

def plot_metric(ax, df, x_col, mean_col, std_col, label, color):
    x = df[x_col]
    mean = df[mean_col]
    std = df[std_col]
    
    ax.plot(x, mean, label=label, color=color, linewidth=2)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

def main():
    parser = argparse.ArgumentParser(description="Visualize surrogate model evaluation metrics.")
    parser.add_argument("csv_files", nargs="+", help="Path to one or multiple .csv files.")
    parser.add_argument("--x-axis", type=str, default="iteration", choices=["iteration", "n_train", "iter"], help="Column to use for x-axis.")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save the plots.")
    args = parser.parse_args()

    # Generate distinct colors for the plots
    colors = plt.cm.tab10(np.linspace(0, 1, len(args.csv_files)))
    if len(args.csv_files) == 1:
        colors = ['#1f77b4'] # Default matplotlib blue

    # Plot 1: RMSE
    fig_rmse, (ax1_rmse, ax2_rmse) = plt.subplots(1, 2, figsize=(12, 5))
    fig_rmse.suptitle('RMSE of Surrogates on Test Set', fontsize=14)

    # Plot 2: Accuracy Correlation
    fig_acc_corr, (ax1_acc_corr, ax2_acc_corr) = plt.subplots(1, 2, figsize=(12, 5))
    fig_acc_corr.suptitle('Accuracy Correlation of Surrogates on Test Set', fontsize=14)

    # Plot 3: Complexity Correlation
    fig_compl_corr, (ax1_compl_corr, ax2_compl_corr) = plt.subplots(1, 2, figsize=(12, 5))
    fig_compl_corr.suptitle('Complexity Correlation of Surrogates on Test Set', fontsize=14)

    for i, csv_file in enumerate(args.csv_files):
        print(f"Processing {csv_file}...")
        df = pd.read_csv(csv_file)
        # The first 100 architectures: iteration 0. Then the next 8 (so 100-107) map to iteration 1. THE NEXT 8  (108-115) iteration 2, and so forth
        df['iteration'] = np.where(df['n_train'] <= 100, 0, (df['n_train'] - 101) // 8 + 1)
        label = get_surrogate_name(csv_file)
        color = colors[i]

        # Plot 1: RMSE
        # subplot 1: Accuracy RMSE
        plot_metric(ax1_rmse, df, args.x_axis, 'acc_rmse_mean', 'acc_rmse_std', label, color)
        ax1_rmse.set_title('Accuracy RMSE')
        ax1_rmse.set_xlabel(args.x_axis)
        ax1_rmse.set_ylabel('RMSE')

        # subplot 2: Complexity RMSE
        plot_metric(ax2_rmse, df, args.x_axis, 'compl_rmse_mean', 'compl_rmse_std', label, color)
        ax2_rmse.set_title('Complexity RMSE')
        ax2_rmse.set_xlabel(args.x_axis)
        ax2_rmse.set_ylabel('RMSE')

        # Plot 2: Accuracy Correlation
        # subplot 1: Accuracy Rho
        plot_metric(ax1_acc_corr, df, args.x_axis, 'acc_rho_mean', 'acc_rho_std', label, color)
        ax1_acc_corr.set_title('Spearman Rho (ρ)')
        ax1_acc_corr.set_xlabel(args.x_axis)
        ax1_acc_corr.set_ylabel('Correlation')

        # subplot 2: Accuracy Tau
        plot_metric(ax2_acc_corr, df, args.x_axis, 'acc_tau_mean', 'acc_tau_std', label, color)
        ax2_acc_corr.set_title('Kendall Tau (τ)')
        ax2_acc_corr.set_xlabel(args.x_axis)
        ax2_acc_corr.set_ylabel('Correlation')

        # Plot 3: Complexity Correlation
        # subplot 1: Complexity Rho
        plot_metric(ax1_compl_corr, df, args.x_axis, 'compl_rho_mean', 'compl_rho_std', label, color)
        ax1_compl_corr.set_title('Spearman Rho (ρ)')
        ax1_compl_corr.set_xlabel(args.x_axis)
        ax1_compl_corr.set_ylabel('Correlation')

        # subplot 2: Complexity Tau
        plot_metric(ax2_compl_corr, df, args.x_axis, 'compl_tau_mean', 'compl_tau_std', label, color)
        ax2_compl_corr.set_title('Kendall Tau (τ)')
        ax2_compl_corr.set_xlabel(args.x_axis)
        ax2_compl_corr.set_ylabel('Correlation')

    # Add legends and grids
    for ax in [ax1_rmse, ax2_rmse, ax1_acc_corr, ax2_acc_corr, ax1_compl_corr, ax2_compl_corr]:
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    fig_rmse.tight_layout()
    fig_acc_corr.tight_layout()
    fig_compl_corr.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    
    out_rmse = os.path.join(args.out_dir, 'plot_rmse.png')
    out_acc_corr = os.path.join(args.out_dir, 'plot_acc_corr.png')
    out_compl_corr = os.path.join(args.out_dir, 'plot_compl_corr.png')
    
    fig_rmse.savefig(out_rmse, dpi=300)
    fig_acc_corr.savefig(out_acc_corr, dpi=300)
    fig_compl_corr.savefig(out_compl_corr, dpi=300)
    
    print(f"Plots successfully saved to {os.path.abspath(args.out_dir)}")

if __name__ == '__main__':
    main()
