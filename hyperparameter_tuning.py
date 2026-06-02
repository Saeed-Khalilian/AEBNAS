import argparse
import ast
import json
import os
import random
import sys
import numpy as np
from sklearn.model_selection import KFold

from acc_predictor.factory import get_acc_predictor
from search_space.ofa_search_space import OFASearchSpace
from utils import get_correlation

def load_stats_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        return ast.literal_eval(content)
    except Exception:
        pass
    try:
        return json.loads(content)
    except Exception as e:
        raise RuntimeError(f"Cannot parse {path}: {e}") from e

def extract_archive(data):
    if "archive" not in data:
        raise KeyError("'archive' key missing.")
    return data["archive"]

def prepare_data(archive, search_space, predictor_type):
    acc_targets = np.array([entry[1] for entry in archive])
    compl_targets = np.array([np.dot(entry[2], entry[3]) for entry in archive])
    
    if predictor_type in ["gin", "transformer"]:
        inputs = np.array([entry[0] for entry in archive])
    else:
        inputs = np.array([search_space.encode(entry[0]) for entry in archive])
        
    return inputs, acc_targets, compl_targets

def sample_hyperparams(surrogate_type):
    if surrogate_type == 'gin':
        return {
            'hidden_dim': random.choice([16, 32, 64, 128]),
            'output_dimension': random.choice([16, 32, 64, 128]),
            'dropout_rate': random.choice([0.0, 0.1, 0.2, 0.3]),
            'num_gnn_layers': random.choice([2, 3, 4]),
            'mlp_hidden_dim': random.choice([32, 64, 128]),
            'activation': random.choice(['relu', 'gelu', 'leakyrelu']),
            'lr': random.choice([1e-4, 3e-4, 1e-3]),
            'epochs': random.choice([300, 500]),
            'weight_decay': random.choice([0.0, 1e-5, 1e-4, 1e-3])
        }
    elif surrogate_type == 'transformer':
        d_model = random.choice([16, 32, 64, 128])
        possible_nheads = [h for h in [1, 2, 4, 8] if d_model % h == 0]
        nhead = random.choice(possible_nheads)
        return {
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': random.choice([1, 2, 3, 4]),
            'dim_feedforward': random.choice([32, 64, 128, 256]),
            'dropout': random.choice([0.0, 0.1, 0.2, 0.3])
        }
    else:
        raise ValueError(f"Unknown surrogate type: {surrogate_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stats_file', type=str, help="Path to .stats file")
    parser.add_argument('surrogate_type', type=str, choices=['gin', 'transformer'])
    parser.add_argument('num_iterations', type=int)
    args = parser.parse_args()

    data = load_stats_file(args.stats_file)
    archive = extract_archive(data)
    print(f"Length of archive: {len(archive)}")

    # Default search space
    search_space = OFASearchSpace('eemobilenetv3', 192, 256, [0.1, 0.2, 1])
    arch_encoder_kwargs = {
        "num_of_blocks": search_space.num_blocks,
        "max_depth": max(search_space.depth),
        "min_res": search_space.resolution[0],
        "max_res": search_space.resolution[-1],
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_comb_rho = -np.inf
    best_hp = None

    try:
        for i in range(args.num_iterations):
            hp = sample_hyperparams(args.surrogate_type)
            print(f"\n--- Iteration {i+1}/{args.num_iterations} ---")
            print(f"Hyperparameters: {hp}")

            fold_acc_rhos = []
            fold_compl_rhos = []
            fold_rmses = []

            fold = 1
            for train_idx, val_idx in kf.split(archive):
                print(f"  Fold {fold}...")
                train_archive = [archive[idx] for idx in train_idx]
                val_archive = [archive[idx] for idx in val_idx]

                x_train, y_train_acc, y_train_compl = prepare_data(train_archive, search_space, args.surrogate_type)
                x_val, y_val_acc, y_val_compl = prepare_data(val_archive, search_space, args.surrogate_type)

                targets_2d_train = np.column_stack([y_train_acc, y_train_compl])
                
                # Fit predictor
                predictor = get_acc_predictor(
                    args.surrogate_type, x_train, targets_2d_train, 
                    arch_encoder_kwargs=arch_encoder_kwargs, **hp
                )

                # Predict
                preds = predictor.predict(x_val)
                acc_preds = preds[:, 0].flatten()
                compl_preds = preds[:, 1].flatten()

                acc_rmse, acc_rho, acc_tau = get_correlation(acc_preds, y_val_acc)
                compl_rmse, compl_rho, compl_tau = get_correlation(compl_preds, y_val_compl)

                fold_acc_rhos.append(acc_rho)
                fold_compl_rhos.append(compl_rho)
                fold_rmses.append((acc_rmse + compl_rmse) / 2) # just tracking average rmse for logging
                fold += 1

            mean_acc_rho = np.mean(fold_acc_rhos)
            mean_compl_rho = np.mean(fold_compl_rhos)
            mean_rmse = np.mean(fold_rmses)
            comb_rho = (mean_acc_rho + mean_compl_rho)/2

            print(f"  Mean Acc Rho: {mean_acc_rho:.4f}, Mean Compl Rho: {mean_compl_rho:.4f}, Comb Rho: {comb_rho:.4f}")

            if comb_rho > best_comb_rho:
                best_comb_rho = comb_rho
                best_hp = hp
                print("  >>> New Best! <<<")
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user (Ctrl+C). Gracefully exiting...")

    print("\n=================================")
    if best_hp is not None:
        print(f"Best Comb Rho: {best_comb_rho:.4f}")
        print(f"Best Hyperparameters: {best_hp}")
    else:
        print("No complete iterations finished. No best hyperparameters found.")
    print("=================================")

if __name__ == '__main__':
    main()


  