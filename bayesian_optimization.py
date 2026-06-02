import argparse
import ast
import json
import os
import sys
import numpy as np
from sklearn.model_selection import KFold

try:
    import optuna
except ImportError:
    print("[!] 'optuna' is not installed. Please run: pip install optuna")
    sys.exit(1)

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

    def objective(trial):
        # 1. Sample Hyperparameters using Optuna
        if args.surrogate_type == 'gin':
            hp = {
                'hidden_dim': trial.suggest_categorical('hidden_dim', [16, 32, 64, 128]),
                'output_dimension': trial.suggest_categorical('output_dimension', [16, 32, 64, 128]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
                'num_gnn_layers': trial.suggest_int('num_gnn_layers', 2, 4),
                'mlp_hidden_dim': trial.suggest_categorical('mlp_hidden_dim', [32, 64, 128]),
                'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'leakyrelu']),
                'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                'epochs': trial.suggest_int('epochs', 300, 500, step=50),
                # 'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            }
        elif args.surrogate_type == 'transformer':
            d_model = trial.suggest_categorical('d_model', [16, 32, 64, 128])
            possible_nheads = [h for h in [1, 2, 4, 8] if d_model % h == 0]
            nhead = trial.suggest_categorical('nhead', possible_nheads)
            hp = {
                'd_model': d_model,
                'nhead': nhead,
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'dim_feedforward': trial.suggest_categorical('dim_feedforward', [32, 64, 128, 256]),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5)
            }
        else:
            raise ValueError("Unknown surrogate")

        print(f"\n--- Trial {trial.number} ---")
        print(f"Testing config: {hp}")

        fold_acc_taus = []
        fold_compl_taus = []

        fold = 1
        for train_idx, val_idx in kf.split(archive):
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

            _, _, acc_tau = get_correlation(acc_preds, y_val_acc)
            _, _, compl_tau = get_correlation(compl_preds, y_val_compl)

            fold_acc_taus.append(acc_tau)
            fold_compl_taus.append(compl_tau)
            
            # Optuna can optionally prune unpromising trials here if needed,
            # but for 5 folds we just run them all.
            fold += 1

        mean_acc_tau = np.mean(fold_acc_taus)
        mean_compl_tau = np.mean(fold_compl_taus)

        optimization_metric = (mean_acc_tau + mean_compl_tau) / 2.0

        print(f"  Result -> Acc Tau: {mean_acc_tau:.4f}, Compl Tau: {mean_compl_tau:.4f}, Metric: {optimization_metric:.4f}")

        return optimization_metric

    # 2. Run Optimization
    # We maximize Kendall's Tau (via our custom metric)
    study = optuna.create_study(direction="maximize")
    try:
        study.optimize(objective, n_trials=args.num_iterations)
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user (Ctrl+C). Gracefully exiting...")

    # 3. Print Results
    print("\n=================================")
    if len(study.trials) > 0 and study.best_trial is not None:
        best_trial = study.best_trial
        print(f"Best Metric (Acc Tau - penalty): {best_trial.value:.4f}")
        print("Best Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("No complete iterations finished. No best hyperparameters found.")
    print("=================================")

if __name__ == '__main__':
    main()
