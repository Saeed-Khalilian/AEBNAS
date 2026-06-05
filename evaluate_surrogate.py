"""
evaluate_surrogate.py
=====================
Fit a surrogate model on each iter_*.stats archive from an AEBNAS run and
evaluate on an external held-out test set.

Usage
-----
    python evaluate_surrogate.py \
        --run_dir results/cifar10/run_20260531_025429 \
        --test_path iter_1_truncated.stats \
        --predictor gin

The script produces a CSV (printed to stdout and saved as
``surrogate_eval_<predictor>.csv`` inside *run_dir*) with columns:

    iter, n_train, acc_rmse, acc_rho, acc_tau, compl_rmse, compl_rho, compl_tau
"""

import argparse
import ast
import json
import os
import random
import re
import sys
import warnings

warnings.simplefilter("ignore")

import numpy as np
import torch

from acc_predictor.factory import get_acc_predictor
from search_space.ofa_search_space import OFASearchSpace
from utils import get_correlation


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    """Set all random seeds for reproducible surrogate training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_stats_file(path):
    """Load an iter_*.stats JSON file and return the parsed dict.

    Handles both strict JSON and Python-literal files (single-quoted keys etc.)
    following the same logic as print_archive_len.py.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try Python literal first (handles single-quoted keys/tuples)
    try:
        return ast.literal_eval(content)
    except Exception:
        pass

    # Fallback to standard JSON
    try:
        return json.loads(content)
    except Exception as e:
        raise RuntimeError(f"Cannot parse {path}: {e}") from e


def extract_archive(data):
    """Return the list of archive entries from a parsed stats dict.

    Each entry is ``[config_dict, top1_err, complexity_list, util_list]``.
    """
    if "archive" not in data:
        raise KeyError(f"'archive' key missing.  Available keys: {list(data.keys())}")
    return data["archive"]


def build_search_space(exp_config):
    """Reconstruct an OFASearchSpace from the saved experiment config."""
    params = exp_config.get("parameters", exp_config)
    supernet_path = params.get("supernet_path", "eembv3")
    lr = params.get("lr", 192)
    ur = params.get("ur", 256)
    threshold = params.get("threshold", [0.1, 0.2, 1])

    # Determine supernet type from supernet_path (mirrors AEBNAS.__init__)
    if "w1.0" in supernet_path or "w1.2" in supernet_path:
        supernet = "mobilenetv3"
    elif "eembv3" in supernet_path:
        supernet = "eemobilenetv3"
    elif "resnet50_he_d" in supernet_path:
        supernet = "resnet50_he"
    elif "resnet50_d" in supernet_path:
        supernet = "resnet50"
    else:
        # Default to eemobilenetv3 for safety
        print(f"WARNING: Could not determine supernet from '{supernet_path}', defaulting to eemobilenetv3",
              file=sys.stderr)
        supernet = "eemobilenetv3"

    return OFASearchSpace(supernet, lr, ur, threshold)


def iter_stats_files(run_dir):
    """Yield (iteration_number, file_path) for each iter_N.stats in run_dir,
    sorted by iteration number.
    """
    pattern = re.compile(r"^iter_(\d+)\.stats$")
    entries = []
    for fname in os.listdir(run_dir):
        m = pattern.match(fname)
        if m:
            entries.append((int(m.group(1)), os.path.join(run_dir, fname)))
    entries.sort(key=lambda x: x[0])
    return entries


# ──────────────────────────────────────────────────────────────────────────────
# Core: prepare data, fit, predict, evaluate
# ──────────────────────────────────────────────────────────────────────────────

COMBINED_PREDICTORS = ("gin", "transformer")


def prepare_data(archive, search_space, predictor_type):
    """Convert an archive list into surrogate inputs and targets.

    Returns
    -------
    inputs : array-like
        Encoded integer vectors for vector-based surrogates **or** raw config
        dicts for combined (GIN/Transformer) surrogates.
    acc_targets : np.ndarray
        Top-1 error values.
    compl_targets : np.ndarray
        Average MACs (dot product of complexity × util).
    """
    acc_targets = np.array([entry[1] for entry in archive])
    compl_targets = np.array([np.dot(entry[2], entry[3]) for entry in archive])

    if predictor_type in COMBINED_PREDICTORS:
        # Combined predictors expect raw config dicts (np.array of dicts)
        inputs = np.array([entry[0] for entry in archive])
    else:
        # Vector-based predictors expect encoded integer vectors
        inputs = np.array([search_space.encode(entry[0]) for entry in archive])

    return inputs, acc_targets, compl_targets


def fit_predictor(predictor_type, inputs, acc_targets, compl_targets, search_space):
    """Fit surrogate(s) and return predictor(s).

    For combined predictors (GIN / Transformer): fits one model predicting both
    accuracy and complexity.

    For others (MLP, RBF, CART, …): fits separate accuracy and complexity
    predictors.

    Returns
    -------
    acc_predictor, compl_predictor
        compl_predictor is None for combined predictors.
    """
    if predictor_type in COMBINED_PREDICTORS:
        targets_2d = np.column_stack([acc_targets, compl_targets])
        arch_encoder_kwargs = {
            "num_of_blocks": search_space.num_blocks,
            "max_depth": max(search_space.depth),
            "min_res": search_space.resolution[0],
            "max_res": search_space.resolution[-1],
        }
        predictor = get_acc_predictor(
            predictor_type, inputs, targets_2d,
            arch_encoder_kwargs=arch_encoder_kwargs,
        )
        return predictor, None
    else:
        acc_pred = get_acc_predictor(predictor_type, inputs, acc_targets)
        compl_pred = get_acc_predictor("mlp", inputs, compl_targets)
        return acc_pred, compl_pred


def evaluate_on_test(acc_predictor, compl_predictor, predictor_type,
                     test_inputs, test_acc_targets, test_compl_targets):
    """Predict on test data and return metrics dict."""

    if predictor_type in COMBINED_PREDICTORS:
        preds = acc_predictor.predict(test_inputs)  # shape (N, 2)
        acc_preds = preds[:, 0].flatten()
        compl_preds = preds[:, 1].flatten()
    else:
        acc_preds = acc_predictor.predict(test_inputs).flatten()
        compl_preds = compl_predictor.predict(test_inputs).flatten()

    acc_rmse, acc_rho, acc_tau = get_correlation(acc_preds, test_acc_targets)
    compl_rmse, compl_rho, compl_tau = get_correlation(compl_preds, test_compl_targets)

    return {
        "acc_rmse": acc_rmse, "acc_rho": acc_rho, "acc_tau": acc_tau,
        "compl_rmse": compl_rmse, "compl_rho": compl_rho, "compl_tau": compl_tau,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate surrogate predictors on unseen test data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to an AEBNAS run directory containing iter_*.stats and exp_config.json.",
    )
    parser.add_argument(
        "--test_path", type=str, required=True,
        help="Path to a stats file whose archive is used as the held-out test set.",
    )
    parser.add_argument(
        "--predictor", type=str, default="mlp",
        choices=["rbf", "carts", "gp", "mlp", "as", "gin", "transformer"],
        help="Surrogate model type (default: mlp).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path.  Default: <run_dir>/surrogate_eval_<predictor>.csv",
    )
    parser.add_argument(
        "--seeds", type=int, nargs='+', default=[42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021],
        help="List of random seeds for reproducible surrogate training (default: 10 seeds).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    test_path = args.test_path
    predictor_type = args.predictor

    # ── Load experiment config ------------------------------------------------
    exp_config_path = os.path.join(run_dir, "exp_config.json")
    if os.path.exists(exp_config_path):
        with open(exp_config_path, "r") as f:
            exp_config = json.load(f)
    else:
        print(f"WARNING: {exp_config_path} not found, using defaults.", file=sys.stderr)
        exp_config = {}

    search_space = build_search_space(exp_config)
    print(f"Search space: supernet={search_space.supernet}, n_var={search_space.n_var}")

    # ── Load test set ---------------------------------------------------------
    print(f"Loading test set from {test_path} ...")
    test_data = load_stats_file(test_path)
    test_archive = extract_archive(test_data)
    test_inputs, test_acc_targets, test_compl_targets = prepare_data(
        test_archive, search_space, predictor_type
    )
    print(f"Test set: {len(test_archive)} architectures")

    # ── Verify no data leakage in final training set --------------------------
    stats_files = iter_stats_files(run_dir)
    if not stats_files:
        print(f"ERROR: No iter_*.stats files found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Checking for data leakage between test set and final archive ({stats_files[-1][1]})...")
    test_encoded_set = set(str(search_space.encode(entry[0])) for entry in test_archive)
    
    last_iter, last_stats_path = stats_files[-1]
    last_train_data = load_stats_file(last_stats_path)
    last_train_archive = extract_archive(last_train_data)
    
    overlap = []
    for entry in last_train_archive:
        encoded_str = str(search_space.encode(entry[0]))
        if encoded_str in test_encoded_set:
            overlap.append(entry[0])
            
    if overlap:
        print(f"WARNING: Data leakage detected! {len(overlap)} architectures found in both testing and training sets.")
        print("Overlapping architectures:")
        for arch in overlap:
            print(arch)
    else:
        print("No overlap between testing and training sets detected.")

    # ── Iterate over training snapshots ---------------------------------------

    header = ("iter,n_train,"
              "acc_rmse_mean,acc_rmse_std,acc_rho_mean,acc_rho_std,acc_tau_mean,acc_tau_std,"
              "compl_rmse_mean,compl_rmse_std,compl_rho_mean,compl_rho_std,compl_tau_mean,compl_tau_std")
    rows = [header]
    print(f"\n{'='*140}")
    print(f"{'iter':>5}  {'n_train':>7}  {'acc_rmse':>15}  {'acc_rho':>15}  {'acc_tau':>15}  "
          f"{'c_rmse':>15}  {'c_rho':>15}  {'c_tau':>15}")
    print(f"{'='*140}")

    for it_num, stats_path in stats_files:
        print("────────────────────────────────────────────────────────────────")#print(f"\n── Iteration {it_num} ({os.path.basename(stats_path)}) ──")
        train_data = load_stats_file(stats_path)
        train_archive = extract_archive(train_data)
        n_train = len(train_archive)

        train_inputs, train_acc_targets, train_compl_targets = prepare_data(
            train_archive, search_space, predictor_type
        )

        iter_metrics = {
            "acc_rmse": [], "acc_rho": [], "acc_tau": [],
            "compl_rmse": [], "compl_rho": [], "compl_tau": []
        }

        for seed in args.seeds:
            # Seed before each fit for independent reproducibility per iteration
            set_seed(seed)

            # Fit
            try:
                acc_pred, compl_pred = fit_predictor(
                    predictor_type, train_inputs, train_acc_targets,
                    train_compl_targets, search_space
                )
            except Exception as e:
                print(f"  FAILED to fit on iter {it_num} with seed {seed}: {e}", file=sys.stderr)
                continue

            # Evaluate on test set
            metrics = evaluate_on_test(
                acc_pred, compl_pred, predictor_type,
                test_inputs, test_acc_targets, test_compl_targets,
            )
            for k, v in metrics.items():
                iter_metrics[k].append(v)
        
        if not iter_metrics["acc_rmse"]:
            row = f"{it_num},{n_train}" + ",NaN" * 12
            rows.append(row)
            continue

        agg_metrics = {}
        for k, v_list in iter_metrics.items():
            agg_metrics[f"{k}_mean"] = np.mean(v_list)
            agg_metrics[f"{k}_std"] = np.std(v_list)

        row = (f"{it_num},{n_train},"
               f"{agg_metrics['acc_rmse_mean']:.6f},{agg_metrics['acc_rmse_std']:.6f},"
               f"{agg_metrics['acc_rho_mean']:.6f},{agg_metrics['acc_rho_std']:.6f},"
               f"{agg_metrics['acc_tau_mean']:.6f},{agg_metrics['acc_tau_std']:.6f},"
               f"{agg_metrics['compl_rmse_mean']:.6f},{agg_metrics['compl_rmse_std']:.6f},"
               f"{agg_metrics['compl_rho_mean']:.6f},{agg_metrics['compl_rho_std']:.6f},"
               f"{agg_metrics['compl_tau_mean']:.6f},{agg_metrics['compl_tau_std']:.6f}")
        rows.append(row)

        print(f"  {it_num:>5}  {n_train:>7}  "
              f"{agg_metrics['acc_rmse_mean']:>7.4f}±{agg_metrics['acc_rmse_std']:<6.4f} "
              f"{agg_metrics['acc_rho_mean']:>7.4f}±{agg_metrics['acc_rho_std']:<6.4f} "
              f"{agg_metrics['acc_tau_mean']:>7.4f}±{agg_metrics['acc_tau_std']:<6.4f} "
              f"{agg_metrics['compl_rmse_mean']:>7.4f}±{agg_metrics['compl_rmse_std']:<6.4f} "
              f"{agg_metrics['compl_rho_mean']:>7.4f}±{agg_metrics['compl_rho_std']:<6.4f} "
              f"{agg_metrics['compl_tau_mean']:>7.4f}±{agg_metrics['compl_tau_std']:<6.4f}")

    # ── Save CSV --------------------------------------------------------------
    out_path = args.output or os.path.join(run_dir, f"surrogate_eval_{predictor_type}.csv")
    csv_content = "\n".join(rows) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(csv_content)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
