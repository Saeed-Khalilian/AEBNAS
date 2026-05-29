import os
import json
import argparse
import csv

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
                    # c dominates other if c is <= other in all dims, and < in at least one
                    # meaning other is dominated by c
                    # But the algorithm from before kept points that are strictly better in at least one dimension.
                    # A better and standard way:
                    # j is dominated by i if i <= j in all objectives and i < j in at least one objective.
                    if c[0] <= other[0] and c[1] <= other[1] and (c[0] < other[0] or c[1] < other[1]):
                        is_efficient[j] = False
    return is_efficient

def extract_pareto_front(run_dir):
    """
    Reads the statistics from the final iteration to find all evaluated models,
    then computes the pareto front (Top-1 Accuracy vs Average MACs).
    """
    stats_files = [f for f in os.listdir(run_dir) if f.startswith("iter_") and f.endswith(".stats")]
    if not stats_files:
        print(f"No iter_*.stats files found in {run_dir}")
        return

    # To find the latest iteration
    # iter_N.stats
    def get_iter_num(filename):
        return int(filename.split('_')[1].split('.')[0])

    latest_stat_file = max(stats_files, key=get_iter_num)
    stat_path = os.path.join(run_dir, latest_stat_file)
    print(f"Reading architectures from: {latest_stat_file}")
    
    with open(stat_path, 'r') as f:
        data = json.load(f)
    
    if 'archive' not in data:
        print(f"No 'archive' key found in {latest_stat_file}")
        return

    archive = data['archive']
    print(f"Total architectures evaluated: {len(archive)}")
    
    # Process the archive
    # Format of x in archive: (arch, top1_error, sec_obj, util)
    results = []
    costs = []
    
    for idx, x in enumerate(archive):
        arch = x[0]
        top1_error = x[1]
        top1_acc = 100.0 - top1_error
        
        # Calculate average MACs if it's an early-exit network
        if len(x) >= 4 and isinstance(x[2], (list, tuple)) and isinstance(x[3], (list, tuple)):
            avg_macs = sum(a * b for a, b in zip(x[2], x[3]))
        elif len(x) >= 3 and not isinstance(x[2], (list, tuple)):
            avg_macs = x[2] # standard NAS without early exits
        else:
            avg_macs = sum(x[2]) / len(x[2]) # fallback
            
        iteration = 0 if idx < 100 else 1 + (idx - 100) // 8
        results.append({
            'arch': arch,
            'top1_acc': top1_acc,
            'top1_error': top1_error,
            'avg_macs': avg_macs,
            'iteration': iteration
        })
        
        # We want to MINIMIZE error and MINIMIZE average macs
        costs.append([top1_error, avg_macs])
        
    is_pareto = get_non_dominated(costs)
    
    pareto_results = [results[i] for i in range(len(results)) if is_pareto[i]]
    
    # Sort the pareto front by average MACs (ascending)
    pareto_results = sorted(pareto_results, key=lambda r: r['avg_macs'])
    
    print("\n" + "="*50)
    print(f"Pareto Front (Non-Dominated Architectures): {len(pareto_results)}")
    print("="*50)
    print(f"{'Top-1 Accuracy (%)':<20} | {'Average MACs':<15} | {'Iteration':<10}")
    print("-" * 53)
    for r in pareto_results:
        print(f"{r['top1_acc']:<20.2f} | {r['avg_macs']:<15.2f} | {r['iteration']:<10}")
    
    # Use built-in csv to avoid any C-extension/pandas hanging issues on remote servers
    out_csv = os.path.join(run_dir, "pareto_front.csv")
    if pareto_results:
        keys = pareto_results[0].keys()
        with open(out_csv, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(pareto_results)
    print(f"\nSaved pareto front to: {out_csv}")
    
    # Also save the full archive to a CSV file for convenience
    out_all_csv = os.path.join(run_dir, "all_evaluated_architectures.csv")
    if results:
        keys = results[0].keys()
        with open(out_all_csv, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
    print(f"Saved all architectures to: {out_all_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract Pareto Front from AEBNAS run')
    parser.add_argument('run_dir', type=str, help='Path to the run directory (e.g., results/cifar10/run_...)')
    args = parser.parse_args()
    
    extract_pareto_front(args.run_dir)
