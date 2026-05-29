import argparse
import pandas as pd
import numpy as np
import os
from pymoo.factory import get_performance_indicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def _calc_hv(ref_pt, F, normalized=True):
    # calculate hypervolume on the non-dominated set of F
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    nd_F = F[front, :]
    ref_point = 1.01 * ref_pt
    hv = get_performance_indicator("hv", ref_point=ref_point).calc(nd_F)
    if normalized:
        hv = hv / np.prod(ref_point)
    return hv

def calculate_hypervolumes(csv_paths, target_macs, output_csv, hv_type="old"):
    dataframes = []
    filenames = []
    
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        # Ensure it's sorted by iteration just in case
        df = df.sort_values(by='iteration')
        dataframes.append(df)
        
        filename = os.path.splitext(os.path.basename(csv_path))[0]
        filenames.append(filename)
        
    # Compute global reference point across all files (even if it's just 1)
    all_Fs = []
    for df in dataframes:
        avg_macs = df['avg_macs'].values
        top1_errors = df['top1_error'].values
        if hv_type == "old":
            F = np.column_stack((top1_errors, avg_macs))
        elif hv_type == "new":
            maep_errors = np.absolute((avg_macs - target_macs) / target_macs) 
            adjusted_top1_errors = top1_errors + 0.2*maep_errors
            F = np.column_stack((adjusted_top1_errors, maep_errors))
        else:
            raise ValueError(f"Invalid hv_type: {hv_type}")
        all_Fs.append(F)
    global_F = np.vstack(all_Fs)
    global_ref_pt = np.array([np.max(global_F[:, 0]), np.max(global_F[:, 1])])
    # global_ref_pt = np.array([100, 200]) # max maep = 300, 160 = 100 + 0.2*300
    print(f"Reference: {global_ref_pt}")
    # TEST_point = np.array([np.min(global_F[:, 0]), np.min(global_F[:, 1])])
    # print(f"TEST POint: {TEST_point}")

    final_df = None
    
    for df, filename in zip(dataframes, filenames):
        max_iter = int(df['iteration'].max())
        results = []
        
        for current_iter in range(max_iter + 1):
            # The archive at current_iter contains all architectures evaluated up to current_iter
            current_archive = df[df['iteration'] <= current_iter]
            if len(current_archive) == 0:
                continue
            
            avg_macs = current_archive['avg_macs'].values
            top1_errors = current_archive['top1_error'].values

            # Construct F
            if hv_type == "old":
                F = np.column_stack((top1_errors, avg_macs))
            elif hv_type == "new":
                 # Calculate maep_errors
                maep_errors = np.absolute((avg_macs - target_macs) / target_macs) 
                adjusted_top1_errors = top1_errors + 0.2*maep_errors
                F = np.column_stack((adjusted_top1_errors, maep_errors ))

            # Reference point (nadir point) for calculating hypervolume based on the entire run(s)
            ref_pt = global_ref_pt
            # Calculate hypervolume
            hv = _calc_hv(ref_pt, F)
            
            col_name = f'{filename}_hv' if len(csv_paths) > 1 else 'hypervolume'
            
            results.append({
                'iteration': current_iter,
                col_name: hv
            })
            
        run_df = pd.DataFrame(results)
        if final_df is None:
            final_df = run_df
        else:
            final_df = pd.merge(final_df, run_df, on='iteration', how='outer')
            
    final_df = final_df.sort_values(by='iteration')
    # Cast iteration back to int in case outer merge made it float
    final_df['iteration'] = final_df['iteration'].astype(int)
    final_df.to_csv(output_csv, index=False)
    print(f"Hypervolume calculation complete. Saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate hypervolume over iterations.')
    parser.add_argument('--csv', nargs='+', type=str, required=True, help='Path(s) to all_evaluated_architectures.csv')
    parser.add_argument('--target_macs', type=float, required=True, help='Target MACs constraint used during the search')
    parser.add_argument('--output', type=str, default='hypervolume_over_iterations.csv', help='Path to save the output CSV')
    parser.add_argument('--hv_type', type=str, choices=['old', 'new'], default='new', help='how to compute. VALUES=["old", "new"]')

    args = parser.parse_args()
    
    calculate_hypervolumes(args.csv, args.target_macs, args.output, args.hv_type)
