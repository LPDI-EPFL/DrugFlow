import argparse
import pickle
import sys
from pathlib import Path

basedir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(basedir))

from src.sbdd_metrics.evaluation import compute_all_metrics_drugflow

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', type=Path, required=True, help='Directory with samples')
    p.add_argument('--out_dir', type=str, required=True, help='Output directory')
    p.add_argument('--reference_smiles', type=str, default=None, help='Path to the .npy file with reference SMILES (optional)')
    p.add_argument('--gnina', type=str, default=None, help='Path to the gnina binary file (optional)')
    p.add_argument('--reduce', type=str, default=None, help='Path to the reduce binary file (optional)')
    p.add_argument('--n_samples', type=int, default=None, help='Top-N sampels to evaluate (optional)')
    p.add_argument('--exclude', type=str, nargs='+', default=[], help='Evaluator IDs to exclude')
    p.add_argument('--job_id', type=int, default=0, help='Job ID')
    p.add_argument('--n_jobs', type=int, default=1, help='Number of jobs')
    args = p.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True, parents=True)
    if args.job_id == 0 and args.n_jobs == 1:
        out_detailed_table = Path(args.out_dir, 'metrics_detailed.csv')
        out_aggregated_table = Path(args.out_dir, 'metrics_aggregated.csv')
        out_distributions_file = Path(args.out_dir, 'metrics_data.pkl')
    else:
        out_detailed_table = Path(args.out_dir, f'metrics_detailed_{args.job_id}.csv')
        out_aggregated_table = Path(args.out_dir, f'metrics_aggregated_{args.job_id}.csv')
        out_distributions_file = Path(args.out_dir, f'metrics_data_{args.job_id}.pkl')
    
    if out_detailed_table.exists() and out_aggregated_table.exists() and out_distributions_file.exists():
        print(f'Data already exist. Terminating')
        sys.exit(0)

    print(f'Evaluating: {args.in_dir}')
    data, detailed, aggregated = compute_all_metrics_drugflow(
        in_dir=args.in_dir,
        gnina_path=args.gnina,
        reduce_path=args.reduce,
        reference_smiles_path=args.reference_smiles,
        n_samples=args.n_samples,
        exclude_evaluators=args.exclude,
        job_id=args.job_id,
        n_jobs=args.n_jobs,
    )

    detailed.to_csv(out_detailed_table, index=False)
    aggregated.to_csv(out_aggregated_table, index=False)
    with open(Path(out_distributions_file), 'wb') as f:
        pickle.dump(data, f)