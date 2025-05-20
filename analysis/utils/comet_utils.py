import os
import argparse
from comet_ml import API
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

def fetch_experiments(
    api_key: str,
    workspace: str,
    project_name: str,
    run_name: Optional[str] = None
) -> List[Dict[str, List[float]]]:
    """
    Fetch all experiments in a project, optionally filtered by run name, and their downstream metrics

    Args:
        api_key: Comet ML API key
        workspace: Comet ML workspace name
        project_name: Project name
        run_name: Optional name of the run to filter by

    Returns:
        List of dictionaries containing metrics for each matching experiment
    """
    # Initialize Comet ML API
    api = API(api_key=api_key)

    # Get all experiments in the project
    experiments = api.get(workspace=workspace, project_name=project_name)

    # Track unique run names
    run_names = set()
    matching_experiments = []

    for exp in tqdm(experiments, desc="Fetching experiments", position=0):
        exp_name = exp.get_name()
        run_names.add(exp_name)
        
        # Skip if run_name specified and doesn't match
        if run_name and exp.get_name() != run_name:
            continue

        downstream_metrics_x: Dict[str, List[int]] = {}
        downstream_metrics_y: Dict[str, List[float]] = {}

        for item in tqdm(exp.get_metrics(), desc=f"Fetching metrics for {exp_name}", position=1):
            if item['step'] is None:
                continue
            if not item['metricName'].startswith('eval/downstream'):
                continue

            downstream_metrics_x.setdefault(item['metricName'], []).append(int(item['step']))
            downstream_metrics_y.setdefault(item['metricName'], []).append(float(item['metricValue']))

        if downstream_metrics_y:  # Only include if it has downstream metrics
            matching_experiments.append({
                'id': exp.id,
                'name': exp.get_name(),
                'step': downstream_metrics_x,
                'metrics': downstream_metrics_y
            })

    # Print summary
    print(f"\nFound {len(run_names)} unique run names in project:")
    for name in sorted(run_names):
        print(f"- {name}")
        
    if run_name:
        print(f"\nFiltered for run name '{run_name}':")
        print(f"Found {len(matching_experiments)} matching experiments")
    else:
        print(f"\nFound {len(matching_experiments)} total experiments with downstream metrics")

    return matching_experiments

def save_metrics_to_csv(experiments: List[Dict], output_dir: Path, out_filename: str):
    """
    Save all metrics data to a single CSV file

    Args:
        experiments: List of experiment dictionaries containing metrics 
        output_dir: Directory to save the CSV file
    """
    data = []
    for exp in experiments:
        for metric_name in exp['metrics']:
            for step, value in zip(exp['step'][metric_name], exp['metrics'][metric_name]):
                data.append({
                    'experiment_id': exp['id'],
                    'run_name': exp['name'], 
                    'metric_name': metric_name,
                    'step': step,
                    'value': value
                })

    if data:
        df = pd.DataFrame(data)
        out_filename = out_filename.replace('-', '_')
        csv_path = output_dir / f"{out_filename}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")

def plot_metric_comparison(
    metric_name: str,
    experiments: List[Dict],
    output_path: Path
):
    """
    Create visualization comparing a single metric across multiple experiments

    Args:
        metric_name: Name of the metric to plot
        experiments: List of experiment dictionaries containing metrics
        output_path: Path to save the plot
    """
    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot line for each experiment
    legend_labels = []
    for exp in experiments:
        if metric_name in exp['metrics']:
            df = pd.DataFrame({'step': exp['step'][metric_name], 'value': exp['metrics'][metric_name]})

            # Plot line
            sns.lineplot(data=df, x='step', y='value', alpha=0.7)
            legend_labels.append(f"{exp['name']} ({exp['id'][:8]})")

    # Customize plot
    clean_name = metric_name.replace('eval/downstream/', '')
    plt.title(clean_name, fontsize=14, pad=20)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    plt.legend(legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add statistics annotation
    all_values = []
    for exp in experiments:
        if metric_name in exp['metrics']:
            all_values.extend(exp['metrics'][metric_name])

    if all_values:
        stats_text = f"Overall Statistics:\n"
        stats_text += f"Mean: {pd.Series(all_values).mean():.4f}\n"
        stats_text += f"Std: {pd.Series(all_values).std():.4f}\n"
        stats_text += f"Min: {min(all_values):.4f}\n"
        stats_text += f"Max: {max(all_values):.4f}"

        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Adjust layout to accommodate legend
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    # Close the figure to free memory
    plt.close()

def plot_all_metrics(experiments: List[Dict], output_dir: Path):
    """
    Create visualizations for all metrics across experiments

    Args:
        experiments: List of experiment dictionaries containing metrics
        output_dir: Directory to save the plots
    """
    # Get unique metric names across all experiments
    metric_names = set()
    for exp in experiments:
        metric_names.update(exp['metrics'].keys())

    # Plot each metric
    for metric_name in sorted(metric_names):
        # Create clean filename from metric name
        clean_name = metric_name.replace('eval/downstream/', '')\
                               .replace('/', '_')\
                               .replace(' ', '_')\
                               .lower()
        output_path = output_dir / f"{clean_name}.png"

        # Plot and save individual metric comparison
        plot_metric_comparison(metric_name, experiments, output_path)

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Fetch and visualize Comet ML downstream metrics'
    )

    parser.add_argument(
        '--workspace',
        type=str,
        required=True,
        help='Comet ML workspace name'
    )

    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='Comet ML project name'
    )

    parser.add_argument(
        '--run-name',
        type=str,
        help='Optional: Name of the run to analyze. If not provided, analyzes all runs.'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save the output plots'
    )

    parser.add_argument(
        '--output-filename',
        type=str,
        help='Optional: Base filename for output CSVs. If not provided, uses project name.'
    )

    return parser.parse_args()

def main():
    """
    Main function to fetch and visualize metrics
    """
    # Parse arguments
    args = parse_arguments()

    # Get API key from environment
    api_key = os.getenv('COMET_API_KEY')
    if not api_key:
        raise ValueError(
            "Please set the COMET_API_KEY environment variable"
        )

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch experiments and their metrics
    experiments = fetch_experiments(
        api_key=api_key,
        workspace=args.workspace,
        project_name=args.project,
        run_name=args.run_name
    )

    if not experiments:
        return
    
    output_filename = args.project if not args.output_filename else args.output_filename
    
    # First save all data to CSV files
    save_metrics_to_csv(experiments, output_dir, output_filename)

if __name__ == "__main__":
    main()
