import pandas as pd
import numpy as np
from rich.table import Table
from rich.console import Console
from rich import box
from tqdm import tqdm

from snr.download.hf import pull_predictions_from_hf
from snr.dataloader import get_slice
from snr.metrics import decision_acc_fast
from snr.ladder_wrapper import run_ladder
from snr.metrics import signal_to_noise_ratio
from snr.constants.ladder import LADDER_MODEL_NAMES
from snr.constants.signal import SNR_MODELS


def compute_decision_accuracy(df, task, small_size):
    scores_small = get_slice(df, size=small_size, task=task)
    scores_target = get_slice(df, size="1B", task=task, step=69369)

    # Get the score at the highest step for each mix
    scores_small = scores_small.loc[scores_small.groupby("mix")["step"].idxmax()]

    decision_acc = decision_acc_fast(
        scores_small=scores_small.sort_values("model")["primary_score"],
        scores_target=scores_target.sort_values("model")["primary_score"],
    )

    return decision_acc


def compute_scaling_law_error(df, task, large_size):
    if large_size == "7B":
        target_model = "peteish7"
    elif large_size == "13B":
        target_model = "peteish13-highlr"
    else:
        raise ValueError(large_size)

    _, _, error = run_ladder(df, task, train_models=LADDER_MODEL_NAMES, eval_models=[target_model])

    return error


def compute_snr_small_scale(df, task, small_size):
    scores_df = get_slice(df, size=small_size, task=task).sort_values("step")

    # numpy array of final 5 scores in shape (mix, checkpoint)
    scores_arr = np.array(
        [lst[-5:] for lst in scores_df.groupby("mix")["primary_score"].apply(list)]
    )

    signal = [np.mean(scores) for scores in scores_arr]
    noise = scores_arr.flatten()

    snr = signal_to_noise_ratio(signal, noise)

    return snr


def compute_snr_large_scale(df, task, large_size):
    if large_size == "1B":
        signal_models = "olmo2_1b"
        noise_model = "peteish1"
    elif large_size == "7B":
        signal_models = "olmo2_7b"
        noise_model = "peteish7"
    elif large_size == "13B":
        signal_models = "olmo2_13b"
        noise_model = "peteish13-highlr"
    elif large_size == "32B":
        signal_models = "olmo2_32b"
        noise_model = "peteish32"
    else:
        raise ValueError(large_size)

    signal_models = SNR_MODELS[signal_models]["models"]
    noise_df = get_slice(df, model=noise_model, task=task)

    signal_df = df[df["model_path"].isin(signal_models) & (df["task"] == task)]

    signal = list(signal_df["primary_score"])
    noise = list(noise_df.sort_values("step")["primary_score"])[-30:]

    snr = signal_to_noise_ratio(signal, noise)

    return snr


def calculate_results(df, tasks, small_sizes, large_sizes_scaling, large_sizes_snr):
    results = []

    for task in tqdm(tasks, desc="Running analysis"):
        row = {"Task": task}

        # Decision Accuracy
        decision_acc_group = {}
        for small_size in small_sizes:
            decision_acc = compute_decision_accuracy(df, task, small_size)
            decision_acc_group[small_size] = decision_acc
        row["Decision Accuracy"] = decision_acc_group

        # Scaling Law Error
        scaling_law_group = {}
        for large_size in large_sizes_scaling:
            scaling_law_error = compute_scaling_law_error(df, task, large_size)
            scaling_law_group[large_size] = scaling_law_error
        row["Scaling Law Error"] = scaling_law_group

        # Small scale SNR
        snr_group = {}
        for small_size in small_sizes:
            snr_small = compute_snr_small_scale(df, task, small_size)
            snr_group[small_size] = snr_small
        
        # Large scale SNR
        for large_size in large_sizes_snr:
            snr_large = compute_snr_large_scale(df, task, large_size)
            snr_group[large_size] = snr_large
        row["SNR"] = snr_group

        results.append(row)
    return results


def render_table(results, small_sizes, large_sizes_scaling, large_sizes_snr):
    table = Table(title="Signal-and-Noise Analysis by Task", box=box.ASCII)

    # Add header
    decision_acc_headers = [f"{size}" for size in small_sizes]
    scaling_law_headers = [f"{size}" for size in large_sizes_scaling]
    snr_headers = [f"{size}" for size in small_sizes + large_sizes_snr]
    table.add_column("Task", justify="left")
    for size in decision_acc_headers:
        table.add_column(f"Decision\nAcc\n{size}", justify="left")
    for size in scaling_law_headers:
        table.add_column(f"Scaling\nLaw Err\n{size}", justify="left")
    for size in snr_headers:
        table.add_column(f"SNR\n{size}", justify="left")

    # Sort results alphabetically by task name
    sorted_results = sorted(results, key=lambda row: str(row["Task"]).lower())

    # Add rows
    for row in sorted_results:
        row_values = [str(row["Task"])]

        for size in decision_acc_headers:
            val = row["Decision Accuracy"].get(size, "")
            if isinstance(val, float):
                row_values.append(f"{int(round(val * 100))}%")
            else:
                row_values.append(str(val))

        for size in scaling_law_headers:
            val = row["Scaling Law Error"].get(size, "")
            if isinstance(val, float):
                row_values.append(f"{val * 100:.1f}%")
            else:
                row_values.append(str(val))

        for size in snr_headers:
            val = row["SNR"].get(size, "")
            if isinstance(val, float):
                row_values.append(f"{val:.1f}")
            else:
                row_values.append(str(val))
        table.add_row(*row_values)

    console = Console()
    console.print(table)


def main():
    local_path = pull_predictions_from_hf("allenai/signal-and-noise", split_name="core")
    df = pd.read_parquet(local_path)

    tasks = [
        "minerva", "mmlu", "agi_eval", "arc_challenge", "arc_easy", "boolq", 
        "csqa", "hellaswag", "openbookqa", "piqa", "socialiqa", "winogrande", 
        "gsm8k", "mbpp", "mbppplus", "codex_humaneval", "codex_humanevalplus", 
        "autobencher", "gsm_plus", "gsm_symbolic_main", "gsm_symbolic_p1", 
        "gsm_symbolic_p2", "medmcqa", "minerva_math_500",
    ]

    small_sizes = ["150M", "300M", "750M"] # ["4M", "20M", "60M", "90M", "150M", "300M", "530M", "750M"]
    large_sizes_scaling = ["7B", "13B"]
    large_sizes_snr = ["1B", "7B", "13B", "32B"]

    results = calculate_results(df, tasks, small_sizes, large_sizes_scaling, large_sizes_snr)
    render_table(results, small_sizes, large_sizes_scaling, large_sizes_snr)


if __name__ == "__main__":
    main()
