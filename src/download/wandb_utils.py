import csv
import json
import os.path
import re
from typing import List

from tqdm import tqdm

import wandb
from wandb.apis.public import Run

from utils.scaling_laws import (
    downstream,
    downstream_bpb,
    downstream_newline,
    downstream_newline_bpb,
    v3_validation,
    validation,
)

RUN_PATH_RE = re.compile(r"^[^/]+/[^/]+/[^/]+$")
RUN_PATH_URL = re.compile(r"^https?://wandb.ai/([^/]+)/([^/]+)/runs/([^/]+)")


def parse_run_path(run_path: str) -> str:
    """For convenience, we allow run paths as well as URLs."""
    run_path = run_path.strip("/")
    if RUN_PATH_RE.match(run_path):
        return run_path

    m = RUN_PATH_URL.match(run_path)
    if m is not None:
        entity, project, run_id = m.groups()
        return f"{entity}/{project}/{run_id}"

    raise ValueError(f"Could not parse '{run_path}'")


def get_runs(run_paths: List[str]) -> List[wandb.apis.public.Run]:
    api = wandb.Api()
    all_wb_runs = []
    for run_path in run_paths:
        run_path = parse_run_path(run_path)
        entity, project, run_name = run_path.split("/")
        wb_path = f"{entity}/{project}"
        wb_filters = {"display_name": run_name}
        wb_runs = api.runs(path=wb_path, filters=wb_filters)
        print(f"Found {len(wb_runs)} matching runs in {wb_path}")
        all_wb_runs.extend(wb_runs)
    return all_wb_runs


def get_run_groups(y_axis: List) -> List:
    # expad multiple run short names
    if y_axis == ["eval/all-validation/CrossEntropyLoss"]:
        y_axis = [f"eval/{d}/CrossEntropyLoss" for d in validation]

    elif y_axis == ["eval/all-validation-and-bpb/CrossEntropyLoss"]:
        y_axis = [f"eval/{d}/CrossEntropyLoss" for d in validation] + [
            f"eval/downstream_bpb/{d}_bpb" for d in downstream_bpb
        ]

    elif y_axis == ["eval/all-v3-validation/CrossEntropyLoss"]:
        y_axis = [f"eval/{d}/CrossEntropyLoss" for d in v3_validation]

    elif y_axis == ["eval/downstream/all"]:
        y_axis = [f"eval/downstream/{d}" for d in downstream]

    elif y_axis == ["eval/validation-and-bpb-and-downstream"]:
        y_axis = (
            [f"eval/{d}/CrossEntropyLoss" for d in validation]
            + [f"eval/downstream_bpb/{d}_bpb" for d in downstream_bpb]
            + [f"eval/downstream/{d}" for d in downstream]
        )

    elif y_axis == ["eval/validation-and-bpb-and-downstream-newline"]:
        y_axis = (
            [f"eval/{d}/CrossEntropyLoss" for d in validation]
            + [f"eval/downstream_bpb/{d}_bpb" for d in downstream_bpb]
            + [f"eval/downstream/{d}" for d in downstream]
            + [f"eval/downstream_bpb/{d}_bpb" for d in downstream_newline_bpb]
            + [f"eval/downstream/{d}" for d in downstream_newline]
        )

    return y_axis


def download_wb(x_axis, y_axis, output_path, wandb_names, additional_keys=[]):
    y_axis = get_run_groups(y_axis)

    wb_runs: List[Run] = get_runs(wandb_names)

    print(wandb_names)
    print(wb_runs)

    print("Downloading the data from the following wandb runs:\n", "\n  ".join([str(run) for run in wb_runs]))

    dirname = os.path.dirname(output_path)
    if dirname: os.makedirs(dirname, exist_ok=True)

    keys = [x_axis] + y_axis + additional_keys
    print(f'Fetching keys: {keys}')

    with open(output_path, "w") as file_ref:
        writer = csv.DictWriter(
            file_ref,
            fieldnames=keys,
        )
        writer.writeheader()

        rows = []
        for wb_run in tqdm(wb_runs, desc="Downloading runs"):
            tqdm.write(f"Processing {wb_run.name}")
            
            history = wb_run.scan_history(
                keys=keys,
                page_size=10000, # page_size cannot be too big, it will make it faster but it will start to downsample
                # page_size=10000000, # page_size cannot be too big, it will make it faster but it will start to downsample
            )

            for wb_step in history:
                print(wb_step)

            try:
                # Calculate additional values using config, add to history
                config = json.loads(wb_run.json_config)
                
                batch_size, max_seq_len = config["global_train_batch_size"]["value"], config["model"]["value"]["max_sequence_length"]
                batch_size_in_tokens = batch_size * max_seq_len

                for wb_step in history:
                    wb_step["learning_rate_peak"] = config["optimizer"]["value"]["learning_rate"]
                    # With certain run restarts, we also update the batch size.
                    wb_step["batch_size_in_tokens"] = batch_size_in_tokens
            except KeyError as e:
                print(f"Failed to process {wb_run.name}: {e}")

            for wb_step in history:
                rows.append(wb_step)

        if len(rows) is 0:
            raise RuntimeError(f'Found no wandb history for run "{wb_run.name}" and keys {keys}. Make sure (1) run name is valid and (2) ALL keys are valid (or it will silent fail).')

        # Remove duplicate rows
        row_by_key = {}
        for row in rows:
            key = row[x_axis]
            row_by_key[key] = row
        rows = list(row_by_key.values())

        rows = sorted(rows, key=lambda x: x[x_axis])
        writer.writerows(rows)
