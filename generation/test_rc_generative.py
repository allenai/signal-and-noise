from oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY
from oe_eval.run_eval import run_eval


def main():
    # for task_name, task_cls in TASK_REGISTRY.items():
    #     TASKS_TO_INCLUDE = ['minerva'] # gsm, humaneval, mpbb
    #     if not any(task in task_name for task in TASKS_TO_INCLUDE): continue
    #     if '_selfc' in task_name: continue # no self-consistency tasks

    #     task_cls = TASK_REGISTRY[task_name]
    #     task_root = task_name.split(':')[0]

    args = {
        "model": ["EleutherAI/pythia-70m"],
        "revision": None,
        "trust_remote_code": None,
        "max_length": 2048,
        "model_type": "hf",
        "model_path": None,
        "model_args": ['{"metadata": {"alias": "pythia-70m"}}'],
        "task": [
            # '{"task_name": "gsm8k", "split": "test", "primary_metric": "exact_match", "num_shots": 8, "fewshot_source": "STD:GSM8k", "metadata": {"regimes": ["OLMES-v0.2"], "alias": "gsm8k::olmes"}}'
            '{"task_name": "minerva_math_number_theory"}'
        ],
        "limit": 20,
        "split": None,
        "random_subsample_seed": None,
        "num_shots": None,
        "fewshot_seed": None,
        "batch_size": "16",
        "max_batch_size": 32,
        "cached_output_dir": None,
        "num_recorded_inputs": 3,
        "gsheet": None,
        "hf_save_dir": None,
        "wandb_run_path": None,
        "no_datalake": False,
        "push_datalake": False,
        "datalake_tags": None,
        "check_datalake": False,
        "save_raw_requests": True,
        "recompute_metrics": False,
    }

    run_eval(args)


if __name__ == "__main__":
    main()
