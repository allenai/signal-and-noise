import os
from typing import List, Literal

from wandb.apis.public import Run

from utils.wandb_utils import download_wb, get_runs
from utils import DATA_DIR


def get_name_size_length(run_name: str):
    run_name = run_name.split("/")[-1]
    size, length = run_name.split("-")[-2:]
    return run_name, size, length


def download_model_ladder(run_name: Literal['amberish', 'peteish']='peteish', use_cache=False):
    ''' 
    Download Akshita's model ladder runs 
    https://wandb.ai/ANON-llm/olmo-ladder/table
    '''
    import itertools
    if run_name == 'peteish':
        model_tag = 'peteish-final'
        model_sizes = itertools.product(["190M", "370M", "600M", "760M", "1B"], ["1xC", "2xC", "5xC", "10xC"])
    elif run_name == 'amberish':
        model_tag = 'amberish-rulebased'
        model_sizes = itertools.product(["150M", "300M", "530M", "750M", "1B"], ["1xC", "2xC", "5xC", "10xC"])
    else:
        raise ValueError(run_name)
    
    WANDB_RESULTS = f"{DATA_DIR}/wandb/{model_tag}"

    run_names = [f"ANON-llm/olmo-ladder/{model_tag}-{size}-{length}" for size, length in model_sizes]

    file_paths = {}

    for run_name in run_names:
        if "peteish7" in run_name:
            output_name = "peteish7"
        else:
            output_name = run_name.split("/")[-1]
            name, size, length = get_name_size_length(run_name)
            output_name = f"{size}-{length}"
        output_path = f"{WANDB_RESULTS}/{output_name}.csv"

        if not use_cache or not os.path.exists(output_path):
            download_wb(
                wandb_names=[run_name],
                # x_axis="throughput/total_tokens", # <- akshita uses total_tokens to index wandb logs
                x_axis="_step",
                y_axis=[
                    "eval/validation-and-bpb-and-downstream"
                ],
                output_path=output_path,
                additional_keys=[
                    "throughput/total_tokens",
                    "throughput/total_training_Gflops",
                    "optim/learning_rate_group0", 
                    "train/CrossEntropyLoss",
                    # "learning_rate_peak",  # <- not in peteish
                    # "batch_size_in_tokens" # <- not in peteish
                ]
            )
        file_paths[output_name] = output_path

    return file_paths


def download_ian_in_loop(use_cache=True):
    ''' 
    Download Ian's 1B 1T runs 
    https://wandb.ai/ANON-llm/cheap_decisions/table
    '''
    WANDB_RESULTS = f"{DATA_DIR}/wandb/cheap_decisions"

    IAN_MODELS = [
        'dolma-v1-6-and-sources-baseline-1B-N-1T-D-mitchish1-001', 
        'dolma-v1-6-and-sources-baseline-docspara-dedup-1B-N-1T-D-mitchish1-001', 
        'dolma-v1-6-and-sources-baseline-3x-code-1B-N-1T-D-mitchish1-001', 
        'dolma-v1-6-and-sources-baseline-docspara-dedup-qc-01-1B-N-1T-D-mitchish1-001',
        'dolma-v1-7-1B-N-1T-D-mitchish1-006',
        'dolma-v1-7-1B-N-1T-D-mitchish1-006-2' # beware the "-2" 😨🫣🫢
    ]

    run_names = [
        f"ANON-llm/cheap_decisions/{model_name}" for model_name in IAN_MODELS
    ]
    
    IAN_EVAL_KEYS = [
        "eval/c4_en-validation/CrossEntropyLoss",
        "eval/c4_en-validation/Perplexity",
        "eval/dolma_books-validation/CrossEntropyLoss",
        "eval/dolma_books-validation/Perplexity",
        "eval/dolma_common-crawl-validation/CrossEntropyLoss",
        "eval/dolma_common-crawl-validation/Perplexity",
        "eval/dolma_pes2o-validation/CrossEntropyLoss",
        "eval/dolma_pes2o-validation/Perplexity",
        "eval/dolma_reddit-validation/CrossEntropyLoss",
        "eval/dolma_reddit-validation/Perplexity",
        "eval/dolma_stack-validation/CrossEntropyLoss",
        "eval/dolma_stack-validation/Perplexity",
        "eval/dolma_wiki-validation/CrossEntropyLoss",
        "eval/dolma_wiki-validation/Perplexity",
        "eval/downstream/arc_challenge_len_norm",
        "eval/downstream/arc_easy_acc",
        "eval/downstream/boolq_acc",
        "eval/downstream/commonsense_qa_len_norm",
        "eval/downstream/copa_acc",
        "eval/downstream/hellaswag_len_norm",
        "eval/downstream/mmlu_humanities_mc_5shot_len_norm",
        "eval/downstream/mmlu_humanities_mc_5shot_test_len_norm",
        "eval/downstream/mmlu_humanities_var_len_norm",
        "eval/downstream/mmlu_other_mc_5shot_len_norm",
        "eval/downstream/mmlu_other_mc_5shot_test_len_norm",
        "eval/downstream/mmlu_other_var_len_norm",
        "eval/downstream/mmlu_social_sciences_mc_5shot_len_norm",
        "eval/downstream/mmlu_social_sciences_mc_5shot_test_len_norm",
        "eval/downstream/mmlu_social_sciences_var_len_norm",
        "eval/downstream/mmlu_stem_mc_5shot_len_norm",
        "eval/downstream/mmlu_stem_mc_5shot_test_len_norm",
        "eval/downstream/mmlu_stem_var_len_norm",
        "eval/downstream/openbook_qa_len_norm",
        "eval/downstream/piqa_len_norm",
        "eval/downstream/sciq_acc",
        "eval/downstream/social_iqa_len_norm",
        "eval/downstream/winogrande_acc",
        "eval/ice-validation/CrossEntropyLoss",
        "eval/ice-validation/Perplexity",
        "eval/m2d2_s2orc-validation/CrossEntropyLoss",
        "eval/m2d2_s2orc-validation/Perplexity",
        "eval/pile-validation/CrossEntropyLoss",
        "eval/pile-validation/Perplexity",
        "eval/wikitext_103-validation/CrossEntropyLoss",
        "eval/wikitext_103-validation/Perplexity",
        "throughput/device/batches_per_second",
        "throughput/device/tokens_per_second",
    ]

    file_paths = {}
    for run_name in run_names:
        model_name = run_name.split("/")[-1]
        output_path = f"{WANDB_RESULTS}/{model_name}.csv"

        wb_runs: List[Run] = get_runs([run_name])
        for wb_run in wb_runs:
            available_keys = wb_run.summary.keys()
            print(f"Available keys for {wb_run.name}:\n{[k for k in available_keys if 'optim' not in k]}")

        if not use_cache or not os.path.exists(output_path):
            download_wb(
                wandb_names=[run_name],
                x_axis="_step",
                y_axis=IAN_EVAL_KEYS,
                additional_keys=[
                    "throughput/total_tokens",
                    "train/CrossEntropyLoss",
                    "train/Perplexity"
                ],
                output_path=output_path
            )
        file_paths[model_name] = output_path

    return file_paths


if __name__ == "__main__": 
    # download_ian_in_loop(use_cache=True)
    download_model_ladder(run_name='peteish', use_cache=True)