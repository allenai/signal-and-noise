from snr.constants.models import DATADECIDE_FINAL_FIVE_CKPTS, MODEL_LIST_DATADECIDE_FINAL, MODEL_LIST_EXTERNAL
from snr.scripts.oe_eval_tasks import CUSTOM_LOSS, LLM_COMPRESSION, PALOMA


def get_resource_util(model, task_list):
    batch_size = None
    gpu_memory_utilization = 0.7

    if model in [m['model'] for m in MODEL_LIST_EXTERNAL]:
        # From my testing, looks like anything less than 4 GPUs on 13B+ models (or Gemma 7B+) breaks
        # Also 70B model do not work on neptune (L40s)
        model_type = 'vllm'
        if 'smol' in model:
            gpus = 1
        elif 'stablelm' in model:
            model_type = 'hf'
        elif 'qwen-' in model or 'llama-2' in model or model == 'nemotron-3-8b-base-4k':
            # Qwen 1 models are broken in vLLM, we use hf instead
            model_type = 'hf'
            gpus = 4
        elif '110b' in model.lower() or '405b' in model.lower() or '8x22b' in model.lower() or ('gemma-3-' in model and '1b' not in model):
            gpus = 8
        elif model in ['gemma-7b', 'gemma2-9b', "gemma2-2b-instruct", "gemma2-9b-instruct", "gemma2-9b-instruct-SimPO", "llama2-13b", "llama3-70b", "llama3.1-70b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b", "llama3.1-70b-instruct", "qwen2.5-14b-instruct"] or '32B' in model or '72B' in model or '22B' in model or '15b' in model or '40b' in model or '70B' in model:
            gpus = 4
        else:
            gpus = 1 # don't need many GPUs for small models

        if 'gemma-3-' in model:
            gpu_memory_utilization = 0.3
    elif 'peteish32' in model or 'peteish13' in model or 'peteish7' in model:
        model_type = 'vllm'
        gpus = 4
    elif model in [m['model'] for m in MODEL_LIST_DATADECIDE_FINAL + DATADECIDE_FINAL_FIVE_CKPTS] or ('-3B-' in model):
        # Our 3B models have a head size of 208. This is not supported by PagedAttention and will throw errors.
        model_type = 'hf'
        gpus = 1

        # For the DataDecide models, manually set the batch size for single GPU A100/H100 eval
        CUSTOM_BZ = {
            '1B': 32,
            '750M': 32,
            '530M': 32,
            '300M': 32,
            '150M': 32,
            '90M': 32,
            '20M': 64,
            '4M': 64,
        }
        for key in CUSTOM_BZ:
            if key in model:
                batch_size = CUSTOM_BZ[key]
                if any('mc' in task for task in task_list):
                    batch_size = int(batch_size / 2)
                if any('gen' in task for task in task_list):
                    batch_size = int(batch_size / 4)
    else:
        model_type = 'vllm'
        gpus = 1

    if any(task in PALOMA + LLM_COMPRESSION + CUSTOM_LOSS for task in task_list):
        model_type = 'hf' # we can only run perplexity on hf for now
        if model in MODEL_LIST_EXTERNAL or '10xC' in model:
            batch_size = 1 # larger corpora will silent fail

    return model_type, gpus, batch_size, gpu_memory_utilization