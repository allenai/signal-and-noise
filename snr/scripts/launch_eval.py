import itertools
import os

from snr.scripts.resource_util import get_resource_util

from snr.constants.models import MODEL_LADDER_LIST, MODEL_LIST_DATADECIDE_FINAL, MODEL_LIST_INTERMEDIATE_1B, MODEL_LIST_INTERMEDIATE_13B, MODEL_LIST_EXTERNAL, MODEL_LIST_FINAL_30_7B, MODEL_LIST_FINAL_30_1B, MODEL_LIST_FINAL_30_13B, MODEL_LIST_FINAL_30_32B, MODEL_LIST_SEED_RUNS, MODEL_LIST_FINAL_SIX_CKPTS, DATADECIDE_FINAL_FIVE_CKPTS
from snr.scripts.oe_eval_tasks import MC_TASKS_COPY_COLORS
from snr.scripts.oe_eval_tasks import RC_TASKS_OLMES, MC_TASKS_OLMES
from snr.scripts.oe_eval_tasks import GEN_TASKS_OLMES
from snr.scripts.oe_eval_tasks import AGI_EVAL_MC, AGI_EVAL_RC
from snr.scripts.oe_eval_tasks import MMLU_PRO_MC, MMLU_PRO_RC
from snr.scripts.oe_eval_tasks import BBH_COT
from snr.scripts.oe_eval_tasks import PALOMA, LLM_COMPRESSION, CUSTOM_LOSS
from snr.scripts.oe_eval_tasks import AUTOBENCHER, MATH_CODE, EXTA_TASKS

MODEL_LIST_ALL = []
MODEL_LIST_ALL += MODEL_LADDER_LIST
MODEL_LIST_ALL += MODEL_LIST_EXTERNAL
MODEL_LIST_ALL += MODEL_LIST_INTERMEDIATE_1B # 1B intermediate ckpts
MODEL_LIST_ALL += MODEL_LIST_INTERMEDIATE_13B # 13B intermediate ckpts
MODEL_LIST_ALL += MODEL_LIST_DATADECIDE_FINAL # DataDecide models
MODEL_LIST_ALL += MODEL_LIST_FINAL_30_7B # 7B Final 30 ckpts (1000 steps apart)
MODEL_LIST_ALL += MODEL_LIST_FINAL_30_32B # 32B Final 30 ckpts (1000 steps apart)
MODEL_LIST_ALL += MODEL_LIST_FINAL_30_13B # 13B Final 30 ckpts (1000 steps apart)
MODEL_LIST_ALL += MODEL_LIST_FINAL_30_1B # 1.5B-4T Final 30 ckpts (1000 steps apart)
MODEL_LIST_ALL += MODEL_LIST_FINAL_SIX_CKPTS # (200) Model ladder final 6 ckpts
MODEL_LIST_ALL += MODEL_LIST_SEED_RUNS # (20) Seed runs (weka only)
MODEL_LIST_ALL += DATADECIDE_FINAL_FIVE_CKPTS # (1125) DataDecide final 5 ckpts

TASK_LIST_ALL = []
TASK_LIST_ALL += RC_TASKS_OLMES
TASK_LIST_ALL += MC_TASKS_OLMES
TASK_LIST_ALL += MC_TASKS_COPY_COLORS
TASK_LIST_ALL += GEN_TASKS_OLMES
TASK_LIST_ALL += AGI_EVAL_MC + MMLU_PRO_MC
TASK_LIST_ALL += BBH_COT
TASK_LIST_ALL += MMLU_PRO_RC + AGI_EVAL_RC
TASK_LIST_ALL += AUTOBENCHER
TASK_LIST_ALL += MATH_CODE
TASK_LIST_ALL += EXTA_TASKS
TASK_LIST_ALL += PALOMA
TASK_LIST_ALL += LLM_COMPRESSION
TASK_LIST_ALL += CUSTOM_LOSS


def run_eval(model_list, task_list, model_type='hf', gpus=1, gpu_memory_utilization=0.7, batch_size=None):
    if isinstance(task_list, list): 
        task_list = ' '.join([f'"{task}"' for task in task_list])
    
    if not isinstance(model_list, list): 
        model_list = [model_list]

    if len(model_list) == 1: # convert back list -> str
        model_list = model_list[0]

    VLLM_MEMORY_USE = f"--model-args gpu_memory_utilization={gpu_memory_utilization}" if model_type == 'vllm' else " "

    command = f"""
    oe-eval \
        --model {model_list} \
        --task {task_list} \
        --model-type {model_type} \
        --gpus {gpus} \
        --recompute-metrics \
        --gantry-args '{{"env": "VLLM_USE_V1=0", "HF_HUB_TIMEOUT": "60"}}' \
        {VLLM_MEMORY_USE} \
    """

    # --cluster {cluster_list} \
    # --run-local \
    # --beaker-workspace {WORKSPACE} \
    # --beaker-priority {PRIORITY}
    # --remote-output-dir s3://ai2-llm/eval-results/downstream/metaeval/ \
    # --beaker-image davidh/oe-eval-metaeval \
    # --gantry-secret-aws-access-key-id AWS_ACCESS_KEY_ID \
    # --gantry-secret-aws-secret-access AWS_SECRET_ACCESS_KEY \
    # --gantry-secret-hf-read-only davidh_HF_TOKEN \


    command = command.replace('\n', '').replace('  ', '')
    if batch_size is not None: 
        command += f" --batch-size {batch_size}"

    print(f'Executing command:\n{command}')
    
    os.system(command)


def main():
    print(f'Launching {len(MODEL_LIST_ALL)} models on {len(TASK_LIST_ALL)} tasks')

    for task, model in itertools.product(TASK_LIST_ALL, MODEL_LIST_ALL):
        task_list = [task]

        model_type, gpus, batch_size, gpu_memory_utilization = get_resource_util(model, task_list)

        run_eval(
            model_list=model, 
            task_list=task_list, 
            model_type=model_type, 
            gpus=gpus,
            batch_size=batch_size,
            gpu_memory_utilization=gpu_memory_utilization
        )


if __name__ == '__main__': main()
