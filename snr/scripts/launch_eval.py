import itertools
import os, time, sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from analysis.utils import weka_to_gcs

from analysis.utils.constants_models import MODEL_LADDER_LIST, MODEL_LIST_MIXES_FINAL, MODEL_LIST_MIXES_FINAL_EXTENDED, MODEL_LIST_INTERMEDIATE, MODEL_LIST_INTERMEDIATE_13B, MODEL_LIST_MIXES, OE_EVAL_BASE_MODELS, OE_EVAL_INSTRUCT_MODELS, OE_EVAL_ALL_MODELS, OE_EVAL_BASE_MODELS_EXTENDED, OE_EVAL_BASE_MODELS_EXTENDED_2, MODEL_LIST_INTERMEDIATE_7B, MODEL_LIST_FINAL_30_1B, MODEL_LIST_FINAL_30_13B, MODEL_LIST_INTERMEDIATE_32B, MODEL_LIST_SEED_RUNS
from analysis.utils.constants_model_ckpts import MODEL_LIST_FINAL_SIX_CKPTS, DATADECIDE_FINAL_FIVE_CKPTS, MODEL_MERGED_DATADECIDE, MODEL_MERGED_LADDER
from analysis.utils.constants_models import WEKA_CLUSTERS, GCP_CLUSTERS
from analysis.utils.constants_tasks import MC_TASKS_COPY_COLORS, MISSING_EVALS

# OLMES Core Tasks
from analysis.utils.constants_tasks import RC_TASKS_OLMES, MC_TASKS_OLMES, PARA_TASKS_OLMES, ENLARGE_TASKS_OLMES, DISTRACTORS_TASKS_OLMES

# OLMES Gen Tasks
from analysis.utils.constants_tasks import GEN_TASKS_OLMES, GEN_TASKS_OLMES_PERTURB_RC

# CoT tasks (mainly Tulu tasks)
from analysis.utils.constants_tasks import AGI_EVAL_MC, AGI_EVAL_RC, AGI_EVAL_COT
from analysis.utils.constants_tasks import MMLU_PRO_MC, MMLU_PRO_RC, MMLU_PRO_COT
from analysis.utils.constants_tasks import MINERVA_MC, MINERVA_COT
from analysis.utils.constants_tasks import BBH_MC, BBH_COT
from analysis.utils.constants_tasks import PERTURB_COT_TASKS

# Perplexity tasks
from analysis.utils.constants_tasks import PALOMA, LLM_COMPRESSION, CUSTOM_LOSS

MODEL_LIST_ALL = []
# MODEL_LIST_ALL += MODEL_LADDER_LIST
# MODEL_LIST_ALL += MODEL_LIST_INTERMEDIATE
# MODEL_LIST_ALL += OE_EVAL_BASE_MODELS
# MODEL_LIST_ALL += MODEL_LIST_INTERMEDIATE_13B # 13B intermediate ckpts
# MODEL_LIST_ALL += MODEL_LIST_MIXES_FINAL # ian's new mixes
# MODEL_LIST_ALL += MODEL_LIST_MIXES_FINAL_EXTENDED # extended set of data mixes
# MODEL_LIST_ALL += OE_EVAL_BASE_MODELS_EXTENDED # OLL 2 leaderboard models
# MODEL_LIST_ALL += OE_EVAL_BASE_MODELS_EXTENDED_2 # Additional external models
# MODEL_LIST_ALL += MODEL_LIST_INTERMEDIATE_7B # 7B Final 30 ckpts (1000 steps apart)
# MODEL_LIST_ALL += MODEL_LIST_INTERMEDIATE_32B # 32B Final 30 ckpts (1000 steps apart)

# MODEL_LIST_ALL += MODEL_LIST_FINAL_30_13B # 13B Final 30 ckpts (1000 steps apart)
# MODEL_LIST_ALL += MODEL_LIST_FINAL_30_1B # 1.5B-4T Final 30 ckpts (1000 steps apart)
# MODEL_LIST_ALL += [
#     'weka://oe-eval-default/ai2-llm/checkpoints/OLMo-medium/peteish7/last-5-model-merged',
#     'weka://oe-eval-default/ai2-llm/checkpoints/OLMo-medium/peteish7/last-30-model-merged',
#     'weka://oe-eval-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/last-5-model-merged',
#     'weka://oe-eval-default/ai2-llm/checkpoints/OLMo-medium/peteish13-highlr/last-30-model-merged',
#     'weka://oe-eval-default/ai2-llm/checkpoints/OLMo-large/peteish32/last-5-model-merged',
#     'weka://oe-eval-default/ai2-llm/checkpoints/OLMo-large/peteish32/last-29-model-merged',
# ] # merged models (weka only)

# MODEL_LIST_ALL += MODEL_LIST_FINAL_SIX_CKPTS # (200) Model ladder final 6 ckpts
# MODEL_LIST_ALL += MODEL_LIST_SEED_RUNS # (20) Seed runs (weka only)

# MODEL_LIST_ALL += DATADECIDE_FINAL_FIVE_CKPTS # (1125) DataDecide final 5 ckpts (only have rc_basic, rc_difficult, autobench, part of mc_basic)

MODEL_LIST_ALL += MODEL_MERGED_DATADECIDE # (225) Merged DataDecide   -- only have rc basic, rc_difficult
MODEL_LIST_ALL += MODEL_MERGED_LADDER # (27) Merged ladder (gcs only) -- only have rc basic, rc_difficult

TASK_LIST_ALL = []

# TASK_LIST_ALL += RC_TASKS_OLMES
# TASK_LIST_ALL += PARA_TASKS_OLMES 
# TASK_LIST_ALL += ENLARGE_TASKS_OLMES
# TASK_LIST_ALL += DISTRACTORS_TASKS_OLMES
# TASK_LIST_ALL += MC_TASKS_OLMES

# TASK_LIST_ALL += MC_TASKS_COPY_COLORS
# TASK_LIST_ALL += GEN_TASKS_OLMES
# TASK_LIST_ALL += AGI_EVAL_MC + MMLU_PRO_MC # + MINERVA_MC
# TASK_LIST_ALL += AGI_EVAL_COT # + MMLU_PRO_COT
# TASK_LIST_ALL += BBH_MC # BPB verison of BBH
# TASK_LIST_ALL += BBH_COT

# TASK_LIST_ALL += MMLU_PRO_RC + AGI_EVAL_RC
# TASK_LIST_ALL += GEN_TASKS_OLMES_PERTURB_RC
# TASK_LIST_ALL += PERTURB_COT_TASKS

# TASK_LIST_ALL += ['autobencher::none', 'autobencher:mc::none']

# TASK_LIST_ALL += [
#     # GSM CoT
#     "gsm8k::olmes:full",
#     # Minerva CoT (olmes version)
#     "minerva_math_algebra::olmes:full",
#     "minerva_math_counting_and_probability::olmes:full",
#     "minerva_math_geometry::olmes:full",
#     "minerva_math_intermediate_algebra::olmes:full",
#     "minerva_math_number_theory::olmes:full",
#     "minerva_math_prealgebra::olmes:full",
#     "minerva_math_precalculus::olmes:full",
#     # Coding
#     "mbpp::ladder",
#     "mbppplus::ladder",
#     "codex_humaneval:temp0.8",
#     "codex_humanevalplus::ladder", 
# ]

# TASK_LIST_ALL += [
#     'deepmind_math_large::none',
#     'medmcqa:rc::none',
#     'medmcqa:mc::none',
#     'gsm_plus::none',
#     'gsm_symbolic::none',
#     'gsm_symbolic_p1::none',
#     'gsm_symbolic_p2::none',
#     # 'gpqa::none', # requires HF token
#     'minerva_math_500::none', 
# ]

# TASK_LIST_ALL += [
#     'aime::none',
# ]

# TASK_LIST_ALL += PALOMA
# TASK_LIST_ALL += LLM_COMPRESSION
# TASK_LIST_ALL += CUSTOM_LOSS

TASK_LIST_ALL += [ # Custom suites to prevent gRPC overload on Beaker
    'rc_basic::custom_suite',
    # 'mc_basic::custom_suite',
    'rc_difficult::custom_suite',
    # 'autobench::custom_suite',
    # 'gen::custom_suite',
    # 'gen_difficult::custom_suite',
]

# # FOR TESTING
# TASK_LIST_ALL = ["arc_challenge:rc::olmes:full"]
# MODEL_LIST_ALL = [
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-1B-5xC-2/step69369-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-750M-5xC-2/step63589-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-530M-5xC-2/step57776-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-300M-5xC-2/step45787-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/benb/prox_fineweb_pro-150M-5xC-2/step38157-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/prox_fineweb_pro-90M-5xC/step29901-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/prox_fineweb_pro-60M-5xC/step29042-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/prox_fineweb_pro-20M-5xC/step14584-unsharded-hf",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-ladder/davidh/prox_fineweb_pro-4M-5xC/step5735-unsharded-hf",
# ]
# MODEL_LIST_ALL = [
#     # "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-large/peteish32/step720000",
#     # "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-large/peteish32/step705000",
#     # "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-large/peteish32/step701000",
#     # "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-large/peteish32/step696000",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-medium/peteish7/step919000",
#     "weka://oe-eval-default/ai2-llm/checkpoints/OLMo-medium/peteish7/step917000"
# ]
# MODEL_LIST_ALL = [
#     "mistral-small-3.1-24b-base-2503",
#     "gemma-2-2b",
# ]
# MODEL_LIST_ALL = [model for model in MODEL_LIST_ALL if '/DCLM-baseline-' not in model] # DCLM only
# MODEL_LIST_ALL = MODEL_MERGED_DATADECIDE[:2] + MODEL_MERGED_LADDER[:2]


def run_eval(model_list, task_list, model_type='hf', gpus=1, gpu_memory_utilization=0.7, limit=None, batch_size=None, save_requests=True):
    if isinstance(task_list, list): 
        task_list = ' '.join([f'"{task}"' for task in task_list])
    if not isinstance(model_list, list): 
        model_list = [model_list]

    # # Use WEKA
    # cluster_list = WEKA_CLUSTERS
    # model_list = [model.replace('weka://', '/weka-mount/') for model in model_list] # beaker
    # # model_list = [model.replace('weka://', '/') for model in model_list] # local

    # Use GCP
    cluster_list = GCP_CLUSTERS
    if any('weka://' in model for model in model_list):
        model_list = [weka_to_gcs(model) for model in model_list]

    if len(model_list) == 1: # convert back list -> str
        model_list = model_list[0]

    WORKSPACE = "ai2/ladder-evals"
    PRIORITY = "normal"

    # WORKSPACE = "ai2/lm-eval"
    # PRIORITY = "high" # high

    VLLM_MEMORY_USE = f"--model-args gpu_memory_utilization={gpu_memory_utilization}" if model_type == 'vllm' else " "

    command = f"""
    oe-eval \
        --model {model_list} \
        --task {task_list} \
        --cluster {cluster_list} \
        --model-type {model_type} \
        --gpus {gpus} \
        --beaker-workspace {WORKSPACE} \
        --beaker-image davidh/oe-eval-metaeval \
        --gantry-secret-aws-access-key-id AWS_ACCESS_KEY_ID \
        --gantry-secret-aws-secret-access AWS_SECRET_ACCESS_KEY \
        --gantry-secret-hf-read-only davidh_HF_TOKEN \
        --remote-output-dir s3://ai2-llm/eval-results/downstream/metaeval/ \
        --recompute-metrics \
        --gantry-args '{{"env": "VLLM_USE_V1=0", "HF_HUB_TIMEOUT": "60"}}' \
        {VLLM_MEMORY_USE} \
        --beaker-priority {PRIORITY}
    """
    # --gantry-secret-hf-read-only lucas_HUGGING_FACE_HUB_TOKEN \
    # --run-local \

    # oe-eval --model pythia-14m --task mmlu_high_school_european_history:rc::olmes:full --gpus 1 --model-type vllm --run-local

    command = command.replace('\n', '').replace('  ', '') # remove extra spacing!
    if limit is not None: 
        print(f'🫢😧 Using a {limit} instance limit 🤫🫣')
        command += f" --limit {limit}"
    if batch_size is not None: 
        print(f'Using a batch_size of {batch_size}')
        command += f" --batch-size {batch_size}"
    if not save_requests:
        command += ' --save-raw-requests false --delete-raw-requests'
    mb = len(command.encode('utf-8')) / (1024 * 1024) # compute size of command

    print(f'Executing command:\n{command}')
    print(f'Estimated size of all argument strings: {(mb * len(MODEL_LIST_ALL)):.2f} MB (4 MB will crash beaker)')
    
    os.system(command)


def main():
    print(f'Launching {len(MODEL_LIST_ALL)} models on {len(TASK_LIST_ALL)} tasks (5 second sleep to confirm...)')
    # time.sleep(5)

    # for (model, missing_tasks) in MISSING_EVALS:
    #     task_list = []
    #     for missing_task in missing_tasks:
    #         task_list += [task_name for task_name in TASK_LIST_ALL if missing_task in task_name]

    #     if len(task_list) == 0:
    #         # print(f'Cant find tasks for {missing_task}')
    #         continue
    #         # raise

    # task_list = TASK_LIST_ALL
    # for model in MODEL_LIST_ALL:

    for task, model in itertools.product(TASK_LIST_ALL, MODEL_LIST_ALL):
        task_list = [task]

        batch_size = None
        save_requests = True
        gpu_memory_utilization = 0.7

        # batch_size = 4 # TMP OVERRIDE FOR LADDER MODELS

        if model in OE_EVAL_ALL_MODELS:
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
            elif model in ['gemma-7b', 'gemma2-9b', "gemma2-2b-instruct", "gemma2-9b-instruct", "gemma2-9b-instruct-SimPO", "llama2-13b", "llama3-70b", "llama3.1-70b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b", "llama3.1-70b-instruct", "qwen2.5-14b-instruct"] or '32B' in model or '72B' in model or '22B' in model or '15b' in model or '40b' in model or '70B' in model or model in OE_EVAL_BASE_MODELS_EXTENDED_2:
                gpus = 4
            else:
                gpus = 1 # don't need many GPUs for small models

            if 'gemma-3-' in model:
                gpu_memory_utilization = 0.3
        elif 'peteish32' in model or 'peteish13' in model or 'peteish7' in model:
            model_type = 'vllm'
            gpus = 4
        elif model in \
            MODEL_LIST_MIXES + MODEL_LIST_MIXES_FINAL + MODEL_LIST_MIXES_FINAL_EXTENDED + DATADECIDE_FINAL_FIVE_CKPTS + MODEL_MERGED_DATADECIDE or \
            ('-3B-' in model) or \
            model in [weka_to_gcs(m) for m in MODEL_LIST_MIXES + MODEL_LIST_MIXES_FINAL + MODEL_LIST_MIXES_FINAL_EXTENDED + DATADECIDE_FINAL_FIVE_CKPTS + MODEL_MERGED_DATADECIDE]:
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
            # model_type = 'hf'
            model_type = 'vllm'
            gpus = 1

        if any(task in PALOMA + LLM_COMPRESSION + CUSTOM_LOSS for task in task_list):
            save_requests = False # don't save the perplexity files
            model_type = 'hf' # we can only run perplexity on hf for now
            if model in OE_EVAL_BASE_MODELS or '10xC' in model:
                batch_size = 1 # larger corpora will silent fail

        run_eval(
            model_list=model, 
            task_list=task_list, 
            model_type=model_type, 
            gpus=gpus,
            # limit=10_000,
            batch_size=batch_size,
            save_requests=save_requests,
            gpu_memory_utilization=gpu_memory_utilization
        )


if __name__ == '__main__': main()
