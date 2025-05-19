import json, os, re, sys, yaml
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import psutil
from tqdm import tqdm
import boto3

# Add parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

parent_dir = Path(__file__).resolve().parent
sys.path.append(str(parent_dir))

from utils import DATA_DIR, weka_to_gcs, fix_model_path
from utils.constants_tasks import RC_TASKS_OLMES, MC_TASKS_OLMES, GEN_TASKS_OLMES, MINERVA_COT

# Metrics to use when converting to table:
METRICS_TO_KEEP = [
    "acc_raw",
    "acc_per_char",
    "predicted_index_per_char",
    "predicted_index_raw",
    "correct_choice",
    "logits_per_char_corr",
    "logits_per_byte_corr",
    "bits_per_byte_corr"

    # Generative metrics
    "exact_match",
    "f1",
    "recall",
    "pass_at_1",
    "pass_at_10",

    # Perplexity metrics (e.g., Paloma)
    "bits_per_byte"
]

MODEL_OUTPUT_TO_KEEP = [
    "sum_logits",
    "logits_per_char",
    "logits_per_byte",
    "bits_per_byte"
]

SIZE_PREFIXES = [
    f'-{size}-' for size in ['3B', '1B', '760M', '750M', '530M', '370M', '300M', '190M', '150M', '90M', '60M', '20M', '16M', '14M', '10M', '8M', '6M', '4M']
]
SIZE_PREFIXES_FIX = {'3B': '3.2B', '1B': '1.3B'}

CHINHILLA_MULT = [
    '0.5xC', '1xC', '2xC', '5xC', '10xC', '15xC', '20xC'
]


def is_excluded_from_lite(m):
    BROKEN_MODELS = [
        # "gemma-2b", 
        # "gemma-7b", 
        # "gemma-2-2b", 
        # "gemma-2-9b",
        # 'gemma-3-1b-pt',
        # 'gemma-3-4b-pt',
        # 'gemma-3-12b-pt',
        # 'Llama-3.1-70B',
        # 'Llama-3.1-8B',
        # 'OLMo-2-0325-32B'
    ]

    OLL2_INSTRUCT_MODELS = [
        # These are models on the OLL2 leaderboard that are actually instruct models
        'instruct',
        'superthoughts',
        'helpingai',
        'fox',
        'llmchat',
        'intern',
        'magistrate', # legal annealing
        'fietje', # phi fine-tune
        'llama-3-6.3b', # pruned llama 3
        'loxa', # very suspicious
        'llumix', # hungarian instruction tune
        'yarm', # instruction tune for context
        'lucie', # looks suspicious, i really think they snuck in instruct data here
        'nepali',
        'windy',
        'yarn', # long context fine-tuned models
        'llama-160m', # these models are just really bad
        'llama-43m',
        'llama-68m',

        # missing evals
        'salamandra',
        'aya',
        'gpt'
    ]

    # These models have broken or incomplete results
    if 'Minitron' in m or \
        'Mistral' in m or \
        'bloom' in m or \
        'granite' in m or \
        'pruned' in m or \
        'INTELLECT-1' in m or \
        'TinyYi-7B-Test' in m or \
        'Qwarkstar-4B' in m or \
        'Priya-10B' in m or \
        'SmolLM2-360M' in m or \
        'InstructLM-500M' in m or \
        'RedPajama-INCITE-Base-3B-v1' in m or \
        'pythia-410m' in m or \
        'Qwen1.5-MoE' in m or \
        'Codestral-22B-v0.1' in m or \
        'RYS-Medium' in m or \
        'falcon-mamba-7b' in m or \
        'Qwen1.5-0.5B' in m or \
        'Falcon3-7B-Base' in m or \
        'OLMo-2-0325-32B' in m or \
        'gemma-2-27b' in m or \
        'gemma-3-12b-pt' in m or \
        'gemma-3-1b-pt' in m or \
        'gemma-3-4b-pt' in m or \
        'stablelm-base-alpha-7b' in m:
        return True
    
    if any(name.lower() in m.lower() for name in OLL2_INSTRUCT_MODELS) or \
        any(name.lower() in m.lower() for name in BROKEN_MODELS):
        return True
    
    return False


def str_find(str_list, input_string):
    """ Get if a list of strings exists in a string. Return first match """
    hits = [item for item in str_list if item in input_string]
    if len(hits) == 0: 
        return None
    else:
        return hits[0]
    

def get_mix(model_name):
    """ falcon_and_cc_eli5_oh_top10p-3B-5xC => falcon_and_cc_eli5_oh_top10p """
    mix = None
    for prefix in SIZE_PREFIXES:
        if prefix in model_name:
            mix = model_name.split(prefix)[0]
            
            # manual overrides for model ladder
            mix = mix.replace('-rerun', '')
            mix = mix.replace('-moreeval', '-ladder')
    return mix


def extract_step(input_string):
    if input_string is None: return None
    match = re.search(r'step(\d+)(?:-[a-zA-Z0-9]+)?', input_string)
    return int(match.group(1)) if match else None


def remove_prefix(input_string):
    return re.sub(r'^task-\d+-', '', input_string)


def nested_defaultdict():
    return defaultdict(nested_defaultdict)


def fsize(file_path):
    return os.path.getsize(file_path) / (1024 ** 3)


def process_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def get_native_id(pred):
    native_id = str(pred['native_id']) if pred['native_id'] is not None else ''
    doc_id = str(pred['doc_id']) if pred['doc_id'] is not None else ''
    return f'{native_id}:{doc_id}'


def compute_mean_safe(predictions, key):
    values = [
        pred[key] for pred in predictions
        if key in pred and pred[key] is not None
    ]
    return np.array(values).mean().item() if values else None
            

def process_predictions(file_path):
    """ Process a predictions.jsonl to a list """
    predictions = process_jsonl(file_path)

    request_ids_to_bytes = None
    if 'consistent_ranking' in str(file_path):
        # Load the requests file to compute BPB
        requests_folder = '/oe-eval-default/davidh/metaeval/analysis/data/consistent_ranking/eval-results/downstream/eval-for-consistent-ranking/baseline-150M-5xC-2/step38157-unsharded-hf'
        file_name = Path(file_path).name.replace('predictions.jsonl', 'requests.jsonl')
        requests_path = f"{requests_folder}/{Path(file_path).parent.name}/{file_name}"
        requests = process_jsonl(requests_path)
        request_ids   = [get_native_id(request) for request in requests]
        request_bytes = [max(len(request["request"].get("continuation", "").encode("utf-8")), 1) for request in requests]
        request_ids_to_bytes = defaultdict(list)
        for request_id, request_byte in zip(request_ids, request_bytes):
            request_ids_to_bytes[request_id].append(request_byte)

    processed = []
    for pred in predictions:
        entry = {}

        task_name = file_path.split('/')[-1].replace('-predictions.jsonl', '')

        native_id = get_native_id(pred)
        entry['native_id'] = native_id
        entry['instance_id'] = str(native_id) + ':::' + str(task_name) # should be get_metadata_from_file_name()?
        metrics = pred['metrics']
        model_output = pred['model_output']

        metrics_to_keep = METRICS_TO_KEEP
        model_output_to_keep = MODEL_OUTPUT_TO_KEEP
        
        for col in metrics_to_keep:
            entry[col] = metrics[col] if col in metrics else None
        for col in model_output_to_keep:
            entry[col] = [output[col] if col in output else None for output in model_output]

        # For some generation benchmarks, correct_choice is a str, but this will cause a type error
        # when indexing this column
        if isinstance(entry['correct_choice'], str):
            entry['correct_choice'] = 0

        # Sometimes exact_match is bool when it should be float
        if 'exact_match' in entry and isinstance(entry['exact_match'], bool):
            entry['exact_match'] = float(entry['exact_match'])

        # If primary_score does not exist, add it
        from constants_olmes import PRIMARY_METRICS_OLMES
        primary_metric_key = PRIMARY_METRICS_OLMES.get(task_name, None)
        if primary_metric_key is None: 
            primary_metric_key = 'acc_per_char'
        if ('primary_score' not in entry and primary_metric_key in metrics) or ('primary_score' in entry and primary_metric_key['primary_score'] is None):
            entry['primary_score'] = metrics[primary_metric_key]
        # assert entry.get('primary_score', None) is not None, (task_name, primary_metric_key, entry, metrics)

        # Compute BPB using request files
        if request_ids_to_bytes is not None:
            all_num_bytes = request_ids_to_bytes[str(entry["native_id"])]
            if len(all_num_bytes) > len(model_output):
                # For whatever reason ian's results have zero- and few-shot...
                # print(f'Seeing len(entry_requests)={len(entry_requests)} and len(model_output)={len(model_output)}. Truncating...')
                all_num_bytes = all_num_bytes[:len(model_output)]
            # assert len(entry_requests) == len(model_output), (entry_requests, entry["native_id"], requests[0])
            assert len(all_num_bytes) == len(model_output), (len(all_num_bytes), len(model_output))
            all_logits_per_byte = []
            for num_bytes, out in zip(all_num_bytes, model_output):
                LOG_2_OF_E = 1.44269504089
                logits_per_byte = -LOG_2_OF_E * (out["sum_logits"] / num_bytes)
                out['num_bytes'] = num_bytes
                out['logits_per_byte'] = logits_per_byte
                all_logits_per_byte.append(logits_per_byte)
            entry["logits_per_byte"] = all_logits_per_byte
            if 0 <= entry["correct_choice"] < len(all_logits_per_byte):
                entry["logits_per_byte_corr"] = all_logits_per_byte[entry["correct_choice"]]
            else:
                print(f'Incorrect correct_choice indexer: {entry["correct_choice"]}, {file_path}')
                entry["logits_per_byte_corr"] = 0

        # Use both names
        if 'bits_per_byte_corr' in entry and entry['bits_per_byte_corr'] is not None:
            entry['logits_per_byte_corr'] = entry['bits_per_byte_corr']

        if 'logits_per_byte_corr' in entry and entry['logits_per_byte_corr'] is not None:
            entry['bits_per_byte_corr'] = entry['logits_per_byte_corr']
 
        processed += [entry]
    return processed


def process_metrics(file_path):
    """ Process a metrics.json to a dict """
    with open(file_path, 'r') as f:
        results = json.load(f)

    if 'beaker_info' in results:    del results['beaker_info']
    if 'compute_config' in results: del results['compute_config']
    if 'task_config' in results:    del results['task_config']

    # Only keep these metrics for Paloma
    PALOMA_METRICS = [
        'bits_per_byte',
        'ppl_token',
        'ppl_char',
        'ppl_word',
        'ppl_byte',
    ]

    if 'metrics' in results:
        for metric in results['metrics']:
            if ('paloma' in file_path or 'llm_compression' in file_path or 'custom_loss' in file_path) and metric not in PALOMA_METRICS:
                continue
            results[metric] = results['metrics'][metric]

    # Get token spend if it exists (num_instances is already a col)
    if 'extra_metrics' in results and 'num_tokens' in results["extra_metrics"]:
        results["num_tokens"] = results['extra_metrics']["num_tokens"]

    # Rename bpb to logits_per_byte_corr if it exists
    if 'bits_per_byte' in results and results['bits_per_byte'] is not None:
        results['logits_per_byte_corr'] = results['bits_per_byte']

    if 'logits_per_byte_corr' not in results:
        # Get bits-per-byte from prediction files if they dont exist
        predictions_path = file_path.replace('metrics.json', 'predictions.jsonl')
        if os.path.exists(predictions_path):
            predictions = process_predictions(predictions_path)

            for prediction in predictions:
                if 'correct_choice' in prediction and prediction['correct_choice'] is not None:
                    try:
                        correct_choice = prediction['correct_choice']

                        if ('logits_per_byte_corr' not in prediction or prediction['logits_per_byte_corr'] is None) and 'logits_per_byte' in prediction:
                            logits_per_byte = prediction['logits_per_byte']

                            if 0 <= correct_choice < len(logits_per_byte):
                                prediction['logits_per_byte_corr'] = logits_per_byte[correct_choice]
                            else:
                                # print(f'Incorrect correct_choice indexer: {correct_choice}, {file_path}')
                                prediction['logits_per_byte_corr'] = 0

                        if ('logits_per_char_corr' not in prediction or prediction['logits_per_char_corr'] is None) and 'logits_per_char' in prediction:
                            logits_per_char = prediction['logits_per_char']

                            if 0 <= correct_choice < len(logits_per_char):
                                prediction['logits_per_char_corr'] = logits_per_char[correct_choice]
                            else:
                                # print(f'Incorrect correct_choice indexer: {correct_choice}, {file_path}')
                                prediction['logits_per_char_corr'] = 0
                    except Exception as e:
                        print(e)
                        raise RuntimeError(prediction, results)

            logits_per_byte = compute_mean_safe(predictions, 'logits_per_byte_corr')
            logits_per_char = compute_mean_safe(predictions, 'logits_per_char_corr')

            if 'logits_per_byte_corr' not in results: 
                results['logits_per_byte_corr'] = logits_per_byte
            if 'logits_per_char_corr' not in results: 
                results['logits_per_char_corr'] = logits_per_char

    return results


def process_chunk(chunk):
    return pd.DataFrame(chunk)


def get_available_cpus(threshold=50):
    cpu_usages = psutil.cpu_percent(percpu=True)
    available_cpus = [i for i, usage in enumerate(cpu_usages) if usage < threshold]
    return available_cpus


def load_df_parallel(data, file_type, usage_threshold=50):
    """ Load data as df w/ a CPU pool. Only use CPUs with usage below usage_threshold """
    available_cpus = get_available_cpus(threshold=usage_threshold)

    if file_type == 'metrics' or file_type == 'questions':
        num_partitions = max(1, len(data) // 1_000)
    elif 'predictions' in file_type:
        # Currently trying both 10_000 w/ 50% threshold and 300_000 with 50% threshold
        num_partitions = max(1, len(data) // 300_000) # default is 10_000, on errors I set to 100_00, 50K chunks led to a broken pipe
    # num_partitions = len(available_cpus) * 100

    print(f'Distributing {num_partitions} chunks across {len(available_cpus)} CPUs')
    
    if num_partitions == 0:
        raise RuntimeError("No CPUs are available below the usage threshold.")
    
    # Use numpy for efficient chunking
    num_partitions = max(1, min(len(data), num_partitions))  # Prevent more partitions than data
    chunk_size = len(data) // num_partitions
    remainder = len(data) % num_partitions
    chunks = [
        data[i * chunk_size + min(i, remainder) : (i + 1) * chunk_size + min(i + 1, remainder)]
        for i in range(num_partitions)
    ]
    chunk_len = set(len(chunk) for chunk in chunks)

    print(f'Launching parallel processing for chunk lengths {chunk_len}...')
    
    with Pool(processes=len(available_cpus)) as pool:
        dataframes = list(tqdm(pool.imap(process_chunk, chunks), desc='Converting to Pandas dataframe', total=len(chunks)))

    # with Pool(processes=num_partitions) as pool:
    #     dataframes = list(tqdm(pool.map(process_chunk, chunks), desc='Converting to Pandas dataframe', total=len(chunks)))
    
    return pd.concat(dataframes, ignore_index=True)


def get_metadata_from_file_name(root, file):
    # Manual override for Ian's folder setup
    root = root.replace('/all_olmes_paraphrase_tasks', '')
    root = root.replace('/all_olmes_rc_tasks', '')

    path_parts = os.path.normpath(root).split(os.sep)

    # Use last two folders as "path"/"step":
    # E.g., ../peteish-moreeval-1B-0.5xC/step8145-unsharded-hf
    if len(path_parts) >= 2:
        if 'step' in path_parts[-1]: # and ('-unsharded' in path_parts[-1] or '-hf' in path_parts[-1])
            # Local OLMo runs (anything that ends in "stepXXX-unsharded")
            model_name = path_parts[-2]
            step_str = path_parts[-1]
        else:
            # External models (e.g., llama)
            model_name = path_parts[-1]
            step_str = None
    else:
        raise RuntimeError(f'Could not process path: {path_parts}')

    # Get task name: "arc_challenge-metrics.json" => "arc_challenge"
    task = remove_prefix(file) # Remove "task-XXX" prefix: task-XXX-task_name.json => task_name.json
    task = task.rsplit('-', 1)[0]

    # Get step name: "stepXXX-unsharded" => "XXX"
    step = extract_step(step_str)

    # Get mix name
    mix_name = get_mix(model_name)

    # Get other metadata
    size = str_find(SIZE_PREFIXES, model_name)
    if size is not None: size = size.replace('-', '')
    token_ratio = str_find(CHINHILLA_MULT, model_name)

    return model_name, mix_name, step, step_str, size, token_ratio, task


def load_file(file_data, _type, load_lite_tasks=None):
    root, file = file_data
    file_path = os.path.join(root, file)

    model_name, mix_name, step, step_str, size, token_ratio, task = get_metadata_from_file_name(root, file)

    # fix for the names of one of Ian's data mixes
    if mix_name == 'baseline': mix_name = 'dolma17'

    if load_lite_tasks is not None:
        if is_excluded_from_lite(model_name):
            # Exclude some broken or instruct external models from instanceslite/instancesmedium
            return []

        if load_lite_tasks == 'lite_predictions':
            LITE_TASKS = RC_TASKS_OLMES
        elif load_lite_tasks == 'medium_predictions':
            LITE_TASKS = RC_TASKS_OLMES + MC_TASKS_OLMES + GEN_TASKS_OLMES + MINERVA_COT + ['gsm8k', 'mbpp', 'mbppplus', 'codex_humaneval', 'codex_humanevalplus', 'autobencher', 'autobencher:mc'] + ["paloma_c4_en", "paloma_m2d2_s2orc_unsplit"]
        lite_task_ids = [task_id.split('::')[0].replace(':rc', '') for task_id in LITE_TASKS]
        if task not in lite_task_ids:
            # For the small instance dataset, only include a small set of tasks
            return []

    if 'predictions' in _type:
        # Load predictions
        if 'predictions.jsonl' not in file_path: 
            return []
        results = process_predictions(file_path)
    elif _type == 'metrics':
        if 'metrics.json' not in file_path:
            return []
        if 'verbose-metrics.json' in file_path:
            return []
        metrics = process_metrics(file_path)

        # Sometimes the metrics file causes OOM errors, so we will delete if it's too big
        if 'metrics' in metrics and len(str(metrics['metrics'])) > 1000:
            metrics['metrics'] = None
        
        results = [metrics]
    elif _type == 'questions':
        if 'predictions.jsonl' not in file_path or 'peteish-moreeval-rerun-1B-1xC' not in file_path:
            return []
        # Load the requests file to compute BPB
        requests_folder = '/oe-eval-default/davidh/metaeval/analysis/data/aws/eval-results/downstream/metaeval/OLMo-ladder/peteish-moreeval-rerun-1B-1xC/step16279-unsharded-hf'
        file_name = Path(file_path).name.replace('predictions.jsonl', 'requests.jsonl') # /{Path(file_path).parent.name}
        requests_path = f"{requests_folder}/{file_name}"

        file_name = Path(file_path).name.replace('predictions.jsonl', 'metrics.json')
        metrics_path = f"{requests_folder}/{file_name}"

        if not os.path.exists(requests_path):
            print(f'Could not find questions path for {task}: {requests_path}')
            return []

        # Get the task alias
        requests = process_jsonl(requests_path)
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        task_alias = metrics['task_config']['metadata']['alias']

        # If OLMES, get the small requests file
        olmes_requests_ids = None
        if '::olmes' in task_alias:
            olmes_requests_folder = '/oe-eval-default/davidh/metaeval/analysis/download/olmes_requests/EleutherAI/pythia-160m'
            file_name = Path(file_path).name
            olmes_small_requests_path = f"{olmes_requests_folder}/{file_name}"
            if os.path.exists(olmes_small_requests_path):
                olmes_requests = process_jsonl(olmes_small_requests_path)
                olmes_requests_ids = []
                for request in olmes_requests:
                    native_id_small = str(request['native_id'])
                    olmes_requests_ids += [native_id_small]
                # print(f'Found request IDs for {task}:')
                # print(olmes_requests_ids)

        for i, request in enumerate(requests):
            native_id = get_native_id(request)
            native_id_small = str(request['native_id'])
            instance_id = str(native_id) + ':::' + str(task)
            context = request['request'].get('context', '')
            continuation = request['request'].get('continuation', '')

            # TODO: Get all instances in the corresponding small OLMES metrics file and check equivalence
            in_olmes_small = False
            if olmes_requests_ids is not None and native_id_small in olmes_requests_ids:
                in_olmes_small = True

            request = {'instance_id': instance_id, 'native_id': native_id, 'task_alias': task_alias, 'in_olmes_small': in_olmes_small, 'context': context, 'continuation': continuation, **request}
            request = {k: str(v) for k, v in request.items()}
            requests[i] = request
        
        results = requests
    else:
        raise ValueError(_type)

    if 'data/reddit/evaluation' in str(root):
        # Load the original metrics.json (useful for adding to instance-level predictions)
        metrics_path = file_path.replace('predictions.jsonl', 'metrics.json')
        with open(metrics_path, 'r') as f:
            orig_metrics = json.load(f)

        # Get the path of the model config
        model_path = orig_metrics['model_config']['model_path']
        model_path = Path(model_path.replace("weka:/", "")) / "config.yaml"

        # NOTE: FOR S3 data, we don't have it in Weka, so need to download
        if 's3:' in str(model_path):
            model_path = str(model_path).replace('s3:/ai2-llm/', '')
            model_path = model_path.replace('-hf', '') # usually the base model dir has the config.yaml
            local_path = f'/root/ai2/metaeval/analysis/data/.configs/{model_path}'
            if not os.path.exists(local_path) and 'metrics.json' in file_path:
                # use AWS to download to .reddit/[path]/config.yaml, if it doesn't exist
                # E.g., s3:/ai2-llm/checkpoints/reddit-2g-ablations/redDC-5050-llama1-mix-Cx5-20241008/step61989-hf/config.yaml
                s3_client = boto3.client('s3')
                print(model_path + ' -> ' + local_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3_client.download_file('ai2-llm', model_path, local_path)
            model_path = Path(local_path)

        if not model_path.exists():
            # continue
            raise RuntimeError(model_path)

        with open(model_path, 'r', encoding='utf-8') as file:
            yaml_content = yaml.safe_load(file)
        
        data_paths = yaml_content.get('data', {}).get('paths', [])

        # Load data paths from config.yaml and add to results
        for result in results:
            result.update({
                'data_paths': data_paths
            })

    # Add metadata to parquet file
    for result in results:
        result.update({
            'model': model_name,
            'mix': mix_name,
            'step': step,
            'size': size,
            'token_ratio': token_ratio,
            'step_str': step_str,
            'task': task,
            's3_path': file_path,
        })

    return results


def process_files_chunk(files_chunk, _type, load_lite_tasks=None):
    results = []
    for file in files_chunk:
        results.extend(load_file(file, _type, load_lite_tasks))
    return results


def filter_model_seeds(all_files):
    # Filter out all but one seed run from Ian's mixes
    print(f'Original # files to load: {len(all_files)}')
    
    # Extract relevant information and organize paths by prefix
    def extract_data_mix_and_step(paths):
        data_dict = defaultdict(list)
        pattern = re.compile(r'([^/]+)/step(\d+)-unsharded-hf')

        for path in tqdm(paths, desc='extract_data_mix_and_step'):
            root, file = path
            full_path = os.path.join(root, file)
            folder_parts = full_path.split('/')
            data_mix_match = re.search(pattern, '/'.join(folder_parts))

            if data_mix_match:
                data_mix = data_mix_match.group(1)
                step = int(data_mix_match.group(2))
                prefix = "-".join(data_mix.split('-')[:-1])

                data_dict[prefix].append((step, full_path))

        return data_dict

    # Find the largest step for each prefix
    def filter_largest_step(data_dict):
        result = {}
        for prefix, steps_and_paths in tqdm(data_dict.items(), desc='filter_largest_step'):
            largest_step_path = max(steps_and_paths, key=lambda x: x[0])
            result[prefix] = largest_step_path

        return result

    # Create the filtered file list based on largest step
    def get_filtered_file_list(filtered_results):
        file_list = [path for _, (_, path) in filtered_results.items()]
        return file_list

    data_dict = extract_data_mix_and_step(all_files)
    filtered_results = filter_largest_step(data_dict)
    all_files = get_filtered_file_list(filtered_results)

    # Get the folders of each mix with the highest checkpoint
    all_files = [
        os.path.dirname(os.path.dirname(os.path.dirname(filepath))) + os.path.sep
        for filepath in all_files
    ]

    # for prefix, (step, path) in filtered_results.items():
    #     print(f"Prefix: {prefix}, Largest Step: {step}, Path: {path}")

    # Re-scan directories
    all_files = scan_dir(all_files)

    print(f'New # files to load: {len(all_files)}')

    return all_files


def scan_dir(data_input):
    all_files = []

    # Ensure the input is a list of paths, even if it's a single path
    paths = [data_input] if isinstance(data_input, (str, os.PathLike)) else data_input
    if not isinstance(paths, (list, tuple)):
        raise ValueError("Input must be a directory path or a list of paths.")

    with tqdm(desc="Scanning paths", total=len(paths), unit="path") as pbar:
        for path in paths:
            if os.path.isfile(path):
                if path.endswith('-predictions.jsonl') or path.endswith('-metrics.json'):
                    all_files.append((os.path.dirname(path), os.path.basename(path)))
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    if 'local_testing' not in root:
                        all_files.extend(
                            (root, file)
                            for file in files
                            if file.endswith('-predictions.jsonl') or file.endswith('-metrics.json')
                        )
            pbar.update(1)
    
    return all_files


def recursive_pull(data_dir, file_type, load_lite_tasks=None):
    all_files = scan_dir(data_dir)

    # all_files = all_files[:10_000] # for testing
    # all_files = all_files[:1_000_000] # for testing

    # chunk_size = 100 # for testing
    chunk_size = 700 # for testing
    # chunk_size = 1_000
    # chunk_size = 10_000 # for testing

    all_preprocessed = []
    file_chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
    chunk_len = set(len(chunk) for chunk in file_chunks)
    total_files = len(all_files)

    with tqdm(total=len(file_chunks), desc=f"Submitting file chunks of lengths {chunk_len}") as submit_pbar:
        with tqdm(total=total_files, desc=f"Recursively loading files in {data_dir.name}") as pbar:
            with ProcessPoolExecutor(max_workers=len(get_available_cpus())) as executor:
                futures = {}
                for chunk in file_chunks:
                    future = executor.submit(process_files_chunk, chunk, file_type, load_lite_tasks)
                    futures[future] = len(chunk)
                    submit_pbar.update(1)  # Update submission progress
                for future in as_completed(futures):
                    all_preprocessed.extend(future.result())
                    pbar.update(futures[future])  # Update based on the chunk size
            pbar.close()
        submit_pbar.close()

    return all_preprocessed


def cleanup_metrics_df(df):
    """ A safe function to clean up benchmark results """
    # Preprocess the df into a usuable format
    df = df.drop(columns=["Unnamed: 0"], errors='ignore')

    # Modify column order to move these up
    desired_order = ['task', 'model', 'step', 'mix', 'size', 'token_ratio', 'primary_score', 'logits_per_byte_corr']
    existing_columns = [col for col in desired_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_columns]
    df = df[existing_columns + remaining_cols]

    # Add primary score if it does not exist
    if 'primary_score' in df.columns:
        if 'acc_per_char' in df.columns:
            df['primary_score'] = df['primary_score'].fillna(df['acc_per_char'])
        if 'exact_match' in df.columns:
            df['primary_score'] = df['primary_score'].fillna(df['exact_match'])
        if 'pass_at_1' in df.columns:
            df['primary_score'] = df['primary_score'].fillna(df['pass_at_1'])

    return df


def verify_df(df):
    # Identify missing models/tasks
    unique_models = df['model'].unique()
    unique_tasks  = df['task'].unique()
    missing_entries = []
    for model in unique_models:
        for task in unique_tasks:
            task_rows = df[(df['model'] == model) & (df['task'] == task)]
            if task_rows.empty:
                missing_entries.append((model, task))

    if missing_entries:
        print("Missing tasks for models:")
        for model, task in missing_entries:
            print(f"  - Model: {model}, Task: {task}")


def sanity_check_file(path):
    root, file = path
    model_name, mix_name, step, step_str, size, token_ratio, task = get_metadata_from_file_name(root, file)
    # # synthetic evals (excluded from Ian's model for now)
    # if ':cot' in task:
    #     return None, None
    
    # # if ':para' in task:
    # #     return None, None
    # # if ':distractors' in task:
    # #     return None, None
    # # if ':enlarge' in task:
    # #     return None, None

    if ':perturb_cot' in task:
        return None, None
    if ':perturb_rc' in task:
        return None, None
    # if 'bbh_' in task:
    #     return None, None

    # if 'coqa' in task:
    #     return None, None
    # if 'copycolors:mc' in task:
    #     return None, None
    # if 'aime' in task:
    #     # Getting OOM errors for AIME
    #     return None, None

    # # # only math/code
    # # if not ('gsm8k' in task or 'mbpp' in task or 'codex' in task or 'minerva' in task):
    # #     return None, None

    # # perplexity evals
    # if '-verbose' in task:
    #     return None, None
    if task in ["paloma_twitterAAE_HELM_fixed", "paloma_c4_100_domains", "paloma_dolma_100_subreddits"]:
        # these tasks are half-evaluated and shouldn't be in there anyways
        return None, None
    # # if 'paloma' in task or 'llm_compression' in task or 'custom_loss' in task:
    # #     return None, None

    # Get model path
    file_name = str(Path(root)) + '/' + Path(file).name.replace('predictions.jsonl', 'metrics.json')
    with open(file_name, 'r') as f:
        metrics = json.load(f)
    if 'model_config' not in metrics: # not valid metrics file?
        return None, None
    if 'metadata' in metrics['model_config']:
        model_path = metrics.get('model_config', {}).get("metadata", {}).get("alias", {})
    else:
        model_path = metrics['model_config']['model_path']
    model_path = model_path.replace('/weka-mount/', 'weka://')
    model_path = weka_to_gcs(model_path) # just use the gcs path for all models
    model_path = fix_model_path(model_path)

    # Get task alias
    # task_alias = task
    task_alias = None
    if 'metadata' in metrics['task_config'] and 'alias' in metrics['task_config']['metadata']:
        task_alias = metrics.get('task_config', {}).get('metadata', {}).get('alias', '')

    if task_alias is None:
        print(f'Found no task alias for task: {task}')
        return None, None
    
    return model_path, task_alias


def sanity_check(folder_name):
    """ 
    All leaf folders should have the same eval data. This prints folders
    that do not have data compared to evals that appear at least once.
    """
    data_dir = Path(DATA_DIR).resolve()
    data_dir.mkdir(exist_ok=True)

    aws_dir = data_dir / folder_name

    all_files = scan_dir(aws_dir)

    model_tasks = defaultdict(set)
    with ProcessPoolExecutor() as executor:
        futures = []
        for file in tqdm(all_files, desc='Parallize load jobs'):
            futures.append(executor.submit(sanity_check_file, file))
            
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc='Loading result files'):
            results.append(future.result())
    
    for model_path, task_alias in results:
        if model_path and task_alias:
            # model_path = model_path.lower() # some model names are capitalized
            model_tasks[model_path].add(task_alias)
    
    all_tasks = set(task for tasks in model_tasks.values() for task in tasks)

    # def include_task(task):
    #     if not isinstance(task, str):
    #         return False
    #     # # Require perturbed evaluation
    #     # return (':para' in task or ':distractors' in task or ':enlarge' in task)
    #     # # Require paloma
    #     # return ('paloma' in task)
    # all_tasks = set([task for task in list(all_tasks) if include_task(task)])

    incomplete_models = {model for model, tasks in model_tasks.items() if tasks != all_tasks}
    missing_entries = {model: sorted(list(all_tasks - model_tasks[model])) for model in sorted(list(incomplete_models))}
    
    data_dir = Path(DATA_DIR).resolve()
    with open(data_dir / f'{folder_name}_missing_tasks.json', 'w') as f:
        json.dump(missing_entries, f, indent=2)
    
    # Output missing tasks
    for model, tasks in missing_entries.items():
        print(f"('{model}', {tasks}),")


def main(folder_name, file_type='predictions'):
    data_dir = Path(DATA_DIR).resolve()
    data_dir.mkdir(exist_ok=True)

    aws_dir                = data_dir / folder_name
    prediction_path        = data_dir / f"{folder_name}_predictions.parquet"
    lite_prediction_path   = data_dir / f"{folder_name}_lite_predictions.parquet"
    medium_prediction_path = data_dir / f"{folder_name}_medium_predictions.parquet"
    questions_path         = data_dir / f"{folder_name}_questions.parquet"
    metrics_path           = data_dir / f"{folder_name}_metrics.parquet"

    # Change settings for prediction subset files
    parquet_path = prediction_path
    if file_type == 'lite_predictions':
        parquet_path = lite_prediction_path
        load_lite_tasks = file_type
    elif file_type == 'medium_predictions':
        parquet_path = medium_prediction_path
        load_lite_tasks = file_type
    else:
        load_lite_tasks = None
    
    predictions_df = recursive_pull(aws_dir, file_type, load_lite_tasks=load_lite_tasks)

    # Save predictions to parquet
    import time
    start_time = time.time()
    
    df = load_df_parallel(predictions_df, file_type) # for 6700 preds: 300s (5 min)

    print(f"Converted to pandas in: {time.time() - start_time:.4f} seconds")
    # verify_df(df)

    if file_type == 'metrics':
        df = cleanup_metrics_df(df)

        print(df.columns)

        df.to_parquet(metrics_path)
        print('Done!')
        return
    elif file_type == 'questions':
        df.to_parquet(questions_path)
        print('Done!')
        return

    # Reset the df index (for faster indexing)
    df.set_index(['task', 'model', 'step', 'mix'], inplace=True)

    # Save to parquet
    df.to_parquet(parquet_path, index=True)
    print(f"Predictions saved to {parquet_path} ({fsize(parquet_path):.2f} GB)")

    print('Done!')


if __name__ == '__main__': 
    folder_name = "aws"

    sanity_check(folder_name)

    # main(folder_name, file_type='metrics')
    # main(folder_name, file_type='predictions')
    # main(folder_name, file_type='medium_predictions')
    # main(folder_name, file_type='lite_predictions')
    # main(folder_name, file_type='questions')
    