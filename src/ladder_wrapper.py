from concurrent.futures import ProcessPoolExecutor
import os, copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from dataloader import get_nd_array
from download.preprocess import SIZE_PREFIXES, SIZE_PREFIXES_FIX, str_find

from utils.constants_ladder import DATA_BY_NAME_LADDER

from scaling.utils import FinalConfig
from scaling.utils import get_final_configs, get_step2_data_by_name
from scaling.utils import get_final_configs, get_step1_data_by_name

from fitting.step1 import fit_step1, predict_step1, plot_step1, str_chinchilla_n_d_fit
from fitting.step2 import fit_step2, predict_step2, plot_step2
from fitting.predict import predict_chained, plot_chained, str_chained_fit

from fitting.step1_flops import fit_step1 as fit_step1_flops, predict_step1 as predict_step1_flops, plot_step1 as plot_step1_flops, str_chinchilla_flops_fit
from fitting.predict_flops import predict_chained_flops, plot_chained as plot_chained_flops, str_chained_fit as str_chained_fit_flops
from fitting.single_step import fit_single_step, predict_single_step, plot_single_step, str_combined_fit

FONTSIZE = 8

TASK_KEY_MAP = {
    "arc_challenge": "arc_challenge_test_5shot",
    "arc_easy": "arc_easy_test_5shot",
    "boolq": "boolq_val_5shot",
    "socialiqa": "socialiqa_val_5shot",
    "csqa": "csqa_val_5shot",
    "hellaswag": "hellaswag_val_5shot",
    "openbookqa": "openbookqa_test_5shot",
    "winogrande": "winogrande_val_5shot",
    "piqa": "piqa_val_5shot",
}

COLOR_MAP = {
    'Qwen': 'green',
    'Llama': 'blue',
    'LLaMA': 'blue',
    'Mistral': 'orange',
    '3B': 'black',
    'OLMo': 'pink',
    'pythia': 'brown',
    'gemma': 'teal',
    'phi': 'black',
    'deepseek': 'pink',
    'zephyr': 'green',
    'neo': 'orange',
    'falcon': 'blue',

    # code models 
    'starcoder': 'grey',
    'stablelm': 'grey',
}


def sort_experiment_names(experiments):
    """ Sort a list of ladder model names by size, then token multiplier """
    def extract_sort_key(entry):
        parts = entry.split('-')
        size_str, xc_str = parts[-2], parts[-1]
        
        if "M" in size_str:
            size = int(size_str.replace("M", "")) * 1e6
        elif "B" in size_str:
            size = int(size_str.replace("B", "")) * 1e9
        else:
            raise ValueError(size_str)
        
        xc = float(xc_str.replace("xC", ""))
        return (size, xc)

    return sorted(experiments, key=extract_sort_key)


def merge_dicts(dict1, dict2, overwrite_xs=False, overwrite_ds_ns_ls=True):
    """ Merge the data_by_name of dict2 into dict1 """
    if dict1.keys() != dict2.keys():
        raise ValueError(f"Keys of dict1 and dict2 do not match. Seeing:\n{dict1.keys()}\n{dict2.keys()}")
    
    for key in dict1:
        l1, l2 = len(dict1[key]['xs']), len(dict2[key]['xs'])

        # Sort all values by the number of tokens seen
        if 'ds' in dict2[key]:
            indices = sorted(range(len(dict2[key]['ds'])), key=lambda i: dict2[key]['ds'][i])
        else:
            indices = range(len(dict2[key]['xs']))
        
        if overwrite_ds_ns_ls:
            dict2[key]['ds'] = [dict2[key]['ds'][i] for i in indices]
            dict2[key]['ns'] = [dict2[key]['ns'][i] for i in indices]
            dict2[key]['ls'] = [dict2[key]['ls'][i] for i in indices]
            if 'fs' in dict2: dict2[key]['fs'] = [dict2[key]['fs'][i] for i in indices]
        if overwrite_xs:  
            dict2[key]['xs'] = [dict2[key]['xs'][i] for i in indices]

        if l1 != l2:
            # A faustian bargain to allow us to use the wandb tokens w/ oe-eval for intermediate ckpts, since we have different numbers
            # of checkpoints for both. 
            if l1 < l2:
                # Shorter is dict1[key]['xs'], sample dict2[key]['xs']
                indices = np.linspace(0, l2 - 1, l1, dtype=int)
                if overwrite_ds_ns_ls:
                    dict1[key]['ns'] = [dict2[key]['ns'][i] for i in indices]
                    dict1[key]['ds'] = [dict2[key]['ds'][i] for i in indices]
                    dict1[key]['ls'] = [dict2[key]['ls'][i] for i in indices]
                    dict1[key]['fs'] = [None for _ in indices]
                if overwrite_xs:  
                    dict1[key]['xs'] = [dict2[key]['xs'][i] for i in indices]
            else:
                raise RuntimeError(f'different sized lists: {l1}, {l2}')
        else:
            if overwrite_ds_ns_ls:
                dict1[key]['ns'] = dict2[key]['ns']
                dict1[key]['ds'] = dict2[key]['ds']
                dict1[key]['ls'] = dict2[key]['ls']
                dict1[key]['fs'] = [None for _ in indices]
            if overwrite_xs:  
                dict1[key]['xs'] = dict2[key]['xs']

    return dict1


def map_corr_labels(bpb, corr, task_name):
    """ Given a tensor of tensors and a tensor of indices, map the indices to their entries """
    if bpb.ndim not in {1, 2, 3}:
        raise ValueError(bpb)

    n_choices = bpb[0].shape[-1]
    correct_bpb = np.empty_like(corr, dtype=np.float64)

    if bpb.ndim == 1:  # Handle 1D case
        for i in range(corr.shape[0]):
            if corr[i] == n_choices and 'enlarge' in task_name:
                corr[i] -= 1
            correct_bpb[i] = bpb[i][int(corr[i])]

    elif bpb.ndim == 2:  # Handle 2D case
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                if corr[i, j] == n_choices and 'enlarge' in task_name:
                    corr[i, j] -= 1
                correct_bpb[i, j] = bpb[i, j][corr[i, j].astype(np.int32)]

    else:  # bpb.ndim == 3, Handle 3D case
        for k in range(corr.shape[0]):
            for i in range(corr.shape[1]):
                for j in range(corr.shape[2]):
                    if corr[k, i, j] == n_choices and 'enlarge' in task_name:
                        corr[k, i, j] -= 1
                    correct_bpb[k, i, j] = bpb[k, i, j][corr[k, i, j].astype(np.int32)]

    return correct_bpb


def get_ladder_size(model_name):
    size = str_find(SIZE_PREFIXES, model_name)
    if model_name == 'peteish7':
        size = '7B-4T'
    elif 'peteish13-highlr' in model_name:
        size = '13B-5T'
    elif size is not None:
        size = size.replace('-', '')
        size = SIZE_PREFIXES_FIX.get(size, size)
    else:
        size = model_name
    return size


def get_ladder_data(
        df, task_name, train_models, eval_models, step='max', 
        intermediate_feature="logits_per_byte_corr", downstream_feature="primary_score"
    ):
    """ Get slices of df and convert to ladder prediction format """
    data_by_name = defaultdict(dict)

    for model in train_models + eval_models:
        mode = 'train' if model in train_models else 'eval'
        
        # Fix string names for ladder models to match ladder
        size = get_ladder_size(model)

        is_multiindex = isinstance(df.index, pd.MultiIndex)
        
        if 'paloma' in task_name:
            metric_names = ["bits_per_byte"]
        elif 'exact_match' in df.columns:
            metric_names = ["correct_choice", "logits_per_byte", "acc_per_char", "exact_match", "pass_at_1", "pass_at_10"]
        elif 'logits_per_byte_corr' in df.columns:
            metric_names = ["logits_per_byte_corr", "acc_per_char"]
        else:
            metric_names = ["correct_choice", "logits_per_byte", "acc_per_char"]

        if not is_multiindex:
            # metric_names = ["logits_per_byte_corr", "primary_score"]
            metric_names = [intermediate_feature, downstream_feature]

        if intermediate_feature is not None and downstream_feature is not None:
            metric_names = [intermediate_feature, downstream_feature]
        
        if step == 'max':
            # Efficient querying for only final model step
            _, scores = get_nd_array(df, "model", metric_names, model=model, task=task_name, step="max")
            if len(scores) == 0:
                raise RuntimeError(f'No scores found for model={model}, metric={metric_names}, task={task_name}. Seeing: {scores}')
            scores_dict = {metric: scores[i] if i < len(scores) else np.array([]) for i, metric in enumerate(metric_names)}
        else:
            # Allow querying all steps
            if step == 'all': 
                step = None
            
            scores_dict = {metric: np.array([]) for metric in metric_names}
            for metric in metric_names:
                columns, scores = get_nd_array(
                    df, ['model', 'step', 'mix', 'task'], metric, 
                    model=model, task=task_name, step=step
                )

                # Macro-average: Group by all columns except the last one (the subtask) and average the scores
                grouped_cols = {}
                for col in columns:
                    key = tuple(col[:-1])  # All elements except last
                    if key not in grouped_cols:
                        grouped_cols[key] = []
                    grouped_cols[key].append(col)

                # sort by step
                grouped_cols = dict(sorted(grouped_cols.items(), key=lambda x: float(x[0][0])))
                
                averaged_columns = []
                averaged_scores = []
                for key, cols in grouped_cols.items():
                    first_col = tuple(list(cols[0])[:-1]) # get rid of task col. e.g., (3400.0, 'peteish-ladder')
                    avg_scores = np.mean([scores[columns.index(col)] for col in cols], axis=0).item()
                    averaged_columns.append(first_col)
                    averaged_scores.append(avg_scores)
                columns = averaged_columns
                scores = np.array(averaged_scores)

                # Remove NaN scores and corresponding columns
                non_nan_mask = ~np.isnan(scores)
                scores = scores[non_nan_mask]
                columns = [col for i, col in enumerate(columns) if non_nan_mask[i]]

                scores_dict[metric] = np.array(scores)[:, np.newaxis]

        
        # Default to empty np array if it does not exist
        corr        = scores_dict.get("correct_choice", np.array([]))
        bpb         = scores_dict.get("logits_per_byte_corr", scores_dict.get("bits_per_byte", scores_dict.get("logits_per_byte", np.array([]))))
        acc         = scores_dict.get("acc_per_char", scores_dict.get("primary_score", np.array([])))
        exact_match = scores_dict.get("exact_match", np.array([]))
        pass_at_1   = scores_dict.get("pass_at_1", np.array([]))
        pass_at_10  = scores_dict.get("pass_at_10", np.array([]))

        if exact_match.size and not np.isnan(np.array(exact_match, dtype=float)).all():
            acc = exact_match
        if pass_at_1.size and not np.isnan(np.array(pass_at_1, dtype=float)).all():
            acc = pass_at_1
        if pass_at_10.size and not np.isnan(np.array(pass_at_10, dtype=float)).all():
            acc = pass_at_10

        if 'paloma' in task_name:
            acc = bpb
        if 'bbh' in task_name:
            bpb = acc
            corr = np.array([])

        if intermediate_feature is not None:
            bpb = scores_dict.get(intermediate_feature, np.array([]))

        if downstream_feature is not None:
            acc = scores_dict.get(downstream_feature, np.array([]))

        # Ensure `correct_choice` is non-NaN
        corr = np.nan_to_num(corr, nan=0)

        if bpb.size == 0:
            print(f'bpb is empty array: {bpb} on model "{model}" for task "{task_name}"')
            continue

        if len(bpb) == 0 or len(acc) == 0: 
            if mode == 'eval':
                print(f'Eval point data not found: {model}')
                continue

        if len(corr) != 0:
            correct_bpb = map_corr_labels(bpb, corr, task_name=task_name)
        else:
            correct_bpb = bpb

        acc = acc.mean(axis=-1)
        correct_bpb = correct_bpb.mean(axis=-1)

        if 'xs' not in data_by_name[size]: data_by_name[size]['xs'] = []
        if 'ys' not in data_by_name[size]: data_by_name[size]['ys'] = []

        # Convert output to float / list of floats
        if correct_bpb.size == 1:
            correct_bpb = correct_bpb.item() 
        else:
            correct_bpb = correct_bpb.squeeze().tolist()
        
        if acc.size == 1:
            acc = acc.item()
        else:
            acc = acc.squeeze().tolist()
        
        data_by_name[size]['xs'] += [correct_bpb]
        data_by_name[size]['ys'] += [acc]
        data_by_name[size]['mode'] = mode

    return data_by_name


def create_ladder_config(config_path, task_name, train_models, eval_models):
    # arc_easy:enlarge => arc_easy
    task_root = task_name.split(':')[0] if isinstance(task_name, str) else None

    # Get ladder model configs
    configs = get_final_configs(config_path)

    # Add new models to config
    for model in train_models + eval_models:
        if model not in configs.keys():
            # Get model color
            color = 'red'
            for key, value in COLOR_MAP.items():
                if key in model:
                    color = value
                    break

            mode = 'eval' if model in eval_models else 'train'

            model_label = model
            if '-3B-' in model_label:
                model_label = '3B'
                color = 'b'
            
            # Create dummy config for new eval points
            configs[model] = FinalConfig(
                paths=None, mode=mode, n=0, label=model_label, color=color, use_last_n_percentage=100
            )

    task_key = TASK_KEY_MAP.get(task_root, None) # the task key is used to get min/max perf and plot title

    return task_key, configs


def aggregate_list(data_by_name, last_n_method_train, last_n_method_eval, last_n):
    """ For xs or ys that have multiple points (e.g., last n points) aggregate them """
    for model_size, model_results in data_by_name.items():
        mode = model_results['mode']

        for datapoint_name, scores_2d in model_results.items(): # xs, ys
            for idx, scores in enumerate(scores_2d):
                if isinstance(scores, list): # if the value is a list of lists
                    if mode == 'train':
                        # For train models:
                        if last_n_method_train == 'avg':
                            score = np.mean(scores[-last_n:])
                        elif last_n_method_train == 'sample':
                            score = random.sample(scores[-last_n:], k=1)[0]
                        elif last_n_method_train == 'final':
                            score = scores[-1]
                        else:
                            raise ValueError(last_n_method_train)
                    elif mode == 'eval':
                        # For eval models:
                        if last_n_method_eval == 'avg':
                            score = np.mean(scores[-last_n:])
                        elif last_n_method_eval == 'sample':
                            score = random.sample(scores[-last_n:], k=1)[0]
                        elif last_n_method_eval == 'final':
                            score = scores[-1]
                        elif last_n_method_eval == 'all':
                            score = scores[-last_n]
                        else:
                            raise ValueError(last_n_method_eval)
                    
                    data_by_name[model_size][datapoint_name][idx] = score
    return data_by_name


def run_ladder(
    df, task_name, train_models, eval_models, config_path, 
    downstream_feature='primary_score', intermediate_feature='bpb', intermediate_task_name=None, y_metric='rc_bpb',  # "y_metric" is the metric type
    use_flops=False, use_single_step=False,
    last_n=None, last_n_method_train=None, last_n_method_eval=None, last_n_resample=None, # sample/avg last n checkpoints
    run_step1=True, run_step2=True, run_stacked=True,
    axes=None, add_texts=False, plot_compute=False,
    return_preds=False, return_reals=False, return_fit_error=False):
    if isinstance(axes, list) and axes[0] is None: axes = None
    
    data_by_name_tokens = DATA_BY_NAME_LADDER

    # Get config
    configs = get_final_configs(config_path)

    include_last_n = (last_n_method_train is not None) or (last_n_method_eval is not None)

    data_by_name_step_1 = None
    if run_step1 or run_stacked:
        # Load data
        data_by_name = get_ladder_data(
            df, task_name, train_models, eval_models, 
            downstream_feature=downstream_feature, 
            step=('all' if include_last_n else 'max'),
        )
        if intermediate_feature != 'bpb':
            assert intermediate_task_name is not None
            data_by_name_intermedaite = get_ladder_data(
                df, intermediate_task_name, train_models, eval_models, 
                intermediate_feature=intermediate_feature, downstream_feature=downstream_feature, 
                step=('all' if include_last_n else 'max'),
            )
            data_by_name = merge_dicts(data_by_name, data_by_name_intermedaite, overwrite_xs=True, overwrite_ds_ns_ls=False)
        
        # Add token data -- this removes models without token data (like Llama for step 2 fitting)
        # data_by_name_tokens = get_step1_data_by_name(configs, 'arc_easy_test_5shot', y_metric=y_metric, moving_avg=1) # we are only using this for the num tokens
        data_by_name = {k: v for k, v in data_by_name.items() if k in data_by_name_tokens.keys()}
        data_by_name_tokens = {k: v for k, v in data_by_name_tokens.items() if k in data_by_name.keys()}
        
        # I shouldn't have to do this. Something is broken when I use external models
        if '1.3B' in data_by_name_tokens: del data_by_name_tokens['1.3B']
        if '1.3B' in data_by_name: del data_by_name['1.3B']
        
        data_by_name_step_1 = merge_dicts(data_by_name, data_by_name_tokens, overwrite_xs=(y_metric == 'c4')) # merge the 'ns', 'ds', 'ls', 'fs' keys into the step 2 data

    task_key = None
    data_by_name_step_2 = None
    if run_step2 or run_stacked:
        # Reload data: this breaks stuff (lets us fit external models for step 2)
        data_by_name_step_2 = get_ladder_data(
            df, task_name, train_models, eval_models, 
            downstream_feature=downstream_feature, 
            step=('all' if include_last_n else 'max'),
        )
        if intermediate_feature != 'bpb':
            data_by_name_intermedaite = get_ladder_data(
                df, intermediate_task_name, train_models, eval_models, 
                intermediate_feature=intermediate_feature, downstream_feature=downstream_feature, 
                step=('all' if include_last_n else 'max'),
            )
            data_by_name_step_2 = merge_dicts(data_by_name_step_2, data_by_name_intermedaite, overwrite_xs=True, overwrite_ds_ns_ls=False)

        task_key, configs = create_ladder_config(config_path, task_name, train_models, eval_models)

    data_by_name_stacked = copy.deepcopy(data_by_name_step_1) # used to fit two-step prediction

    def _fit_ladder(_data_by_name_step_1, _data_by_name_step_2, _data_by_name_stacked):
        return fit_ladder(
            task_name, eval_models,
            configs,
            _data_by_name_step_1,  _data_by_name_step_2, _data_by_name_stacked, 
            task_key,
            y_metric=y_metric, use_flops=use_flops, use_single_step=use_single_step,
            run_step1=run_step1, run_step2=run_step2, run_stacked=run_stacked,
            axes=axes, add_texts=add_texts, plot_compute=plot_compute,
        )

    if last_n_method_train == 'sample' or last_n_method_eval == 'sample':
        # Resample and fit multiple model ladders
        rel_errors_step_1, rel_errors_step_2, rel_errors_stacked = [], [], []
        step_1_ys, step_2_ys, stacked_ys = [], [], []
        step_1_y_preds, step_2_y_preds, stacked_y_preds = [], [], []
        fit_errors = []

        # First generate all resampled data
        resampled_data = []
        for _ in range(last_n_resample):
            _data_by_name_step_1 = copy.deepcopy(data_by_name_step_1)
            _data_by_name_step_2 = copy.deepcopy(data_by_name_step_2)
            _data_by_name_stacked = copy.deepcopy(data_by_name_stacked)

            _data_by_name_step_1 = aggregate_list(_data_by_name_step_1, last_n_method_train, last_n_method_eval, last_n)
            _data_by_name_step_2 = aggregate_list(_data_by_name_step_2, last_n_method_train, last_n_method_eval, last_n)
            _data_by_name_stacked = aggregate_list(_data_by_name_stacked, last_n_method_train, last_n_method_eval, last_n)
            
            resampled_data.append((_data_by_name_step_1, _data_by_name_step_2, _data_by_name_stacked))

        # Run ladder fits in parallel
        with ProcessPoolExecutor() as executor:
            futures = []
            for data_tuple in resampled_data:
                futures.append(executor.submit(fit_ladder, 
                    task_name, eval_models, configs, data_tuple[0], data_tuple[1], data_tuple[2], task_key,
                    y_metric, use_flops, use_single_step, run_step1, run_step2, run_stacked, axes, add_texts, plot_compute,
                ))
            
            results = list(tqdm(
                (f.result() for f in futures), 
                total=len(futures),
                desc='Computing ladder fits'
            ))

        # Unpack results
        for result in results:
            (rel_error_step_1, rel_error_step_2, _rel_errors_stacked), \
            (step_1_y, step_2_y, stacked_y), \
            (step_1_y_pred, step_2_y_pred, stacked_y_pred), \
            fit_error = result

            rel_errors_step_1.append(rel_error_step_1)
            rel_errors_step_2.append(rel_error_step_2)
            rel_errors_stacked.append(_rel_errors_stacked) 
            step_1_ys.append(step_1_y)
            step_2_ys.append(step_2_y)
            stacked_ys.append(stacked_y)
            step_1_y_preds.append(step_1_y_pred)
            step_2_y_preds.append(step_2_y_pred)
            stacked_y_preds.append(stacked_y_pred)
            fit_errors.append(fit_error)
        rel_error_step_1, rel_error_step_2, rel_errors_stacked = (rel_errors_step_1, rel_errors_step_2, rel_errors_stacked)
        step_1_y, step_2_y, stacked_y = (step_1_ys, step_2_ys, stacked_ys)
        step_1_y_pred, step_2_y_pred, stacked_y_pred = (step_1_y_preds, step_2_y_preds, stacked_y_preds)
        fit_error = fit_errors

    elif last_n_method_train == 'avg' or last_n_method_eval == 'avg':
        data_by_name_step_1 = aggregate_list(data_by_name_step_1, last_n_method_train, last_n_method_eval, last_n)
        data_by_name_step_2 = aggregate_list(data_by_name_step_2, last_n_method_train, last_n_method_eval, last_n)
        data_by_name_stacked = aggregate_list(data_by_name_stacked, last_n_method_train, last_n_method_eval, last_n)

        # Only fit a single ladder
        (rel_error_step_1, rel_error_step_2, rel_errors_stacked), \
        (step_1_y, step_2_y, stacked_y), \
        (step_1_y_pred, step_2_y_pred, stacked_y_pred), \
        fit_error = _fit_ladder(data_by_name_step_1, data_by_name_step_2, data_by_name_stacked)

    elif last_n_method_train == 'all' or last_n_method_eval == 'all':
        rel_errors_step_1, rel_errors_step_2, rel_errors_stacked = [], [], []
        step_1_ys, step_2_ys, stacked_ys = [], [], []
        step_1_y_preds, step_2_y_preds, stacked_y_preds = [], [], []
        fit_errors = []
        for n in range(1, last_n):
            # Copy the data
            _data_by_name_step_1 = copy.deepcopy(data_by_name_step_1)
            _data_by_name_step_2 = copy.deepcopy(data_by_name_step_2)
            _data_by_name_stacked = copy.deepcopy(data_by_name_stacked)

            # Get only one of n results
            _data_by_name_step_1 = aggregate_list(_data_by_name_step_1, last_n_method_train, last_n_method_eval, n)
            _data_by_name_step_2 = aggregate_list(_data_by_name_step_2, last_n_method_train, last_n_method_eval, n)
            _data_by_name_stacked = aggregate_list(_data_by_name_stacked, last_n_method_train, last_n_method_eval, n)

            # Only fit a single ladder
            (rel_error_step_1, rel_error_step_2, _rel_errors_stacked), \
            (step_1_y, step_2_y, stacked_y), \
            (step_1_y_pred, step_2_y_pred, stacked_y_pred), \
            fit_error = _fit_ladder(_data_by_name_step_1, _data_by_name_step_2, _data_by_name_stacked)

            rel_errors_step_1.append(rel_error_step_1)
            rel_errors_step_2.append(rel_error_step_2)
            rel_errors_stacked.append(_rel_errors_stacked) 
            step_1_ys.append(step_1_y)
            step_2_ys.append(step_2_y)
            stacked_ys.append(stacked_y)
            step_1_y_preds.append(step_1_y_pred)
            step_2_y_preds.append(step_2_y_pred)
            stacked_y_preds.append(stacked_y_pred)
            fit_errors.append(fit_error)
        rel_error_step_1, rel_error_step_2, rel_errors_stacked = (rel_errors_step_1, rel_errors_step_2, rel_errors_stacked)
        step_1_y, step_2_y, stacked_y = (step_1_ys, step_2_ys, stacked_ys)
        step_1_y_pred, step_2_y_pred, stacked_y_pred = (step_1_y_preds, step_2_y_preds, stacked_y_preds)
        fit_error = fit_errors

    else:
        # Only fit a single ladder
        (rel_error_step_1, rel_error_step_2, rel_errors_stacked), \
        (step_1_y, step_2_y, stacked_y), \
        (step_1_y_pred, step_2_y_pred, stacked_y_pred), \
        fit_error = _fit_ladder(data_by_name_step_1, data_by_name_step_2, data_by_name_stacked)
    
    if return_reals:
        return (rel_error_step_1, rel_error_step_2, rel_errors_stacked), (step_1_y, step_2_y, stacked_y), (step_1_y_pred, step_2_y_pred, stacked_y_pred)
    if return_preds:
        return (rel_error_step_1, rel_error_step_2, rel_errors_stacked), (step_1_y_pred, step_2_y_pred, stacked_y_pred)
    if return_fit_error:
        return (rel_error_step_1, rel_error_step_2, rel_errors_stacked), fit_error
    return rel_error_step_1, rel_error_step_2, rel_errors_stacked



def fit_ladder(
        task_name, eval_models,
        configs, data_by_name,  data_by_name_step_2, data_by_name_stacked, task_key,
        y_metric='rc_bpb',  # "y_metric" is the metric type
        use_flops=False, use_single_step=False,
        run_step1=True, run_step2=True, run_stacked=True,
        axes=None, add_texts=False, plot_compute=False, 
    ):
    ax_i = 0

    if use_flops:
        # [min(data["fs"]) for data in data_by_name.values()]

        for name, data in data_by_name.items():
            fs = [float(6*n*d) for n, d in zip(data["ns"], data["ds"])]
            data_by_name[name]['fs'] = fs

        for name, data in data_by_name_stacked.items():
            fs = [float(6*n*d) for n, d in zip(data["ns"], data["ds"])]
            data_by_name_stacked[name]['fs'] = fs

    if use_single_step:
        assert not run_stacked and not run_step2, 'Single step prediction will only run step 1!'

    if run_step1 or run_stacked:
        # Fit step 1
        if use_single_step:
            step1_coefficients = fit_single_step(data_by_name, task_name=None, use_flops=use_flops)
        elif use_flops:
            step1_coefficients, cov = fit_step1_flops(data_by_name, y_metric)
        else:
            step1_coefficients, cov = fit_step1(data_by_name, y_metric)

        if use_single_step:
            (
                predicted_data_by_name_step_1, plotted_predicted_data,
                (step_1_y, step_1_y_pred, rel_error_step_1),
            ) = predict_single_step(
                data_by_name, step1_coefficients, use_flops=use_flops
            )
        elif use_flops:
            (
                predicted_data_by_name_step_1, plotted_predicted_data,
                (y, step_1_y_pred, rel_error_step_1), _,
            ) = predict_step1_flops(
                configs, data_by_name, step1_coefficients, y_metric=y_metric, 
            )
        else:
            (
                predicted_data_by_name_step_1, plotted_predicted_data,
                (y, step_1_y_pred, rel_error_step_1), _,
            ) = predict_step1(
                configs, data_by_name, step1_coefficients, y_metric=y_metric, 
            )

        # Plot step 1
        if axes is not None and run_step1:
            ax = axes[ax_i]
            ax_i += 1
            if use_single_step:
                print(data_by_name)
                plot_single_step(
                    configs, data_by_name, predicted_data_by_name_step_1, plotted_predicted_data,
                    task_name, str_combined_fit(step1_coefficients), use_flops, ax,
                )
            elif use_flops:
                plot_step1_flops(
                    configs, data_by_name, predicted_data_by_name_step_1, plotted_predicted_data,
                    task_name, str_chinchilla_flops_fit(step1_coefficients), y_metric,
                    step1_coefficients, cov, ax,
                )
            else:
                plot_step1(
                    configs, data_by_name, predicted_data_by_name_step_1, plotted_predicted_data,
                    task_name, str_chinchilla_n_d_fit(step1_coefficients), y_metric,
                    step1_coefficients, cov, ax,
                )

    fit_error = None
    if run_step2 or run_stacked:
        _min, _max = None, None
        if task_key is None:
            _min, _max = 0, 1 # TODO: Use utils.constants_task to get correct values

        try:
            # Fit step 2
            step2_coefficients, cov = fit_step2(data_by_name_step_2, task_key, y_metric=None, _min=_min, _max=_max, use_log_sigmoid=False)

            (
                predicted_data_by_name_step_2, plotted_predicted_data_step_2,
                (y, step_2_y_pred, rel_error_step_2, delta_error), all_rel_errors,
            ) = predict_step2(
                configs, data_by_name_step_2, step2_coefficients, cov, y_metric=None, use_log_sigmoid=False
            )

            # Plot step 2
            if axes is not None and run_step2:
                ax = axes[ax_i]
                ax_i += 1
                plot_step2(
                    configs, data_by_name_step_2, predicted_data_by_name_step_2, plotted_predicted_data_step_2, 
                    task_key, None, y_metric, 'rc_acc',
                    step2_coefficients, cov, use_log_sigmoid=False, add_texts=add_texts, ax=ax
                )

            all_y, all_pred_y = [], []
            for name, data in data_by_name_step_2.items():
                config = configs[name]
                if config.mode == "train":
                    continue
                predicted_data = predicted_data_by_name_step_2[name]
                all_y.extend(data["ys"])
                all_pred_y.extend(predicted_data["ys"])
            
            if len(all_y) > 0:
                fit_error = np.mean(np.abs(np.array(all_y) - np.array(all_pred_y)))
                # print(f"Mean error for all eval points: {me:.4f}")
                # mse = np.mean((np.array(all_y) - np.array(all_pred_y)) ** 2)
                # print(f"Mean squared error for all eval points: {mse:.4f}")
        except Exception as e:
            print(data_by_name_step_2)
            raise RuntimeError(f'Step 2 failed to fit: {e}')

    if run_stacked:
        # Predict stacked
        if use_flops:
            (
                predicted_data_by_name_stacked, plotted_predicted_data_by_name_stacked, 
                (y, stacked_y_pred, rel_error)
            ) = predict_chained_flops(
                data_by_name_stacked, step1_coefficients, step2_coefficients
            )
        else:
            (
                predicted_data_by_name_stacked, plotted_predicted_data_by_name_stacked, 
                (y, stacked_y_pred, rel_error)
            ) = predict_chained(
                data_by_name_stacked, step1_coefficients, step2_coefficients, use_log_sigmoid=False
            )

        # For stacked predictions, the x axis is now the y axis
        for key in data_by_name_stacked:
            data_by_name_stacked[key]['xs'] = data_by_name_stacked[key]['ys']

        # Plot stacked prediction
        if axes is not None:
            ax = axes[ax_i]
            if use_flops:
                plot_chained_flops(
                    configs,
                    data_by_name_stacked,
                    predicted_data_by_name_stacked,
                    plotted_predicted_data_by_name_stacked,
                    task_name,
                    str_chained_fit_flops(step1_coefficients, step2_coefficients),
                    ax,
                    plot_compute=plot_compute
                )
            else:
                plot_chained(
                    configs,
                    data_by_name_stacked,
                    predicted_data_by_name_stacked,
                    plotted_predicted_data_by_name_stacked,
                    task_name,
                    str_chained_fit(step1_coefficients, step2_coefficients, use_log_sigmoid=False),
                    ax,
                    plot_compute=plot_compute
                )
            ax.legend(loc='upper left')

        # if 'peteish7' in eval_models:
        #     # make 7B prediction
        #     n = 6887575552 
        #     d = 3945065873408 
        #     target_name = '7B-4T'

        #     pred_loss = chinchilla_n_d_fit([n, d], step1_coefficients)
        #     fit_fn = sigmoid
        #     pred_acc = fit_fn(pred_loss, *step2_coefficients)
        #     data = data_by_name[target_name]
        #     if "ys" in data:
        #         actual_acc = data["ys"][-1]
        #         delta_error=pred_acc - actual_acc
        #         rel_error_stacked = np.abs(delta_error) / actual_acc if actual_acc > 0 else float('inf')
        #         rel_errors_stacked += [rel_error_stacked]

    if axes is not None:
        for ax in axes:
            ax.set_title(task_name)
            # ax.legend(fontsize=6)

    # Add prediction results for models
    rel_error_step_1, rel_error_step_2, rel_errors_stacked = [], [], []
    step_1_y, step_2_y, stacked_y = [], [], []
    step_1_y_pred, step_2_y_pred, stacked_y_pred = [], [], []

    def compute_rel_error(data, predicted_data, target_name, key):
        y = data[target_name][key][0]
        y_pred = predicted_data[target_name][key][0]
        rel_error = np.abs(y_pred - y) / np.abs(y) # if y > 0 else float('inf')
        return y, y_pred, rel_error

    def process_step(data, predicted_data, target_name, key, y_list, y_pred_list, rel_error_list):
        y, y_pred, rel_error = compute_rel_error(data, predicted_data, target_name, key)
        y_list.append(y)
        y_pred_list.append(y_pred)
        rel_error_list.append(rel_error)

    def process_model(eval_models, model_name, target_name):
        if model_name in eval_models:
            if run_step1:
                process_step(data_by_name, predicted_data_by_name_step_1, target_name, ('xs' if not use_single_step else 'ys'), step_1_y, step_1_y_pred, rel_error_step_1)
            if run_step2:
                process_step(data_by_name_step_2, predicted_data_by_name_step_2, target_name, 'ys', step_2_y, step_2_y_pred, rel_error_step_2)
            if run_stacked:
                process_step(data_by_name_stacked, predicted_data_by_name_stacked, target_name, 'ys', stacked_y, stacked_y_pred, rel_errors_stacked)

    process_model(eval_models, 'peteish7', '7B-4T')
    process_model(eval_models, 'peteish7-last-5-model-merged', '7B-4T')
    process_model(eval_models, 'peteish7-last-30-model-merged', '13B-5T')

    process_model(eval_models, 'peteish13-highlr', '13B-5T')
    process_model(eval_models, 'peteish13-highlr-last-5-model-merged', '13B-5T')
    process_model(eval_models, 'peteish13-highlr-last-30-model-merged', '13B-5T')

    def simplify_list(lst):
        return lst[0] if len(lst) == 1 else lst

    step_1_y = simplify_list(step_1_y)
    step_2_y = simplify_list(step_2_y)
    stacked_y = simplify_list(stacked_y)
    step_1_y_pred = simplify_list(step_1_y_pred)
    step_2_y_pred = simplify_list(step_2_y_pred)
    rel_error_step_1 = simplify_list(rel_error_step_1)
    rel_error_step_2 = simplify_list(rel_error_step_2)
    rel_errors_stacked = simplify_list(rel_errors_stacked)
    
    return \
        (rel_error_step_1, rel_error_step_2, rel_errors_stacked), \
        (step_1_y, step_2_y, stacked_y), \
        (step_1_y_pred, step_2_y_pred, stacked_y_pred), \
        fit_error
