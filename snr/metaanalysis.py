from concurrent.futures import ProcessPoolExecutor
import os, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from snr.constants import ROOT_DIR
from snr.dataloader import get_slice
from snr.ladder_wrapper import run_ladder
from snr.stats import compute_total_variation
from snr.constants.datadecide import DATADECIDE_MODEL_NAMES
from snr.plot import plot_task_accuracy
from snr.constants import get_title_from_task
from snr.constants.models import MODEL_LIST_DATADECIDE_FINAL
from snr.metrics import decision_acc_fast
from snr.ladder_wrapper import sort_experiment_names
from snr.constants.datadecide import DATADECIDE_SIZES, DATADECIDE_COMPUTE

os.environ["MallocStackLogging"] = "0" # disable malloc logs for macos

DEFAULT_LADDER_CONFIG_PATH = f'{ROOT_DIR}/snr/constants/ladder_config.json'

ALL_METRICS = ['logits_per_char_corr', 'primary_score']
REVERSED_METRICS = ['margin_per_byte', 'norm_correct_prob_per_byte', 'correct_prob_per_byte', 'correct_logit_per_byte', 'logits_per_byte_corr']


def compute_2_class(ranking_a, ranking_b):
    """ Old version of decision accuracy """
    ranking_a = list(ranking_a)
    ranking_b = list(ranking_b)

    assert len(ranking_b) == len(ranking_b)
    
    n = len(ranking_a)
    same_order_count = 0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            i_pred = ranking_b.index(ranking_a[i])
            j_pred = ranking_b.index(ranking_a[j])
            
            if (i < j and i_pred < j_pred) or (i > j and i_pred > j_pred):
                same_order_count += 1
            total_pairs += 1
    
    return same_order_count / total_pairs if total_pairs > 0 else 0.0


def get_perf_size(df, size, task, metric, models=DATADECIDE_MODEL_NAMES, agg_method='max_n'):
    """ Get performance of all models at a specific size """
    _slice: pd.DataFrame = get_slice(df, task=task)
    _slice = _slice[(_slice['model'].isin(models))]
    _slice = _slice[((_slice['size'] == size))]
    if len(_slice) == 0:
        raise AssertionError(f"Slice is empty for models: {models}")
    if isinstance(task, str):
        _slice = _slice[_slice['task'] == task]
    elif isinstance(task, list):
        _slice = _slice[_slice['task'].isin(task)] 

    # Only aggregate numerical columns
    numerical_cols = _slice.select_dtypes(include='number').columns.tolist()
    non_numerical_cols = _slice.select_dtypes(exclude='number').columns.tolist()
    _slice = _slice.sort_values('step')

    def agg_func(group, method=agg_method):
        """For numerical columns, handle different aggregation methods"""
        num_aggs = {}
        for col in numerical_cols:
            if method == 'mean': # take the mean of final steps
                num_aggs[col] = group[col].mean()
            elif method == 'max_n': # take the final step
                num_aggs[col] = group[col].iloc[-1]
            elif method == 'sample': # sample from last n steps
                num_aggs[col] = group[col].sample(n=1).iloc[0]
            else:
                raise ValueError(method)
    
        # For non-numerical columns, take first value
        non_num_aggs = {col: group[col].iloc[0] for col in non_numerical_cols}
        
        return pd.Series({**num_aggs, **non_num_aggs})

    if agg_method is not None:
        # Aggregate points for different steps
        _slice = _slice.groupby(['model', 'task'], as_index=False).apply(lambda x: agg_func(x, agg_method))
        
    if isinstance(task, list):
        # Aggregate points for different subtasks
        _slice = _slice.groupby(['model', 'step'], as_index=False).apply(lambda x: agg_func(x, 'mean'))
        _slice['task_name'] = 'aggregate'

    _slice = _slice.reset_index().sort_values('step')[['model', 'mix', 'step', 'size', metric]]
    _slice['compute'] = _slice['flops']
    _slice = _slice.sort_values(metric, ignore_index=True)
    return _slice


def construct_2class_table(
        df, selected_tasks, 
        small_metric=ALL_METRICS, target_metric='primary_metric', 
        model_sizes=DATADECIDE_SIZES, 
        agg_method_pred='max_n',
        agg_method_target='max_n',
        merge_small_alias=None,
        merge_target_alias=None,
        n_samples=1000
    ):
    """ Compute 2-class accuracy. We are predicting primary_metric at 1B using the metric at a smaller scale """
    if not isinstance(small_metric, list): small_metric = [small_metric]

    combinations = list(itertools.product(small_metric, model_sizes, selected_tasks))
    two_class = pd.DataFrame(columns=['metric', 'size', 'task', 'accuracy'])

    for metric, size, task in tqdm(combinations, desc='Computing two class accuracy', disable=(len(combinations) < 50)):
        _slice = get_slice(df, task=task)
        _slice = _slice[((_slice['size'] == size)) & (_slice['model'].isin(DATADECIDE_MODEL_NAMES))] # get data for small scale
        if _slice.empty:
            raise RuntimeError(f"Empty slice for metric={metric}, size={size}, task={task}")
        steps = [sorted(_slice['step'].unique())[-1]]
        
        for step in steps:
            _agg_method_pred = agg_method_pred
            if agg_method_pred == 'sample':
                _agg_method_pred = None # disable aggregation within get_perf_size
            _agg_method_target = agg_method_target
            if agg_method_target == 'sample':
                _agg_method_target = None # disable aggregation within get_perf_size

            # get data at the small scale
            small_models = DATADECIDE_MODEL_NAMES
            if merge_small_alias is not None:
                small_models = [f'{model}-{merge_small_alias}' for model in small_models]
            small_scale = get_perf_size(
                df, size, task, metric, 
                agg_method=_agg_method_pred, models=small_models
            )

            # predict at the target scale (1B) 
            target_models = DATADECIDE_MODEL_NAMES
            if merge_target_alias is not None:
                target_models = [f'{model}-{merge_target_alias}' for model in target_models]
            target_scale = get_perf_size(
                df, '1B', task, target_metric, 
                agg_method=_agg_method_target, models=target_models
            )

            if agg_method_pred == 'sample' or agg_method_target == 'sample':
                # Convert small_scale into 2d array: (# models, # steps) ndarrays
                mixes = sorted(small_scale['mix'].unique())
                steps_per_mix = small_scale.groupby('mix').apply(lambda x: list(x.sort_values('step')[metric]))
                small_scale_array = [steps_per_mix[mix] for mix in mixes]
                
                target_steps_per_mix = target_scale.groupby('mix').apply(lambda x: list(x.sort_values('step')[metric]))
                target_scale_array = [target_steps_per_mix[mix] for mix in mixes]
                
                # For each trial, sample one value per mix and compute decision accuracy
                trial_accuracies = []
                
                quiet = True
                for _ in tqdm(range(n_samples), desc=f'Sampling for size={size}, task={get_title_from_task(task)}, metric={metric}', disable=quiet):
                    # Sample one value per mix for both sizes
                    sampled_scores_small = np.array([np.random.choice(values) for values in small_scale_array])
                    sampled_scores_1b = np.array([np.random.choice(values) for values in target_scale_array])

                    if metric in REVERSED_METRICS and target_metric not in REVERSED_METRICS: sampled_scores_small = -sampled_scores_small
                    
                    # Compute decision accuracy between sampled values
                    acc = decision_acc_fast(sampled_scores_small, sampled_scores_1b)
                    trial_accuracies.append(acc)

                accuracy = trial_accuracies
            else:
                small_scale = small_scale['mix']
                target_scale = target_scale['mix']

                if metric in REVERSED_METRICS and target_metric not in REVERSED_METRICS: small_scale = reversed(small_scale)
                try:
                    accuracy = compute_2_class(small_scale, target_scale)
                except Exception as e:
                    print((metric, size, task), e)
                    accuracy = float('-inf')

            # Get tokens/compute of small scale
            step_slice = _slice[_slice['step'] == float(step)]
            step_slice = step_slice.reset_index(drop=True)
            try:
                compute = step_slice['flops'][0]
            except Exception as e:
                print((metric, size, task), e)
                compute = float('-inf')

            new_entry = pd.DataFrame({
                'metric': [metric],
                'size': [size], 
                'step': [step], 
                'task': [str(task)],
                'accuracy': [accuracy],
                'compute': [compute]
            })
            new_entry = new_entry.dropna(axis=1, how='all')            
            two_class = two_class.dropna(axis=1, how='all')            
            two_class = pd.concat([two_class, new_entry], ignore_index=True)

    # Create two dataframes - one for best accuracies and one for corresponding metrics
    best_acc_df = two_class.loc[two_class.groupby(['task', 'size', 'step'])['accuracy'].idxmax()][['task', 'size', 'step', 'accuracy', 'compute']].reset_index(drop=True)
    best_metric_df = two_class.loc[two_class.groupby(['task', 'size', 'step'])['accuracy'].idxmax()][['task', 'size', 'step', 'metric', 'compute']].reset_index(drop=True)

    # Create pivot tables with size in specified order
    acc_pivot = best_acc_df.pivot(index='task', columns=['size', 'compute'], values='accuracy')[model_sizes]
    metric_pivot = best_metric_df.pivot(index='task', columns=['size', 'compute'], values='metric')[model_sizes]

    return two_class, acc_pivot, metric_pivot


def run_analysis(
        df, task, ladder_models, external_ladder_models, eval_ladder_models, 
        metric='primary_score', axes=None, small_fig=False, 
        ladder_config_path=DEFAULT_LADDER_CONFIG_PATH
    ):
    results = {}

    # Observational noise
    observational_models = external_ladder_models+eval_ladder_models+DATADECIDE_MODEL_NAMES
    _slice = get_slice(df, task=task, model=observational_models)
    numerical_cols     = [col for col in _slice.select_dtypes(include='number').columns if col != 'extracted_size']
    non_numerical_cols = _slice.select_dtypes(exclude='number').columns.tolist() + ['extracted_size']
    _slice = _slice.groupby('model', as_index=False).agg({col: 'mean' for col in numerical_cols} | {col: 'first' for col in non_numerical_cols})
    weight_classes = {
        'olmo2_1b':  {'flops_range': (2.1176470588235293e+22, 6.12e+22)}, # 'count': 8,
        'olmo2_7b':  {'flops_range': (9.88235294117647e+22, 2.8559999999999996e+23)}, # 'count': 14,
        'olmo2_13b': {'flops_range': (2.2941176470588235e+23, 6.63e+23)}, # 'count': 9,
        'olmo2_32b': {'flops_range': (6.764705882352941e+23, 1.955e+24)}, # 'count': 11,
    }
    observational_metrics = ['primary_score', 'logits_per_char_corr', 'logits_per_byte_corr']
    for observational_metric in observational_metrics:
        for weight_label, weight_range in weight_classes.items():
            size_label = weight_label.split('_')[-1].upper() # olmo2_32b => 32B
            lower, upper = weight_range['flops_range']

            _weight_class_scores = _slice[(_slice['flops'] >= lower) & (_slice['flops'] <= upper)]

            # assert len(_weight_class_scores) > 0, f'Found no external models for weight class: {weight_class}'

            _weight_class_scores = _weight_class_scores[observational_metric]

            results.update({
                f'mean:{observational_metric}:{size_label}': _weight_class_scores.mean(),
                f'range:{observational_metric}:{size_label}': _weight_class_scores.max() - _weight_class_scores.min(),
                f'std_dev:{observational_metric}:{size_label}': _weight_class_scores.std()
            })
    
    # Scaling laws
    # primary_score_name = PRIMARY_METRICS_OLMES[task] if isinstance(task, str) and task in PRIMARY_METRICS_OLMES else 'primary_score'
    primary_score_name = 'primary_score'
    try:
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignore fitting warnings

        # Standard error around the ladder prediction
        rel_errors_step_1, rel_errors_step_2, rel_errors_stacked = run_ladder(
            df,
            task,
            train_models=ladder_models,
            eval_models=["peteish13-highlr"], # "peteish7",
            downstream_feature=metric,
            config_path=ladder_config_path,
            run_step1=True, run_step2=False,
            last_n_method_train='final', last_n_method_eval='all', last_n=30
        )

        # Calculate margin-of-error using the set of ladder errors
        def calc_margin_of_err(errors, confidence_level = 0.95):
            errors = np.array(errors)
            n = len(errors)
            std_error = np.std(errors, ddof=1) / np.sqrt(n)
            margin_of_error = std_error * stats.t.ppf((1 + confidence_level) / 2, n - 1)
            return margin_of_error
        
        # Calculate std dev
        def calc_std_dev(errors):
            return np.std(np.array(errors), ddof=1)
        
        results.update({
            "scaling_margin_of_error:step_1:13B:bpb_to_primary": calc_margin_of_err(rel_errors_step_1), 
            "scaling_margin_of_error:step_2:13B:bpb_to_primary": calc_margin_of_err(rel_errors_step_2), 
            "scaling_margin_of_error:stacked:13B:bpb_to_primary": calc_margin_of_err(rel_errors_stacked), 
            "scaling_std_dev:step_1:13B:bpb_to_primary": calc_std_dev(rel_errors_step_1), 
            "scaling_std_dev:step_2:13B:bpb_to_primary": calc_std_dev(rel_errors_step_2), 
            "scaling_std_dev:stacked:13B:bpb_to_primary": calc_std_dev(rel_errors_stacked), 
        })

        # Step 1 ladder prediction (base models)
        ax = None
        if not small_fig:
            ax: plt.Axes = axes[0, 0] if axes is not None else None
        rel_error_step_1, _, _ = run_ladder(
            df,
            task,
            train_models=ladder_models,
            eval_models=["peteish7", "peteish13-highlr"],
            config_path=ladder_config_path,
            run_step2=False, run_stacked=False,
            axes=[ax]
        )
        results.update({
            "rel_error:step_1:7B:bpb_to_primary": rel_error_step_1[0], 
            "rel_error:step_1:13B:bpb_to_primary": rel_error_step_1[1], 
        })
        if ax:
            ax.set_ylabel('Task loss (BPB)')
            ax.legend(fontsize=6)

        # Step 2 ladder prediction (base models)
        ax = None
        if not small_fig:
            ax: plt.Axes = axes[1, 0] if axes is not None else None
        _, rel_error_step_2, _ = run_ladder(
            df,
            task,
            train_models=ladder_models,
            eval_models=["peteish7", "peteish13-highlr"],
            downstream_feature=metric,
            config_path=ladder_config_path,
            run_step1=False, run_stacked=False,
            axes=[ax]
        )
        results.update({
            "rel_error:step_2:7B:bpb_to_primary": rel_error_step_2[0], 
            "rel_error:step_2:13B:bpb_to_primary": rel_error_step_2[1], 
        })
        if ax:
            ax.set_xlabel('Task loss (BPB)')
            # ax.set_ylabel(primary_score_name)
            ax.set_ylabel(metric)
            ax.legend(fontsize=6)

        # Step 2 ladder prediction (external models)
        ax: plt.Axes = axes[2, 0] if axes is not None else None
        _, mean_error_step_2 = run_ladder(
            df,
            task,
            train_models=ladder_models + external_ladder_models,
            eval_models=eval_ladder_models,
            config_path=ladder_config_path,
            run_step1=False, run_stacked=False,
            return_fit_error=True,
            axes=[ax]
        )
        results.update({
            "mean_error:step_2:external:bpb_to_primary": mean_error_step_2, 
        })
        if ax:
            ax.get_legend().remove()
            # ax.legend(fontsize=3, ncols=2)
            ax.set_xlabel('Task loss (BPB)')
            ax.set_ylabel(primary_score_name)
            ax.text(
                x=0.02, y=0.02, s=f'Mean Error={mean_error_step_2*100:.2f}%',
                transform=ax.transAxes,
                va='bottom', ha='left',
                fontsize=8
            )
            ax.set_title('Perplexity -> Task Metric')
        
        # Stacked ladder prediction
        ax: plt.Axes = axes[3, 0] if axes is not None else None
        _, _, rel_error_stacked = run_ladder(
            df,
            task,
            train_models=ladder_models,
            eval_models=["peteish7", "peteish13-highlr"],
            downstream_feature=metric,
            config_path=ladder_config_path,
            run_step1=False, run_step2=False,
            axes=[ax]
        )
        results.update({
            "rel_error:stacked:7B:bpb_to_primary": rel_error_stacked[0], 
            "rel_error:stacked:13B:bpb_to_primary": rel_error_stacked[1], 
        })
        if ax:
            # ax.set_ylabel(primary_score_name)
            ax.set_ylabel(metric)
            ax.legend(fontsize=6)
            ax.set_title('Scaling Law Prediction')

        # fig, ax = plt.subplots(figsize=(10, 6))
        # plt.savefig(os.path.join(PLOT_DIR, f'debug:{task}:{metric}.pdf'))
        # plt.close()

        # Stacked prediction -- C4 as intermediate feature
        rel_error_step_1, _, rel_error_stacked = run_ladder(
            df,
            task,
            train_models=[model for model in ladder_models if 'peteish-moreeval-1B-1xC' not in model],
            # eval_models=["peteish7", "peteish13-highlr"],
            eval_models=["peteish13-highlr"],
            # Use C4 loss for intermediate feature!
            intermediate_task_name="paloma_c4_en",
            intermediate_feature='logits_per_byte_corr', 
            downstream_feature=metric, # 'primary_score', 
            config_path=ladder_config_path,
        )
        results.update({
            # "rel_error:step_1:7B:c4_to_primary": rel_error_step_1[0], 
            # "rel_error:step_1:13B:c4_to_primary": rel_error_step_1[1], 
            # "rel_error:stacked:7B:c4_to_primary": rel_error_stacked[0], 
            # "rel_error:stacked:13B:c4_to_primary": rel_error_stacked[1], 

            "rel_error:step_1:13B:c4_to_primary": rel_error_step_1, 
            "rel_error:stacked:13B:c4_to_primary": rel_error_stacked, 
        })
    except Exception as e:
        print(task, 'failed on ladder fits', e)
        # raise RuntimeError(task, 'failed on ladder fits', e)

    # Step-to-step noise
    intermediate_models = ['peteish-moreeval-1B-5xC', 'peteish1', 'peteish7', 'peteish13-highlr', 'peteish32'] # peteish-moreeval-1B-5xC
    intermediate_model_names = ['1B-100B', '1B', '7B', '13B', '32B']
    for j, model in enumerate(intermediate_models):
        model_name = intermediate_model_names[j]

        # logits_per_char_corr intermediate checkpoinrts
        if small_fig:
            ax: plt.Axes = axes[2+j, 1] if axes is not None else None
        else:
            ax: plt.Axes = axes[0+(j*2), 1] if axes is not None else None
        tv, _ = compute_total_variation(
            df, models=[model], metric='logits_per_char_corr', tasks=[task], axes=[ax]
        )
        tv_bpb = tv[task]['total_variation'] if not isinstance(task, list) else tv.loc['total_variation']['aggregate']
        if ax and ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=6)
            ax.set_ylabel('Task loss (BPB)')
            ax.set_title('Smoothness')
            
            # Get the y-values from the current axis
            lines = ax.get_lines()
            if len(lines) > 0:
                y_data = lines[0].get_ydata()
                # Set top limit 10% above max y value
                y_max = np.max(y_data)
                # Get y-value 10% into the curve for bottom limit
                idx = int(len(y_data) * 0.1)
                y_20_percent = y_data[idx]
                if not (np.isnan(y_20_percent) or np.isnan(y_max) or np.isinf(y_20_percent) or np.isinf(y_max)):
                    ax.set_ylim(bottom=y_20_percent, top=y_max * (0.95 if y_max < 0 else 1.05))

        results.update({
            f'tv:logits_per_char_corr:{model_name}': tv_bpb
        })

        # primary_metric intermediate checkpoinrts
        ax = None
        if not small_fig:
            ax: plt.Axes = axes[1+(j*2), 1] if axes is not None else None
        tv, _ = compute_total_variation(
            df, models=[model], metric=metric, tasks=[task], axes=[ax]
        )
        tv_primary = tv[task]['total_variation'] if not isinstance(task, list) else tv.loc['total_variation']['aggregate']
        if ax and ax.get_legend_handles_labels()[1]:
            # ax.set_ylabel(primary_score_name)
            ax.set_ylabel(metric)
            ax.legend(fontsize=6)

        results.update({
            f'tv:{metric}:{model_name}': tv_primary
        })

        # Additional metric calculations
        additional_metrics = ['primary_score', 'logits_per_char_corr', 'logits_per_byte_corr']
        for additional_metric in additional_metrics:
            try:
                tv, _ = compute_total_variation(
                    df, models=[model], metric=additional_metric, tasks=[task]
                )
                tv_result = tv[task]['total_variation'] if not isinstance(task, list) else tv.loc['total_variation']['aggregate']
                tv_result_no_norm = tv[task]['total_variation:no_norm'] if not isinstance(task, list) else tv.loc['total_variation:no_norm']['aggregate']
                step_std_20 = tv[task]['step_std:perc20'] if not isinstance(task, list) else tv.loc['step_std:perc20']['aggregate']
                step_rel_std_20 = tv[task]['step_rel_std:perc20'] if not isinstance(task, list) else tv.loc['step_rel_std:perc20']['aggregate']
                step_std_10 = tv[task]['step_std:last10'] if not isinstance(task, list) else tv.loc['step_std:last10']['aggregate']
                step_rel_std_10 = tv[task]['step_rel_std:last10'] if not isinstance(task, list) else tv.loc['step_rel_std:last10']['aggregate']
                step_std_30 = tv[task]['step_std:last30'] if not isinstance(task, list) else tv.loc['step_std:last30']['aggregate']
                step_rel_std_30 = tv[task]['step_rel_std:last30'] if not isinstance(task, list) else tv.loc['step_rel_std:last30']['aggregate']
                
                results.update({
                    f'tv:{additional_metric}:{model_name}': tv_result,
                    f'tv_no_norm:{additional_metric}:{model_name}': tv_result_no_norm,
                    f'step_std:perc20:{additional_metric}:{model_name}': step_std_20,
                    f'step_rel_std:perc20:{additional_metric}:{model_name}': step_rel_std_20,
                    f'step_std:last10:{additional_metric}:{model_name}': step_std_10,
                    f'step_rel_std:last10:{additional_metric}:{model_name}': step_rel_std_10,
                    f'step_std:last30:{additional_metric}:{model_name}': step_std_30,
                    f'step_rel_std:last30:{additional_metric}:{model_name}': step_rel_std_30
                })
            except Exception as e:
                print(task, f'failed to compute total variation for {additional_metric}', e)

    # Decision accuracy
    try:
        two_class, acc_pivot_bpb_primary, metric_pivot = construct_2class_table(
            df, [task], small_metric='logits_per_byte_corr', target_metric='primary_score'
        )
        two_class_results = acc_pivot_bpb_primary.loc[str(task)].unstack()
        if axes is not None and not small_fig:
            ax: plt.Axes = axes[1, 2]
            plot_task_accuracy(ax, two_class_results, str(task), DATADECIDE_COMPUTE)
            ax.set_ylabel(f'Decision Acc (BPB on {primary_score_name})')
            ax.set_ylim(0.75, 1)

        two_class, acc_pivot_best_metric, metric_pivot = construct_2class_table(
            df, [task], small_metric=metric, target_metric=metric
        )
        two_class_results = acc_pivot_best_metric.loc[str(task)].unstack()
        if axes is not None and not small_fig:
            ax: plt.Axes = axes[2, 2]
            plot_task_accuracy(ax, two_class_results, str(task), DATADECIDE_COMPUTE)
            ax.set_ylabel(f'Decision Acc (best on {primary_score_name})')
            ax.set_ylim(0.75, 1)

        results.update({
            f"dec_acc:{metric}:4M": acc_pivot_best_metric['4M'].loc[str(task)].item(),
            f"dec_acc:{metric}:20M": acc_pivot_best_metric['20M'].loc[str(task)].item(),
            f"dec_acc:{metric}:60M": acc_pivot_best_metric['60M'].loc[str(task)].item(),
            f"dec_acc:{metric}:90M": acc_pivot_best_metric['90M'].loc[str(task)].item(),
            f"dec_acc:{metric}:150M": acc_pivot_best_metric['150M'].loc[str(task)].item(),
            f"dec_acc:{metric}:300M": acc_pivot_best_metric['300M'].loc[str(task)].item(),
            f"dec_acc:{metric}:530M": acc_pivot_best_metric['530M'].loc[str(task)].item(),
            f"dec_acc:{metric}:750M": acc_pivot_best_metric['750M'].loc[str(task)].item(),
        })
            

        two_class, acc_pivot_bpb, metric_pivot = construct_2class_table(
            df, [task], 
            small_metric='logits_per_byte_corr', target_metric='logits_per_byte_corr'
        )
        two_class_results = acc_pivot_bpb.loc[str(task)].unstack()
        if axes is not None:
            ax: plt.Axes = axes[3, 2]
            plot_task_accuracy(ax, two_class_results, str(task), DATADECIDE_COMPUTE, show_legend=True)
            ax.legend(fontsize=6, ncols=2)
            ax.set_ylabel('Decision Acc (BPB on BPB)')
            ax.set_ylim(0.75, 1)
            ax.set_title('Decision Accuracy')

        results.update({
            "dec_acc:logits_per_byte_corr:4M": acc_pivot_bpb['4M'].loc[str(task)].item(),
            "dec_acc:logits_per_byte_corr:20M": acc_pivot_bpb['20M'].loc[str(task)].item(),
            "dec_acc:logits_per_byte_corr:60M": acc_pivot_bpb['60M'].loc[str(task)].item(),
            "dec_acc:logits_per_byte_corr:90M": acc_pivot_bpb['90M'].loc[str(task)].item(),
            "dec_acc:logits_per_byte_corr:150M": acc_pivot_bpb['150M'].loc[str(task)].item(),
            "dec_acc:logits_per_byte_corr:300M": acc_pivot_bpb['300M'].loc[str(task)].item(),
            "dec_acc:logits_per_byte_corr:530M": acc_pivot_bpb['530M'].loc[str(task)].item(),
            "dec_acc:logits_per_byte_corr:750M": acc_pivot_bpb['750M'].loc[str(task)].item(),
        })

        additional_metrics = ['primary_score', 'logits_per_char_corr', 'logits_per_byte_corr']
        for additional_metric in additional_metrics:
            two_class, acc_pivot_bpb, metric_pivot = construct_2class_table(
                df, [task], 
                small_metric=additional_metric, target_metric=additional_metric
            )
            two_class_results = acc_pivot_bpb.loc[str(task)].unstack()
            results.update({
                f"dec_acc:{additional_metric}:4M": acc_pivot_bpb['4M'].loc[str(task)].item(),
                f"dec_acc:{additional_metric}:20M": acc_pivot_bpb['20M'].loc[str(task)].item(),
                f"dec_acc:{additional_metric}:60M": acc_pivot_bpb['60M'].loc[str(task)].item(),
                f"dec_acc:{additional_metric}:90M": acc_pivot_bpb['90M'].loc[str(task)].item(),
                f"dec_acc:{additional_metric}:150M": acc_pivot_bpb['150M'].loc[str(task)].item(),
                f"dec_acc:{additional_metric}:300M": acc_pivot_bpb['300M'].loc[str(task)].item(),
                f"dec_acc:{additional_metric}:530M": acc_pivot_bpb['530M'].loc[str(task)].item(),
                f"dec_acc:{additional_metric}:750M": acc_pivot_bpb['750M'].loc[str(task)].item(),
            })

        # Compute range and std dev between models at each compute scale
        for additional_metric in additional_metrics:
            for size in DATADECIDE_SIZES:
                scores = get_perf_size(df, size, task, additional_metric)[additional_metric]
                if size == '1B':
                    size = '1B-100B' # rename to not confuse with OLMo 2 1B-4T
                results.update({
                    f'mean:{additional_metric}:{size}': scores.mean(),
                    f'range:{additional_metric}:{size}': scores.max() - scores.min(),
                    f'std_dev:{additional_metric}:{size}': scores.std()
                })
    except Exception as e:
        print(task, 'failed on consistent ranking analysis', e)
        # raise RuntimeError(task, 'failed on consistent ranking analysis', e)

    if axes is not None:
        for ax in axes.flat:
            ax.set_ylabel(ax.get_ylabel(), fontsize=10)
            if not small_fig:
                ax.set_title(get_title_from_task(task))

            if not ax.has_data():
                ax.remove()

    # SNR
    snr_metrics = ['primary_score', 'logits_per_char_corr', 'logits_per_byte_corr']
    for snr_metric in snr_metrics:
        for size in DATADECIDE_SIZES + ['1B-100B', '1B', '7B', '13B', '32B']:
            if f'mean:{snr_metric}:{size}' in results and \
                f'std_dev:{snr_metric}:{size}' in results and \
                f'step_rel_std:last30:{snr_metric}:{size}' in results:

                mean = results[f'mean:{snr_metric}:{size}']
                data_std_dev = results[f'std_dev:{snr_metric}:{size}']
                step_rel_std = results[f'step_rel_std:last30:{snr_metric}:{size}']
                
                data_rel_std = abs(data_std_dev / mean) if abs(mean) > 0 else 0
                snr = data_rel_std / abs(step_rel_std) if abs(step_rel_std) > 0 else float('-inf')
                
                results[f'rel_std:{snr_metric}:{size}'] = data_rel_std
                results[f'snr:{snr_metric}:{size}'] = snr
    
    # Total cost of evaluation
    try:
        task_as_list = [task] if isinstance(task, str) else task
        total_instances = 0
        for subtask in task_as_list:
            task_results = get_slice(df, task=subtask)
            num_instances = task_results['num_instances'].iloc[0]
            
            if 'hellaswag' in task_results['task'].unique():
                # override for HS
                task_results.loc[task_results['task'] == 'hellaswag', 'num_instances'] = 10042 
            if 'squad' in task_results['task'].unique():
                # override for SQuAD
                task_results.loc[task_results['task'] == 'squad', 'num_instances'] = 10570 
            
            assert (task_results['num_instances'] == num_instances).all(), \
                f"num_instances should be constant across task={subtask} for task_as_list={task_as_list}"
            total_instances += num_instances
        total_instances = int(total_instances)
        results.update({
            "n_instances": total_instances,
            "total_cost": total_instances,
            "total_cost_div_4": total_instances / 4 # Hacky way to estimate BPB vs. RC cost (assumes all tasks have 4 answer choices)
        })
    except Exception as e:
        print('Failed to calculate compute cost:', e)

    return results


def compute_metaproperties(
        df_benchmarks, selected_tasks, 
        use_parallel=True, quiet=False
    ):
    task_names = [get_title_from_task(task) for task in selected_tasks]

    external_models = list(df_benchmarks[df_benchmarks['model_type'] == 'external']['model'].unique())
    ladder_models = list(df_benchmarks[df_benchmarks['model_type'] == 'ladder']['model'].unique())
    llama_3_models = [model for model in external_models if "Llama-3" in model]
    ladder_models = sort_experiment_names(ladder_models)

    benchmark_results = []
    instance_results = []

    # Run benchmark analysis
    benchmark_args = []
    for task in selected_tasks:
        benchmark_args.append({
            'df': df_benchmarks,
            'task': task,
            'ladder_models': ladder_models,
            'eval_ladder_models': ladder_models + llama_3_models,
            'external_ladder_models': external_models,
        })
    
    if not use_parallel:
        benchmark_results = []
        for kwargs in tqdm(benchmark_args, desc="Computing benchmark properties"):
            benchmark_results.append(run_analysis(**kwargs))
    else:
        with ProcessPoolExecutor() as executor:
            futures = []
            for kwargs in benchmark_args:
                futures.append(executor.submit(run_analysis, **kwargs))
            
            benchmark_results = list(tqdm(
                (f.result() for f in futures),
                total=len(benchmark_args),
                desc="Computing benchmark properties"
            ))

    # Create dataframe, filling in missing results as -inf
    all_keys = set().union(*benchmark_results)
    normalized_results = [{key: d.get(key, float('-inf')) for key in all_keys} for d in benchmark_results]
    df_benchmark_results = pd.DataFrame(normalized_results, index=task_names)
    df_instance_results = pd.DataFrame(instance_results, index=task_names)
    df_results = pd.concat([df_benchmark_results, df_instance_results], axis=1)

    # Remove duplicate results if they exist
    n_duplicates = len(df_results.index) - len(df_results.index.unique())
    if n_duplicates > 0:
        print(f"Removing {n_duplicates} duplicates")
        df_results = df_results[~df_results.index.duplicated()]

    return df_results

def run_single_ladder(df, task, train_models, eval_models, ladder_config_path):
    _, _, stacked_error = run_ladder(
        df,
        task,
        train_models=train_models,
        eval_models=eval_models,
        config_path=ladder_config_path,
        plot_compute=True,
        run_step1=False, run_step2=False,
        
        last_n=6, last_n_method='sample'
    )
    return stacked_error