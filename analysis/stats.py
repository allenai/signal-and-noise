import itertools
from pathlib import Path
import pandas as pd
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os, sys, warnings

from utils import DATA_DIR, get_title_from_task
from plot import plot_heatmap, plot_training
from dataloader import get_nd_array, get_slice


def calc_total_variation(arr, norm=False, improvement=False):
    """ Total variation """
    if len(arr) == 0: return 0
    arr = np.array(arr)

    tv = np.mean(np.abs(np.diff(arr)))

    if tv == 0:
        return tv

    if norm: tv /= (max(arr) - min(arr))
    if improvement: tv -= calc_improvement(arr)

    return tv


def calc_monotonicity(arr):
    diffs = np.diff(arr)
    pos = np.sum(diffs > 0)
    neg = np.sum(diffs < 0)
    return (pos - neg) / (pos + neg) if (pos + neg) != 0 else 0


def calc_improvement(arr):
    if len(arr) == 0: return 0
    return (arr[-1] - arr[0]) / len(arr)


def calc_improvement_last_n(arr, n=5):
    if len(arr) == 0: return 0
    return (sum(arr[-n:]) / n - sum(arr[:n]) / n) / len(arr)


def convert_sci(n):
    """ Convert to 10^X notation """
    exponent = math.floor(math.log10(n))
    mantissa = n / (10 ** exponent)
    result = f"{mantissa:.2f} * 10^{exponent}"
    return result


def compute_f1(gold_mixes, pred_mixes):
    gold_set = set(gold_mixes)
    pred_set = set(pred_mixes)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1 # we really care about precision (minimize false negatives)


def compute_f1_binary(gold_arr, pred_arr):
    if len(gold_arr) != len(pred_arr):
        raise ValueError("Input arrays must have the same length")
    
    tp = sum((g == 1 and p == 1) for g, p in zip(gold_arr, pred_arr))
    fp = sum((g == 0 and p == 1) for g, p in zip(gold_arr, pred_arr))
    fn = sum((g == 1 and p == 0) for g, p in zip(gold_arr, pred_arr))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def compute_decision_accuracy(mixes_1b, mixes_size):
    # Count pairs that agree in relative ordering
    agree_count = 0
    total_pairs = 0
    for i in range(len(mixes_1b)):
        for j in range(i+1, len(mixes_1b)):
            mix1_1b, mix2_1b = mixes_1b[i], mixes_1b[j]
            # Find positions of same mixes in size ordering
            try:
                pos1_size = mixes_size.index(mix1_1b)
                pos2_size = mixes_size.index(mix2_1b)
                # Check if relative ordering agrees
                if (pos1_size < pos2_size) == (i < j):
                    agree_count += 1
                total_pairs += 1
            except ValueError:
                continue

    decision_accuracy = agree_count / total_pairs if total_pairs > 0 else 0
    return decision_accuracy


def reorder_items_and_ranks(itemsA, itemsB, ranksA, ranksB):
    index_map = {value: i for i, value in enumerate(itemsA)}
    indices =  [index_map[value] for value in itemsB]
    sorted_itemsB = [itemsB[i] for i in indices]
    ranksB_sorted = [ranksB[i] for i in indices]
    return itemsA, sorted_itemsB, ranksA, ranksB_sorted


def kendall_tau_a(itemsA, itemsB, ranksA, ranksB):
    """
    # Example usage
    itemsA = ["apple", "banana", "cherry", "date", "elderberry"]
    itemsB = ["apple", "banana", "cherry", "elderberry", "date"]
    ranksA = [1, 2, 3, 4, 5]
    ranksB = [1, 2, 3, 5, 4]
    print(kendall_tau_a(itemsA, itemsB, ranksA, ranksB))
    """
    itemsA, itemsB, ranksA, ranksB = reorder_items_and_ranks(itemsA, itemsB, ranksA, ranksB)

    if len(itemsA) != len(ranksA) or len(itemsB) != len(ranksB) or len(itemsA) != len(itemsB):
        raise ValueError("All input lists must have the same length.")
    
    n = len(itemsA)
    concordant, discordant = 0, 0
    
    for (i, j) in itertools.combinations(range(n), 2):
        pair_x = np.sign(ranksA[i] - ranksA[j])
        pair_y = np.sign(ranksB[i] - ranksB[j])
        
        if pair_x * pair_y > 0:
            concordant += 1
        elif pair_x * pair_y < 0:
            discordant += 1
    
    tau_a = (concordant - discordant) / (0.5 * n * (n - 1))
    return tau_a


def perc_significant(p_values, alpha=0.05):
    """ Calculate the % of statistically significant comparisons """
    return ((p_values > (1-alpha)).sum() + (p_values < alpha).sum()) / (~np.isnan(p_values)).sum()


def calculate_standard_error(avg_score, num_scores):
    """ https://arxiv.org/pdf/2411.00640#page=2.55 """
    return np.sqrt((avg_score * (1 - avg_score)) / num_scores)


def create_stratified_array(counts):
    """ Convert counts to 1D array of weights: [1172, 2304] => [1172, 1172, ..., 2304, 2304, ...] """
    return np.concatenate([
        np.full(count, count, dtype=np.float64) 
        for value, count in enumerate(counts)
    ])


def get_bound(arr, idx, alpha):
    """ Get the first index where arr[i] < alpha """
    condition = (arr < alpha) # | (arr > (1-alpha))
    indices = np.argwhere(condition)

    if indices.size > 0:
        first_index = tuple(indices[0])
    else:
        return len(arr) + idx

    return first_index[0] + idx


def get_sig_cluster_bound(p_vals, idx, alpha):
    """ Given an idx and p_vals, compute the boundary of the next significance cluster """
    col = p_vals[:, idx][idx+1:]
    row = p_vals[idx, :][idx+1:]
    return min(get_bound(col, idx, alpha=alpha), get_bound(row, idx, alpha=alpha)) + 1 # PERM-INPUTS clustering
    # return max(get_bound(col, idx, alpha=alpha), get_bound(row, idx, alpha=alpha)) + 1 # conservative clustering (gives less clusters)


def get_sig_clusters(p_vals, alpha=0.01):
    """
    The pièce de résistance.

    Start with highest scoring mix, assign rank 1 to all mixes until we have 
    encountered a mix statistically significantly different from any mix so far.
    """
    sig_clusters = np.zeros(p_vals.shape[0])

    curr, curr_cluster = 0, 0
    while curr < p_vals.shape[0]:
        idx = curr
        cluster_bound = get_sig_cluster_bound(p_vals, idx, alpha)
        for _ in range(idx, cluster_bound):
            sig_clusters[curr] = curr_cluster
            curr += 1
        curr_cluster += 1

    # # Draw cluster boundaries conservatively
    # n = p_vals.shape[0]
    # count = -1
    # for i in range(n):  
    #     if all(p_vals[j, i] < 0.05 for j in range(i)):  
    #         count += 1
    #     sig_clusters[i] = count

    return sig_clusters


def compute_significance(
    df, models, metric, tasks=None, 
    aggregator='macro', # macro, micro, irt (for single tasks, macro avg == micro avg)
    step='max', last_n=1, alpha=0.05, num_permutations=1_000, binarize=False,
    do_plot=False, pretty_mix_names=None, plot_sig_clusters=True, plot_clean=False, quiet=False):
    if tasks is None: 
        tasks = df.index.get_level_values('task').unique()

    sig_results = pd.DataFrame(index=['perc_sig'], columns=tasks)
    all_p_values = {}

    n_tasks = len(tasks)
    if do_plot is not None: 
        if isinstance(do_plot, plt.Axes):
            axes = [do_plot] # allow passing in an axes object for plotting
        elif isinstance(do_plot, bool):
            if do_plot:
                fig, axes = plt.subplots(n_tasks, 1, figsize=(0.5*len(models), 0.4*len(models)*n_tasks))
                if n_tasks == 1: axes = [axes]
            else:
                do_plot = None
        else:
            axes = do_plot

    for i, task in tqdm(enumerate(tasks), desc='Computing pairwise comparisons', total=len(tasks), disable=quiet):
        task_name = get_title_from_task(task)

        if last_n > 1:
            assert step == 'max'
            
            instance_names, mixes, scores = get_nd_array(df, ['mix', 'step'], 'acc_per_char', model=models, task=task, return_index=True)

            scores = scores[:, -last_n:, :] # get last n steps

            scores = scores.mean(axis=1) # approach 1: average over last n ckpts
            # scores = scores.reshape(scores.shape[0], scores.shape[1] * scores.shape[2]) # approach 2: concat last n ckpts -- I'm concerned this wont work with instance weights

            # Recover just the mix names
            mixes = np.array([name for name, step in mixes])

            # Sort based on new aggregate (average/concat)
            mix_sums = scores.sum(axis=1)
            sorted_indices = mix_sums.argsort()[::-1]
            mixes = mixes[sorted_indices].tolist()
            scores = scores[sorted_indices]
        else:
            if metric == 'logits_per_byte':
                # TMP: Handle 3D array
                from ladder_wrapper import map_corr_labels
                instance_names, mixes, bpb = get_nd_array(df, ['model', 'step', 'mix'], metric, model=models, task=task, step=step, return_index=True)
                mixes = np.array([mix for mix, _, _ in mixes])
                bpb = bpb[:, 0, :]
                _, corr = get_nd_array(df, ['model', 'step', 'mix'], 'correct_choice', model=models, task=task, step=step)
                corr = corr[:, 0, :]
                correct_bpb = map_corr_labels(bpb, corr, task_name=task)
                scores = correct_bpb
                # Sort by overall performance
                mix_sums = scores.sum(axis=1)
                sorted_indices = mix_sums.argsort()[::-1]
                mixes = mixes[sorted_indices].tolist()
                scores = scores[sorted_indices]
            else:
                # instance_names, mixes, scores = get_nd_array(df, 'mix', metric, model=models, task=task, step=step, sorted=True, return_index=True)
                instance_names, mixes, scores = get_nd_array(df, 'mix', metric, model=models, task=task, step=step, sorted=False, return_index=True)

                # Sort by overall performance (get_nd_array sorting is broken!)
                mix_sums = scores.sum(axis=1)
                sorted_indices = mix_sums.argsort()[::-1]
                mixes = np.array(mixes)[sorted_indices].tolist()
                scores = scores[sorted_indices]
        
        if binarize:
            # Check if results are {0, 1}. If results are [0, 1], then we binarize
            is_binary = np.all(np.logical_or(scores == 0, scores == 1))
            if not is_binary and np.all((scores >= 0) & (scores <= 1)):
                scores = (scores > 0.5).astype(float) # binarize with threshold 0.5
                is_binary = True

            # If we cannot binarize scores, we cannot compute stats test
            if not is_binary:
                sig_results.loc['perc_sig', task_name] = float('-inf')
                all_p_values[task_name] = (mixes, scores, float('-inf'), float('-inf'))
                continue

        if aggregator == 'irt':
            raise NotImplementedError()
            sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/irt') # Add IRT code to PATH
            from irt_utils.irt_inference import load_irt_params

            train_instance_names, discriminations, difficulties = load_irt_params(
                load_path=Path(DATA_DIR) / "irt" / f"{task_name}.json",
            )
            irt_params = (difficulties, discriminations, train_instance_names)

            def compute_irt_f(test_instance_names, test_scores):
                return compute_irt(irt_params, test_instance_names, test_scores, metric)

            num_permutations = 10

            print('Computing IRT permutation test...')
            p_values, mix_scores, _ = compute_weighted_pairwise_p_values(
                scores, instance_names=instance_names, aggregator=compute_irt_f, 
                num_permutations=num_permutations, return_scores=True
            )
            print('Done!')

            p_values[np.tril_indices_from(p_values, k=-1)] = np.nan # set tril to nan
        elif isinstance(task, list) and aggregator == 'macro':
            raise NotImplementedError()
            # Get value counts for each task
            slices = get_slice(df, model=models, task=task)
            unique_counts = slices.groupby('task')['native_id'].nunique()
            weights = create_stratified_array(unique_counts)

            # Compute paired permutation test with instance weights
            p_values, mix_scores, _ = compute_weighted_pairwise_p_values(scores, num_permutations=num_permutations, weights=weights, return_scores=True)

            # Reorder both rows and columns of p_values
            sorted_indices = np.argsort(mix_scores)[::-1]
            p_values       = p_values[np.ix_(sorted_indices, sorted_indices)]
            mixes          = np.array(mixes)[sorted_indices]
            mix_scores     = mix_scores[sorted_indices]
            p_values[np.tril_indices_from(p_values, k=-1)] = np.nan
        elif isinstance(task, str) or aggregator == 'micro':
            raise NotImplementedError()
            # Micro average (default setup for single tasks)
            p_values, mix_scores, _ = compute_pairwise_p_values(scores, num_permutations=num_permutations, return_scores=True)
            
            # p_values, mix_scores, _ = compute_pairwise_p_values_paired_t_test(scores, return_scores=True)
            # p_values, mix_scores, _ = compute_pairwise_mcnemar(scores, return_scores=True)

            # p_values = np.nan_to_num(compute_pairwise_p_values(scores), nan=0) + np.nan_to_num(compute_pairwise_p_values(scores[::-1]).T, nan=0)
            # np.fill_diagonal(p_values, np.nan)
        else:
            raise ValueError(aggregator)

        sig_clusters = None
        if plot_sig_clusters:
            sig_clusters = get_sig_clusters(p_values, alpha=alpha)

        perc_sig = perc_significant(p_values, alpha=alpha)
        sig_results.loc['perc_sig', task_name] = perc_sig
        all_p_values[task_name] = (mixes, scores, p_values, sig_clusters)

        if do_plot is not None: 
            if pretty_mix_names is not None:
                mix_names = [pretty_mix_names[mix] for mix in mixes]
            else:
                mix_names = mixes
            axes[i] = plot_heatmap(axes[i], p_values, mix_names, mix_scores, sig_clusters, alpha=alpha, plot_clean=plot_clean, use_sig_colors=plot_clean)
            title = r'$p$' + f'-values for {task_name} (n={scores.shape[1]}) across data mixes at {("last " + str(last_n) + " steps" if last_n > 1 else "final checkpoint")} ({metric}), perc sig={(perc_sig*100):.2f}%'
            if len(models) < 15:
                title = r'$p$' + f'-values for {task_name}, perc sig={(perc_sig*100):.2f}%'
            axes[i].set_title(title, fontsize=10)

    if do_plot is not None: 
        if isinstance(do_plot, plt.Figure):
            fig.tight_layout()
        return sig_results, all_p_values, axes
    return sig_results, all_p_values, None

def compute_agreement(
    df, models, metric, tasks=None, 
    do_plot=False, pretty_mix_names=None, plot_clean=False, quiet=False):
    if tasks is None: 
        tasks = df.index.get_level_values('task').unique()

    agree_results = pd.DataFrame(index=['avg_agreement'], columns=tasks)
    all_p_values = {}

    n_tasks = len(tasks)
    if do_plot is not None: 
        if isinstance(do_plot, plt.Axes):
            axes = [do_plot] # allow passing in an axes object for plotting
        elif isinstance(do_plot, bool):
            if do_plot:
                fig, axes = plt.subplots(n_tasks, 1, figsize=(0.5*len(models), 0.4*len(models)*n_tasks))
                if n_tasks == 1: axes = [axes]
            else:
                do_plot = None
        else:
            axes = do_plot

    for i, task in tqdm(enumerate(tasks), desc='Computing pairwise comparisons', total=len(tasks), disable=quiet):
        task_name = get_title_from_task(task)
        
        instance_names, mixes, scores = get_nd_array(
            df, 'mix', metric, model=models, task=task, 
            step='max', sorted=False, return_index=True
        )

        # Sort by overall performance
        mix_sums = scores.sum(axis=1)
        sorted_indices = mix_sums.argsort()[::-1]
        mixes = np.array(mixes)[sorted_indices].tolist()
        scores = scores[sorted_indices]

        non_nan_scores = scores[~np.isnan(scores)]
        assert np.all(np.isin(non_nan_scores, [0, 1])), "Can only calculate agreemeent rate on binary scores"

        # Compute pairwise agreement rates between models
        n_models = len(mixes)
        agreement_rates = np.zeros((n_models, n_models))
        for i_ in range(n_models):
            for j_ in range(n_models):
                if i_ == j_:
                    agreement_rates[i_,j_] = 1.0
                else:
                    agreement = np.mean(scores[i_] == scores[j_])
                    agreement_rates[i_,j_] = agreement
                    agreement_rates[j_,i_] = agreement

        # Set lower triangle to nan to match p-value matrix format
        agreement_rates[np.tril_indices_from(agreement_rates, k=-1)] = np.nan

        avg_agreement = np.nanmean(agreement_rates)
        agree_results.loc['avg_agreement', task_name] = avg_agreement
        all_p_values[task_name] = (mixes, scores, agreement_rates)

        if do_plot is not None:
            if pretty_mix_names is not None:
                mix_names = [pretty_mix_names[mix] for mix in mixes]
            else:
                mix_names = mixes

            axes[i] = plot_heatmap(
                axes[i], agreement_rates, mix_names, mix_sums, 
                sig_clusters=None, alpha=0, plot_clean=plot_clean
            )
            title = f'Agreement rates for {task_name} (n={scores.shape[1]}) across data mixes ({metric}), avg agreement={(avg_agreement*100):.2f}%'
            if len(models) < 15:
                title = f'Agreement rates for {task_name}, avg agreement={(avg_agreement*100):.2f}%'
            axes[i].set_title(title, fontsize=10)

    if do_plot is not None:
        if isinstance(do_plot, plt.Figure):
            fig.tight_layout()
        return agree_results, all_p_values, axes
    return agree_results, all_p_values, None


def calculate_and_plot_total_variation(
        x, y, metric, 
        norm=True, improvement=True,
        model_name=None, num_scores=None, title=None, color=None, ax=None, add_text=True
    ):
    # Sort by x
    x, y = np.array(x), np.array(y)
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    
    tv = calc_total_variation(y, improvement=True, norm=True) * 100
    monotonicity = calc_monotonicity(y) * 100
    late_improvement = calc_improvement(y[int(len(y)*0.1):]) * 100 * 100

    # Add analytical CI
    ci = None
    # if num_scores is not None:
    #     ci = 1.96 * calculate_standard_error(y, num_scores=num_scores)

    if ax is not None and len(x) > 0:
        _ = plot_training(
            ax=ax, 
            label=model_name,
            x=x, y=y, ci=ci,
            xlabel='step', ylabel=metric, 
            title=title, color=color
        )

        if add_text:
            # Add total variation text
            text = ''
            text += f'\nTV-I={tv:.3f}'
            text = text.lstrip('\n')
            if text != '':
                ax.text(
                    x=x[-1], y=y[-1], s=text, color=color, 
                    va='center', ha='left', zorder=5, fontsize=10
                )
            
                if metric != 'c4_loss' and metric != 'll_per_char': 
                    ax.set_xlim(right=max(x) * 1.25)

            if metric == 'logits_per_byte':
                ax.set_ylim(top=max(y[int(len(y)*0.1):]), bottom=min(y)*0.95)

            # Add monotonicity text
            text = f'Monotonicity={monotonicity:.2f}%'
            ax.text(0.98, 0.02, text, transform=ax.transAxes, 
                    verticalalignment='bottom', horizontalalignment='right', fontsize=8)

            if 'logits' not in metric:
                # Add improvement text
                text = f'Improvement after 20% of steps={late_improvement:.2f}%'
                ax.text(0.98, 0.09, text, transform=ax.transAxes, 
                        verticalalignment='bottom', horizontalalignment='right', fontsize=8)

    return tv


def compute_total_variation(df, tasks, models, metric='acc_per_char', axes=None, color=None, add_text=True):
    if isinstance(axes, list) and axes[0] is None: axes = None
    
    tv_results = pd.DataFrame(index=['total_variation'], columns=tasks)

    assert isinstance(models, list) 

    for i, task in enumerate(tasks):
        for j, model in enumerate(models):
            if metric == 'logits_per_char' or metric == 'logits_per_byte':
                # TMP: map correct choice to metric
                step, bpb  = get_nd_array(df, 'step', metric, model=model, task=task)
                _, corr = get_nd_array(df, 'step', 'correct_choice', model=model, task=task)

                from ladder_wrapper import map_corr_labels
                correct_bpb = map_corr_labels(bpb, corr, task_name=task)
                acc = correct_bpb.mean(axis=1)
                scores = correct_bpb
            else:
                # step, scores = get_nd_array(df, 'step', metric, model=model, task=task)
                step, scores = get_nd_array(df, ['task', 'step'], metric, model=model, task=task)
                
                if scores.ndim > 1:
                    # Average all dims except dim 1
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        with np.errstate(invalid='ignore', divide='ignore'):
                            acc = np.nanmean(scores, axis=tuple(range(1, scores.ndim)))
                else:
                    acc = scores

            task_name = task
            if isinstance(task, list):
                task_name = 'aggregate'

            num_scores = scores.shape[1] if scores.ndim == 2 else None

            # Remove the NaN entries
            step = np.array(step, dtype=np.float64)
            acc = np.array(acc, dtype=np.float64)
            mask = ~np.isnan(acc)
            step = step[mask]
            acc = acc[mask]

            tv_results.loc['total_variation', task_name] = calculate_and_plot_total_variation(
                x=step,
                y=acc,
                metric=metric,
                model_name=model,
                num_scores=num_scores,
                # color=(color[j] if isinstance(color, list) else None),
                color=(color[j] if isinstance(color, list) else color),
                title=f'{task_name} (n={num_scores}) {"on " + models if len(models) == 0 else ""}',
                ax=axes[i] if axes is not None else None,
                add_text=add_text
            )

            tv_results.loc['total_variation:no_norm', task_name] = calculate_and_plot_total_variation(
                x=step,
                y=acc,
                metric=metric,
                norm=False
            )

            # Compute std
            sorted_indices = np.argsort(step)
            step = step[sorted_indices]
            acc = acc[sorted_indices]

            # Get last 20% and last 10 checkpoints
            n_20_percent = int(len(step) * 0.2)
            n_10 = min(10, len(step))  # Take last 10 checkpoints or all if less than 10

            # Calculate std and relative std for last 20%
            last_20_std = np.std(acc[-n_20_percent:])
            last_20_mean = np.mean(acc[-n_20_percent:])
            last_20_rel_std = last_20_std / abs(last_20_mean) if last_20_mean != 0 else np.nan

            # Calculate std and relative std for last 10 checkpoints
            last_10_std = np.std(acc[-n_10:])
            last_10_mean = np.mean(acc[-n_10:])
            last_10_rel_std = last_10_std / abs(last_10_mean) if last_10_mean != 0 else np.nan

            # Calculate std and relative std for last 30 checkpoints
            n_30 = min(30, len(step))  # Take last 30 checkpoints or all if less than 30
            last_30_std = np.std(acc[-n_30:])
            last_30_mean = np.mean(acc[-n_30:])
            last_30_rel_std = last_30_std / abs(last_30_mean) if last_30_mean != 0 else np.nan

            tv_results.loc['step_std:perc20', task_name] = last_20_std
            tv_results.loc['step_rel_std:perc20', task_name] = last_20_rel_std
            tv_results.loc['step_std:last10', task_name] = last_10_std
            tv_results.loc['step_rel_std:last10', task_name] = last_10_rel_std
            tv_results.loc['step_std:last30', task_name] = last_30_std
            tv_results.loc['step_rel_std:last30', task_name] = last_30_rel_std
        
        if axes is not None and axes[i].get_legend_handles_labels()[1]:
            axes[i].legend(fontsize=8)

    return tv_results, axes