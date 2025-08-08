import pandas as pd
import numpy as np
import warnings

from snr.plot import plot_training
from snr.dataloader import get_nd_array


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


def calculate_standard_error(avg_score, num_scores):
    """ https://arxiv.org/pdf/2411.00640#page=2.55 """
    return np.sqrt((avg_score * (1 - avg_score)) / num_scores)


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