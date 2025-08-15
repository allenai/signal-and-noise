import warnings
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from snr.dataloader import get_nd_array, get_slice
from snr.ladder_wrapper import run_ladder
from snr.constants import ROOT_DIR, get_pretty_task_name, get_title_from_task
from snr.constants.plot import MODEL_FAMILY_COLORS
from snr.constants.datadecide import DATADECIDE_SIZES, DATADECIDE_COMPUTE
plt.close()

LADDER_CONFIG_PATH = f'{ROOT_DIR}/snr/constants/ladder_config.json'

def _plt_stacked_predictions_clean(ax: plt.Axes, df, ladder_models, task):
    ## Stacked ladder prediction
    (_, _, stacked_error), (_, _, stacked_y), (_, _, stacked_y_pred) = run_ladder(
        df,
        task,
        train_models=ladder_models,
        eval_models=["peteish13-highlr"],
        config_path=LADDER_CONFIG_PATH,
        plot_compute=True,
        run_step1=False, run_step2=False,
        axes=[ax],
        return_reals=True
    )
    
    ax.get_legend().remove()

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='grey', label='Scaling Law Models', linestyle='None'),
        Line2D([0], [0], marker='x', color='#0A3234', label='Predicted 13B Model', linestyle='None'),
        Line2D([0], [0], marker='o', color='#F0519C', label='Real 13B Model', linestyle='None'),
        Line2D([0], [0], color='#F0519C', label='Scaling Law Fit', linestyle='-'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='lower right')

    stacked_y_pred = stacked_y_pred[0]

    return stacked_error, stacked_y, stacked_y_pred


def _plt_intermediate_checkpoints_13b(ax: plt.Axes, df, task, metric='primary_score'):
    ### intermediate checkpoints
    step, scores = get_nd_array(df, ['task', 'step'], metric, model='peteish13-highlr', task=task)

    if len(scores.shape) > 1:
        scores = np.mean(scores, axis=-1)

    step = np.array(step, dtype=np.float64)
    scores = np.array(scores, dtype=np.float64)

    compute = (step / max(step)) * 6*13202396160*5000088518656

    x = compute[3:]
    y = scores[3:]
    
    x, y = np.array(x), np.array(y)
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    ax.plot(x, y, color='#0A3234', linewidth=0.25, marker='.', markersize=1)

    final_scores = y[-30:]
    return final_scores


def _plot_with_inset(ax: plt.Axes, df, ladder_models, task, task_y, task_title):
    _plt_stacked_predictions_clean(ax, df, ladder_models, task)
    _plt_intermediate_checkpoints_13b(ax, df, task)

    # Increase range by 10%
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + y_range * 0.1)

    axins = ax.inset_axes([0.03, 0.57, 0.5, 0.4])
    
    stacked_error, stacked_y, stacked_y_pred = _plt_stacked_predictions_clean(
        axins, 
        df, 
        ladder_models, 
        task
    )
    final_scores = _plt_intermediate_checkpoints_13b(axins, df, task)

    y1_final, y2_final = max(final_scores), min(final_scores)
    mid_final = (y1_final + y2_final) / 2

    y1, y2 = stacked_y, stacked_y_pred
    mid = (y1 + y2) / 2
    
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    zoom_x_min = x_min + (x_max - x_min) * 0.5 # 0.3
    y_range = (y_max - y_min) * 0.15
    axins.set_xlim(zoom_x_min, x_max*0.6) # 0.45
    axins.set_ylim(mid-y_range, mid+y_range)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.tick_params(axis='x', which='both', labelbottom=False)
    
    ax.indicate_inset_zoom(axins, edgecolor="black")

    ### Add bracket

    x_pos = axins.get_xlim()[1] * 0.91 # 0.5
    
    axins.plot([x_pos, x_pos], [y1, y2], 'k-', linewidth=0.5)
    axins.plot([x_pos-x_pos*0.0025, x_pos], [y1, y1], 'k-', linewidth=0.5)  # Top horizontal line
    axins.plot([x_pos-x_pos*0.0025, x_pos], [y2, y2], 'k-', linewidth=0.5)  # Bottom horizontal line
    
    error_pct = abs(y2 - y1) / y1 * 100
    axins.text(x_pos+x_pos*0.005, mid, f'error\n{error_pct:.1f}%', 
                verticalalignment='center', fontsize=8, color='r')
    
    # Add margin of error bracket
    x_pos2 = axins.get_xlim()[1] * 0.95

    axins.plot([x_pos2, x_pos2], [y1_final, y2_final], 'k-', linewidth=0.5)
    axins.plot([x_pos2-x_pos2*0.0025, x_pos2], [y1_final, y1_final], 'k-', linewidth=0.5)
    axins.plot([x_pos2-x_pos2*0.0025, x_pos2], [y2_final, y2_final], 'k-', linewidth=0.5)
    
    error_pct_final = abs(y2_final - y1_final) / y1_final * 100
    axins.text(x_pos2+x_pos2*0.005, mid_final, 
               'noise', 
               # 'std.\ndev.', 
               # f'margin\nof\nerror', 
               verticalalignment='center', fontsize=8, color='k')

    axins.set_title("")
    axins.set_xlabel("")
    axins.set_ylabel("")
    axins.get_legend().remove() if axins.get_legend() is not None else None

    ax.set_title(task_title, fontsize=14)
    ax.set_xlabel('Compute (FLOPs)', fontsize=12)
    ax.set_ylabel(task_y, fontsize=12)

def format_func(x, p): return f'{int(x/1000)}K'

def _plt_intermediate_checkpoints_1b(ax: plt.Axes, df, task, metric, show_legend=False):
    ### Plot training curve
    step, scores = get_nd_array(
        df, ['task', 'step'], metric, model='peteish-moreeval-1B-5xC', task=task
    )
    if scores.ndim > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(invalid='ignore', divide='ignore'):
                acc = np.nanmean(scores, axis=tuple(range(1, scores.ndim)))
    else:
        acc = scores
    
    step = np.array(step, dtype=np.float64)
    acc  = np.array(acc, dtype=np.float64)
    mask = ~np.isnan(acc)
    step = step[mask]
    acc  = acc[mask]
    
    ax.plot(step, acc, label='Training curve\n(1B model)', linewidth=0.7, color='#0A3234') # '1B-5xC\ntraining curve'

    ### Plot DataDecide models
    datadecide = set(df[df['model_type'] == 'datadecide']['model'])
    datadecide_1b = [model for model in datadecide if '1B' in model]

    _, scores = get_nd_array(
        df, ['task', 'step', 'model'], metric, model=datadecide_1b, task=task, step='max'
    )
    if scores.ndim > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(invalid='ignore', divide='ignore'):
                final_scores = np.nanmean(scores, axis=tuple(range(1, scores.ndim)))
    else:
        final_scores = scores
    final_scores = final_scores[~np.isnan(final_scores)]

    ax.plot([step[-1]] * len(final_scores), final_scores, 'x', color='#F0519C', markersize=4, label='Final checkpoints\n(DataDecide 1B)', alpha=0.5)

    # Compute SNR
    signal = np.std(final_scores[-30:]) / np.mean(final_scores[-30:])
    noise  = np.std(acc[-30:]) / np.mean(acc[-30:])
    snr = signal / noise

    ax.text(0.03, 0.97, f'Signal-to-noise ratio={snr:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

    # Increase ylim
    y_min, y_max = min(final_scores), max(final_scores)
    padding = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_xlabel('Training Step (100B tokens)')
    if show_legend == 'two_col':
        ax.legend(loc='lower center', bbox_to_anchor=(1.15, -0.7), ncols=2, fontsize=8)
    elif show_legend == 'one_col':
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.9), ncols=1, fontsize=10)

    return snr

def plot_task_observational(ax, task, df, observational_models, datadecide, models_in_range, add_region_shading=True):
    data_7b = get_slice(df, model=observational_models + datadecide, task=task)

    # Get colors based on model family
    colors = []
    for model in data_7b['model']:
        color = None
        for family in MODEL_FAMILY_COLORS:
            if family in model.lower():
                color = MODEL_FAMILY_COLORS[family]
                # color = "#F0519C"
                break
        colors.append(color if color else 'grey')
        # colors.append(color if color else '#0A3234')

    # Plot DataDecide models with triangles and others with x's
    dd_mask = data_7b['model'].isin(datadecide)
    ax.scatter(data_7b[dd_mask]['flops'], data_7b[dd_mask]['primary_score'], 
              c=[colors[i] for i, m in enumerate(data_7b['model']) if m in datadecide],
              marker='^', alpha=0.5)
    ax.scatter(data_7b[~dd_mask]['flops'], data_7b[~dd_mask]['primary_score'],
              c=[colors[i] for i, m in enumerate(data_7b['model']) if m not in datadecide], 
              marker='x', alpha=0.8)
    
    # # Add text labels for each model point
    # texts = []
    # all_models = set()
    # for i, row in data_7b.iterrows():
    #     model_name = row['model']
    #     if model_name in all_models:
    #         continue
    #     # if len(all_models) > 30:
    #     #     continue
    #     if row['flops'] < 10**18:
    #         continue
    #     if model_name not in datadecide:
    #         texts.append(ax.text(
    #             row['flops'], row['primary_score'], model_name, fontsize=6
    #         ))
    #         all_models.add(model_name)

    # Add compute ranges
    labels = ['1.5B-4T', '7B-4T', '13B-5T', '32B-6T']
    for (model, info), label in zip(models_in_range.items(), labels):
        target = info['flops_target']
        lower, upper = info['flops_range']
        
        # Plot vertical dotted line at target
        ax.axvline(x=target, color='black', linestyle=':', alpha=0.5)
        
        if add_region_shading:
            # Add shaded region for range
            ymin, ymax = ax.get_ylim()
            ax.axvspan(lower, upper, alpha=0.1, color='gray')
        ax.text(target, ax.get_ylim()[1]+0.005, label, horizontalalignment='left', verticalalignment='bottom', fontsize=8, rotation=35)

    for size, compute in zip(DATADECIDE_SIZES, DATADECIDE_COMPUTE):
        ax.axvline(x=compute, color='gray', linestyle=':', alpha=0.5)

        # Extract params from size and compute tokens
        if 'M' in size:
            params = float(size.replace('M','')) * 1e6
        elif 'B' in size:
            params = float(size.replace('B','')) * 1e9
        toks = f'{int(params * 100/1e9)}B'
        
        if size == '4M' or size == '20M' or size == '750M':
            continue

        ax.text(compute, ax.get_ylim()[1]+0.005, f'{size}-{toks}', 
                horizontalalignment='left', verticalalignment='bottom', 
                fontsize=8, rotation=35)

    # adjustText(ax, texts)

    ax.set_xscale('log')
    ax.set_xlabel('Compute (Est. FLOPs)')
    ax.set_title(get_pretty_task_name(task), pad=40)

def macro_avg(_slice, task_set):
    new_task = _slice.fillna('')
    new_task = new_task.groupby(['model', 'step', 'mix', 'size'])[['model_config', 'primary_score', 'logits_per_byte_corr', 'logits_per_char_corr']]
    new_task = new_task.agg(lambda x: x.iloc[0] if x.name == 'model_config' else x[pd.to_numeric(x, errors='coerce').notnull()].mean())
    new_task = new_task.reset_index()
    new_task['step'] = pd.to_numeric(new_task['step'], errors='coerce') 
    new_task['task'] = get_title_from_task(task_set)
    return new_task


def plot_training_progress(task, ax: plt.Axes, df, type='line', quiet=True, plot_merged=True):
    """Plot training progress for a single task across model sizes."""
    # Get the data slices for this task
    data_1b = get_slice(df, model='peteish1', task=task)
    data_7b = get_slice(df, model='peteish7', task=task)
    data_13b = get_slice(df, model='peteish13-highlr', task=task) 
    data_32b = get_slice(df, model='peteish32', task=task)

    if isinstance(task, list):
        data_1b = macro_avg(data_1b, task)
        data_7b = macro_avg(data_7b, task)
        data_13b = macro_avg(data_13b, task)
        data_32b = macro_avg(data_32b, task)

    # Sort by steps
    data_1b = data_1b.sort_values('step')
    data_7b = data_7b.sort_values('step')
    data_13b = data_13b.sort_values('step')
    data_32b = data_32b.sort_values('step')

    # Keep only last 30 steps of 13B data
    data_13b = data_13b.iloc[-30:]

    # Compute relative standard deviation (CV) for each dataset
    data_1b_cv = data_1b['primary_score'].std() / data_1b['primary_score'].mean()
    data_7b_cv = data_7b['primary_score'].std() / data_7b['primary_score'].mean()
    data_13b_cv = data_13b['primary_score'].std() / data_13b['primary_score'].mean()
    data_32b_cv = data_32b['primary_score'].std() / data_32b['primary_score'].mean()

    if ax is None:
        return data_32b_cv

    if not quiet:
        print(f"\nRelative std dev (CV) for {task}:")
        print(f"1B: {data_1b_cv:.3f}")
        print(f"7B: {data_7b_cv:.3f}")
        print(f"13B: {data_13b_cv:.3f}")
        print(f"32B: {data_32b_cv:.3f}")

    if type == 'line':
        try:
            # Plot on the corresponding subplot
            ax.plot(range(len(data_1b['step'])), data_1b['primary_score'], linewidth=1, label='7B', color='orange')
            ax.text(
                len(data_1b['step'])-1, data_1b['primary_score'].iloc[-1], 
                f'{data_1b_cv:.3f}', verticalalignment='center', color='r'
            )

            ax.plot(range(len(data_7b['step'])), data_7b['primary_score'], linewidth=1, label='7B', color='orange')
            ax.text(
                len(data_7b['step'])-1, data_7b['primary_score'].iloc[-1], 
                f'{data_7b_cv:.3f}', verticalalignment='center', color='orange'
            )

            ax.plot(range(len(data_13b['step'])), data_13b['primary_score'], linewidth=1, label='13B', color='b')
            ax.text(
                len(data_13b['step'])-1, data_13b['primary_score'].iloc[-1], 
                f'{data_13b_cv:.3f}', verticalalignment='center', color='b'
            )

            ax.plot(range(len(data_32b['step'])), data_32b['primary_score'], linewidth=1, label='32B', color='g')
            ax.text(
                len(data_32b['step'])-1, data_32b['primary_score'].iloc[-1], 
                f'{data_32b_cv:.3f}', verticalalignment='center', color='g'
            )
        except Exception as e:
            print(e)

        ax.set_xlabel('Final 30K Training Steps')
        ax.set_ylabel('Primary Score')

        ax.set_xticks([0, 10, 20, 30], labels=['0K', '10K', '20K', '30K'])
        ax.set_xlim([0, 40])

    elif type == 'histogram':
        # Plot histograms for each model size
        ax.hist(data_7b['primary_score'], bins=15, alpha=0.3, label='7B last 30 ckpts', color='orange')
        ax.hist(data_13b['primary_score'], bins=15, alpha=0.3, label='13B last 30 ckpts', color='b')
        ax.hist(data_32b['primary_score'], bins=15, alpha=0.3, label='32B last 30 ckpts', color='g')
        ax.set_xlabel('Primary Score')
        ax.set_ylabel('Count')

    ax.set_title(get_pretty_task_name(task))
    ax.grid(True, linestyle='--', alpha=0.3)

    return data_32b_cv