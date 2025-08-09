import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy import stats
from scipy.interpolate import UnivariateSpline

from snr.dataloader import get_slice
from snr.constants import get_title_from_task, get_pretty_task_name
from snr.constants.plot import CATEGORY_COLORS, SIZE_COLORS, TASK_CATEGORIES

# Global dictionary to store colors for labels
LABEL_COLOR_MAP = {}
COLOR_IDX = {'col': 0}

def get_valid_points(df_results, x_col, y_col, z_col=None):
    """ Helper function to get valid points from rows in a df """
    points = []
    for task in df_results.index:
        x = df_results[x_col][task]
        y = df_results[y_col][task]
        z = None
        if isinstance(z_col, str) and z_col in df_results.columns:
            z = df_results[z_col][task]
        if x != float('nan') and y != float('nan') and not callable(x) and not callable(y):
            points.append((x, y, z, task))
    return points


def adjustText(ax, texts):
    """ Adjust text annotations in matplot figure to not overlap with each other """
    if len(texts) > 0:
        import matplotlib

        existing_annotations = [
            child for child in ax.get_children() if isinstance(child, matplotlib.text.Annotation)
        ]

        # Remove existing annotation
        for child in existing_annotations:
            child.remove()

        from adjustText import adjust_text

        adjust_text(
            texts,
            arrowprops=dict(
                arrowstyle="-", 
                color="gray", 
                lw=0.5, 
                alpha=0.5,
                clip_on=True  # Enable clipping for arrows
            ),
            avoid_points=True,
            avoid_self=True,
            avoid_lines=True,
            existing_annotations=existing_annotations,
            autoalign="xy",
            force_points=0.5,
            force_text=0.2,
            expand_points=(1.5, 1.5),
            ax=ax,
        )

        # # Set clip_on for all annotation objects after adjustment
        # for text in texts:
        #     text.set_clip_on(True)
        #     if hasattr(text, 'arrow_patch') and text.arrow_patch:
        #         text.arrow_patch.set_clip_on(True)


def draw_pareto_frontier(ax, xs, ys, invert_x=False, invert_y=False, color='grey', linestyle='--'):
    """Draw Pareto frontier lines on the given axes"""
    points = list(zip(xs, ys))
    frontier_points_y = set()
    frontier_points_x = set()
    
    # Find points that are optimal in x dimension (scan downward on x dim)
    sorted_by_x = sorted(points, reverse=not invert_x, key=lambda p: p[0])
    max_y = float('-inf') if not invert_y else float('inf')
    
    for x, y in sorted_by_x:
        if (y > max_y and not invert_y) or (y < max_y and invert_y):
            frontier_points_y.add((x, y))
            max_y = y
        elif y == max_y:
            frontier_points_y.add((x, y))
            
    # Convert to list and sort for drawing
    frontier_points_y = sorted(list(frontier_points_y), key=lambda p: p[0], reverse=invert_x)
    frontier_points_x = sorted(list(frontier_points_x), key=lambda p: p[1], reverse=invert_y)
    
    # Draw dotted grey line connecting frontier points
    if frontier_points_y:
        frontier_xs, frontier_ys = zip(*frontier_points_y)
        ax.plot(frontier_xs, frontier_ys, color=color, linestyle=linestyle, linewidth=1)
    if frontier_points_x:
        frontier_xs, frontier_ys = zip(*frontier_points_x)
        ax.plot(frontier_xs, frontier_ys, color=color, linestyle=linestyle, linewidth=1)
    if len(frontier_points_x) > 0 and len(frontier_points_y) > 0:
        # Connect the ends of both frontiers
        frontier_points_y_end = frontier_points_y[-1]
        frontier_points_x_end = frontier_points_x[-1]
        ax.plot([frontier_points_y_end[0], frontier_points_x_end[0]], 
                [frontier_points_y_end[1], frontier_points_x_end[1]], 
                color=color, linestyle=linestyle, linewidth=1)


def plot_task_scatter(
    ax: plt.Axes, df, x_col, y_col, xlabel, ylabel, title=None, labeled_tasks=None,
    category=None, percentage=False, threshold=None,
    invert_x=False, invert_y=False, log_x=False, log_y=False, xlim=None, ylim=None, x_col_b=None, y_col_b=None,
    xdesc=None, ydesc=None, draw_frontier=True, compute_fit=False, color=None, zlabel=None, invert_z=False,
    ):
    points = get_valid_points(df, x_col, y_col, z_col=color)
    if not points:
        return
    
    # Filter out points not in the specified task category (e.g., math)
    if category is not None:
        category = [category] if not isinstance(category, list) else category
        task_categories = [TASK_CATEGORIES.get(task, 'knowledge') for _, _, _, task in points]
        points = [p for p, cat in zip(points, task_categories) if cat in category]
        if not points:
            return
    
    # points = points[:-1]
    xs, ys, zs, tasks = zip(*points)
    
    # Filter out -inf values if needed
    if log_x or log_y:
        valid_indices = [i for i in range(len(xs)) if xs[i] != float('-inf') and ys[i] != float('-inf')]
        xs = [xs[i] for i in valid_indices]
        ys = [ys[i] for i in valid_indices]
        tasks = [tasks[i] for i in valid_indices]
    
    if color is None:
        colors = [CATEGORY_COLORS[TASK_CATEGORIES.get(task, 'knowledge')] for task in tasks]
    elif zs is not None and any(z is not None for z in zs):
        color_values = zs
        norm = plt.Normalize(vmin=min(color_values), vmax=max(color_values))
        colors = plt.cm.viridis_r(norm(color_values)) if invert_z else plt.cm.viridis(norm(color_values))
    else:
        colors = [color for _ in tasks]
    
    # If diff mode (both x_col_b and y_col_b provided)
    if x_col_b is not None and y_col_b is not None:
        points_b = get_valid_points(df, x_col_b, y_col_b)
        if points_b:
            xs_b, ys_b, _, tasks_b = zip(*points_b)
            
            # Only keep points that exist in both sets
            common_tasks = set(tasks).intersection(tasks_b)
            xs = [x for x, t in zip(xs, tasks) if t in common_tasks]
            ys = [y for y, t in zip(ys, tasks) if t in common_tasks]
            xs_b = [x for x, t in zip(xs_b, tasks_b) if t in common_tasks]
            ys_b = [y for y, t in zip(ys_b, tasks_b) if t in common_tasks]
            colors = [c for c, t in zip(colors, tasks) if t in common_tasks]
            tasks = [t for t in tasks if t in common_tasks]

            if category is None:
                colors_a = colors_b = line_colors = colors
            else:
                colors_a = 'r'
                colors_b = 'g'
                line_colors = ['k' for _ in colors]
            
            # Draw arrows between corresponding points
            for i, (x, y, x_b, y_b) in enumerate(zip(xs, ys, xs_b, ys_b)):
                # ax.arrow(x, y, x_b-x, y_b-y, color=colors[i], length_includes_head=True, alpha=0.2)
                ax.plot([x, x_b], [y, y_b], color=line_colors[i], alpha=0.2, linewidth=0.5)
            
            # Plot both sets of points
            ax.scatter(xs, ys, s=4, c=colors_a, marker='o')
            ax.scatter(xs_b, ys_b, s=4, c=colors_b, marker='s')

        # Draw separate Pareto frontiers for before and after points
        if draw_frontier:
            for category_name in set(TASK_CATEGORIES.values()):
                # Get before points for this category
                category_points = [(x, y) for x, y, task in zip(xs, ys, tasks) if TASK_CATEGORIES.get(task, 'knowledge') == category]
                if category_points:
                    cat_xs, cat_ys = zip(*category_points)
                    color = CATEGORY_COLORS[category_name] if category is None else 'r'
                    draw_pareto_frontier(ax, cat_xs, cat_ys, invert_x=invert_x, invert_y=invert_y, color=color, linestyle=':')
                
                # Get after points for this category
                category_points_b = [(x, y) for x, y, task in zip(xs_b, ys_b, tasks) if TASK_CATEGORIES.get(task, 'knowledge') == category]
                if category_points_b:
                    cat_xs_b, cat_ys_b = zip(*category_points_b)
                    color = CATEGORY_COLORS[category_name] if category is None else 'g'
                    draw_pareto_frontier(ax, cat_xs_b, cat_ys_b, invert_x=invert_x, invert_y=invert_y, color=color, linestyle='--')
    else:
        # Regular scatter plot
        ax.scatter(xs, ys, s=8, c=colors) # s=4

        if draw_frontier:
            # Draw separate Pareto frontiers for each task category
            for category_name in set(TASK_CATEGORIES.values()):
                # Get points for this category
                category_points = [(x, y) for x, y, task in zip(xs, ys, tasks) if TASK_CATEGORIES.get(task, 'knowledge') == category_name]
                if category_points:
                    cat_xs, cat_ys = zip(*category_points)
                    draw_pareto_frontier(ax, cat_xs, cat_ys, invert_x=invert_x, invert_y=invert_y, color=(CATEGORY_COLORS[category_name] if color is None else color))

        #### Add line of best fit here
        if compute_fit:
            # Add line of best fit with confidence interval
            x_log = np.log10(xs)
            y_log = np.log10(ys)
            z = np.polyfit(x_log, y_log, 1)
            p = np.poly1d(z)
            x_line = np.logspace(np.log10(min(xs)), np.log10(max(xs)), 100)
            y_line = 10**p(np.log10(x_line))
            
            n = len(xs)
            y_mean = np.mean(y_log)
            x_mean = np.mean(x_log)
            s_err = np.sqrt(np.sum((y_log - p(x_log))**2)/(n-2))
            x_new = np.log10(x_line)
            conf = stats.t.ppf(0.975, n-2) * s_err * np.sqrt(1/n + (x_new - x_mean)**2 / np.sum((x_log - x_mean)**2))
            
            r = np.corrcoef(x_log, y_log)[0,1]
            r2 = r**2
            stderr = s_err * np.sqrt((1-r2)/(n-2))
            if 'SNR' in xlabel or 'Scaling' in xlabel:
                background = None
            else:
                background = dict(facecolor='white', alpha=0.7, edgecolor='none')

            v_adjust = 0.03 if xdesc is not None else 0

            ax.text(0.03, 0.97-v_adjust, f'R = {r:.3f} ± {stderr:.3f}\nR² = {r2:.3f}', 
                    transform=ax.transAxes, fontsize=11, verticalalignment='top',
                    bbox=background)
        
        plot_fit = compute_fit # looks odd, i know
        if plot_fit:
            assert compute_fit
            ax.plot(x_line, y_line, '--', color='black', alpha=0.5)
            print(f"n={n}, s_err={s_err}, x_mean={x_mean}, x_log spread={np.ptp(x_log)}")
            # ax.fill_between(x_line, y_line-conf, y_line+conf, color='gray', alpha=0.2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    if xlim is not None:
        ax.set_xlim(**xlim)
    if ylim is not None:
        ax.set_ylim(**ylim)

    # if color is not None:
    if zs is not None and any(z is not None for z in zs):
        cmap = plt.cm.viridis if not invert_z else plt.cm.viridis_r
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=(zlabel if zlabel is not None else color))
        if invert_z: cbar.ax.invert_yaxis()

    if percentage:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        if log_x:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x) if x >= 0.01 else '{:.1%}'.format(x) if x >= 0.001 else '{:.2%}'.format(x)))
        if log_y:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y) if y >= 0.01 else '{:.1%}'.format(y) if y >= 0.001 else '{:.2%}'.format(y)))
    
    texts = []
    for x, y, task in zip(xs, ys, tasks):
        # Only add text if point is within axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if ((xlim[0] <= x <= xlim[1]) if not ax.xaxis_inverted() else (xlim[1] <= x <= xlim[0])) and \
           ((ylim[0] <= y <= ylim[1]) if not ax.yaxis_inverted() else (ylim[1] <= y <= ylim[0])):
            task_name = get_title_from_task(task)
            if labeled_tasks is None or task_name in labeled_tasks:
                text = ax.text(x, y, get_pretty_task_name(task_name), fontsize=(7 if labeled_tasks is None else 10), clip_on=True, ha='right')
                texts += [text] 
        
    adjustText(ax, texts)

    # Add axis description text if provided
    if xdesc is not None or ydesc is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        transform = ax.transData

        if xdesc is not None:
            # Add x-axis description text to bottom right
            x_pos = xlim[1]
            y_pos = ylim[0]
            display_coords = transform.transform((x_pos, y_pos))
            display_coords = (display_coords[0] - 5, display_coords[1] + 5)
            data_coords = transform.inverted().transform(display_coords)
            x_pos, y_pos = data_coords
            ax.text(x_pos, y_pos, 
                    # f'← {xdesc}',
                    f'{xdesc} →',
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=8,
                    weight='bold')

        if ydesc is not None:
            # Add y-axis description text to top left
            x_pos = xlim[0]
            y_pos = ylim[1]
            display_coords = transform.transform((x_pos, y_pos))
            display_coords = (display_coords[0] + 5, display_coords[1] - 5)
            data_coords = transform.inverted().transform(display_coords)
            x_pos, y_pos = data_coords
            ax.text(x_pos, y_pos, 
                    # f'↓ {ydesc}',
                    f'↑ {ydesc}',
                    horizontalalignment='left',
                    verticalalignment='top',
                    fontsize=8,
                    weight='bold')
    
    if threshold is not None:
        # Add shaded regions
        x_threshold = np.percentile(df[x_col].dropna(), threshold[0])
        y_threshold = np.percentile(df[y_col].dropna(), threshold[1])
        
        ax.axvspan(ax.get_xlim()[0], x_threshold, color='red', alpha=0.2)
        ax.axhspan(y_threshold, ax.get_ylim()[0], color='blue', alpha=0.2)
        # ax.axvspan(x_threshold, ax.get_xlim()[1], ymin=(ax.get_ylim()[0]-y_threshold)/abs(ax.get_ylim()[0]-ax.get_ylim()[1]), ymax=1, color='green', alpha=0.05)
        
        ax.text(0.02, 0.98, 'Too noisy', transform=ax.transAxes, ha='left', va='top', color='darkred', fontsize=8)
        ax.text(0.98, 0.02, 'Not correlated with\ndownstream task', transform=ax.transAxes, ha='right', va='bottom', color='darkblue', fontsize=8)
        ax.text(0.98, 0.98, 'Low risk & high correlation\nwith downstream task', transform=ax.transAxes, ha='right', va='top', color='darkgreen', fontsize=8)

    return ax


def assign_color(label):
    if label not in LABEL_COLOR_MAP:
        available_colors = list(mcolors.TABLEAU_COLORS.keys())
        assigned_color = available_colors[COLOR_IDX['col'] % len(available_colors)]
        LABEL_COLOR_MAP[label] = assigned_color
        COLOR_IDX['col'] += 1
    return LABEL_COLOR_MAP[label]


def lighten_color(color, amount=0.2):
    r, g, b = mcolors.to_rgb(color)
    new_r = min(r + (1 - r) * amount, 1)
    new_g = min(g + (1 - g) * amount, 1)
    new_b = min(b + (1 - b) * amount, 1)
    return new_r, new_g, new_b


def plot_training(ax: plt.Axes, x, y, xlabel: str, ylabel: str, label=None, title=None, color=None, fit=None, ci=None, sma_window=None):
    if color is None and label is not None:
        label_for_color = label
        # label_for_color = label.replace('_rc', '').replace('_mc', '').replace('_val', '').replace('_test', '') # peteish32 override
        # if '_5shot' in label_for_color: label_for_color = label_for_color.split('_5shot')[0]
        color = assign_color(label_for_color)
        # if 'rc' in label: 
        #     color = lighten_color(color, amount=0.5)

    if xlabel == 'step':
        if sma_window is not None:
            import numpy as np
            sma = np.convolve(y, np.ones(sma_window)/sma_window, mode='valid')
            x_sma = x[sma_window-1:]
            x_plt, y_plt = x_sma, sma
        else:
            x_plt, y_plt = x, y
        
        ax.plot(x_plt, y_plt, label=label, color=color, linewidth=0.5, marker='.', markersize=2)
        # ax.plot(df_slice[xlabel], df_slice[ylabel].rolling(window=5).mean(), label=label, color=color, linewidth=0.5, marker='.', markersize=2)
    else:
        ax.scatter(x, y, label=label, color=color, s=3)

    if ci is not None:
        ax.fill_between(x, y - ci, y + ci, alpha=0.1, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if label is not None: ax.legend()

    ax.tick_params(axis='both', which='major', labelsize=8)

    # add fitted log function
    if fit is not None:
        # ax.set_xscale('log')

        from scipy.optimize import curve_fit
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore", message="invalid value encountered in log")

        def log(x, a, b, c): return a * np.log(b * x) + c
        # def log(x, epsilon, k, gamma): return epsilon - k * np.exp(-gamma * x) # samir err

        # x, y = df_slice[xlabel].values, df_slice[ylabel].values
        x_max = np.max(x)
        x_scaled = x / x_max
        popt, _ = curve_fit(log, x_scaled, y, maxfev=10000)
        
        x_fit = np.linspace(min(x), max(x), 100)
        x_fit_scaled = x_fit / x_max
        y_fit = log(x_fit_scaled, *popt)

        ax.plot(x_fit, y_fit, color=color, alpha=0.5, linestyle='dotted')

    return ax

def setup_plot_grid(metric_names: List[str], plotted_tasks: List[str], num_rows: int) -> Tuple[plt.Figure, np.ndarray]:
    """Create and setup the plot grid"""
    num_cols = len(plotted_tasks)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4.5*num_cols, 2.5*num_rows))
    return fig, axes.flatten()

def plot_single_run(ax: plt.Axes, metric_data: pd.DataFrame, task_label: str, inset_only=False, noise_label='noise', inset_legend=True):
    """Plot single run with inset showing step noise"""
    first_run = metric_data['run_name'].unique()[0]
    run_data = metric_data[metric_data['run_name'] == first_run]
    
    if inset_only is False:
        # Main plot
        ax.plot(run_data['step'], run_data['value'], linewidth=0.5)
    
    # Inset
    # ax_inset = ax.inset_axes([0.5, 0.1, 0.45, 0.45])
    ax_inset = ax.inset_axes([0.5, 0.05, 0.45, 0.35])
    ax_inset.plot(run_data['step'], run_data['value'], linewidth=0.5)
    
    # Set inset zoom region
    x_range = run_data['step'].max() - run_data['step'].min()
    x_min_zoom = run_data['step'].max() - 0.04 * x_range
    x_max_zoom = run_data['step'].max()*1.017
    y_range = run_data['value'].max() - run_data['value'].min() 

    last_30_values = run_data['value'].iloc[-30:]
    y_mean_last_30 = last_30_values.mean()

    y_min_zoom = y_mean_last_30 - 0.1 * y_range
    y_max_zoom = y_mean_last_30 + 0.1 * y_range
    
    ax_inset.set_xlim(x_min_zoom, x_max_zoom)
    ax_inset.set_ylim(y_min_zoom, y_max_zoom)
    ax_inset.set_xticklabels([])
    
    # Add bracket showing step noise
    add_bracket(ax_inset, run_data, noise_label, inset=True)
    
    if inset_only is False:
        ax.indicate_inset_zoom(ax_inset, edgecolor="black")
    
    if inset_legend is False:
        ax.legend([ax.plot([], [], color='grey')[0]], ['1B Run'], loc='upper left', fontsize=8)

def plot_random_seeds(ax: plt.Axes, metric_data: pd.DataFrame, label: str, emw_width=20, plot_inset=True, inset_legend=True):
    """Plot multiple random seeds with inset showing seed noise"""
    num_runs = len(metric_data['run_name'].unique())
    blues = plt.cm.Blues(np.linspace(0.3, 0.9, num_runs))
    
    # Plot each run
    final_values = []
    for idx, run_name in enumerate(metric_data['run_name'].unique()):
        run_data = metric_data[metric_data['run_name'] == run_name]
        ax.plot(run_data['step'], run_data['value'], color=blues[idx], alpha=0.15, linewidth=1)
        ema = run_data['value'].ewm(span=emw_width).mean()
        ax.plot(run_data['step'], ema, color=blues[idx], alpha=0.7, linewidth=1)        
        final_values.append(ema.iloc[-1])

    if plot_inset:
        # Add inset
        ax_inset = ax.inset_axes([0.5, 0.1, 0.45, 0.45])
        plot_inset_seeds(ax_inset, metric_data, blues, label=label, inset_legend=inset_legend)
        ax.indicate_inset_zoom(ax_inset, edgecolor="black")

    legend_loc = 'upper left'

    if label == 'total variation':
        label = 'seed + data order' # rename for TV

        # # Add total variation bracket across full x-range
        # x_min = metric_data['step'].min()
        # x_max = metric_data['step'].max()
        # y_max = metric_data['value'].max()
        
        # # Draw the bracket
        # bracket_y = y_max * 1.02  # Place slightly above max y value
        # ax.plot([x_min, x_max], [bracket_y, bracket_y], color='black', linewidth=1)
        # ax.plot([x_min, x_min], [bracket_y, bracket_y - 0.005], color='black', linewidth=1) 
        # ax.plot([x_max, x_max], [bracket_y, bracket_y - 0.005], color='black', linewidth=1)

        # ylim = ax.get_ylim()
        
        # # Add label
        # ax.text((x_min + x_max)/2, bracket_y + (ylim[1]-ylim[0])*0.02, 'total variation', 
        #         horizontalalignment='center', verticalalignment='bottom')
        
        # # Increase ylim
        # ax.set_ylim(ylim[0], ylim[1] + (ylim[1]-ylim[0])*0.08)

        # legend_loc = 'lower right'


        # Add total variation bracket along y-axis
        x_min = metric_data['step'].min()
        x_max = metric_data['step'].max()
        y_min = metric_data['value'].min()
        y_max = metric_data['value'].max()
        
        # Draw the bracket
        bracket_x = x_max * 1.02  # Place slightly right of max x value
        ax.plot([bracket_x, bracket_x], [y_min, y_max], color='black', linewidth=1)
        ax.plot([bracket_x, bracket_x - 0.005*x_max], [y_min, y_min], color='black', linewidth=1)
        ax.plot([bracket_x, bracket_x - 0.005*x_max], [y_max, y_max], color='black', linewidth=1)

        xlim = ax.get_xlim()
        
        # Add label
        ax.text(bracket_x + (xlim[1]-xlim[0])*0.02, (y_min + y_max)/2, 'total variation',
                rotation=90, horizontalalignment='left', verticalalignment='center')
        
        # Increase xlim
        ax.set_xlim(xlim[0], xlim[1] + (xlim[1]-xlim[0])*0.08)
    
    ax.grid(True, alpha=0.3)
    ax.legend([ax.plot([], [], color='grey')[0]], [f'1B Run (varying {label})'], loc=legend_loc, fontsize=8)

def plot_inset_seeds(ax_inset: plt.Axes, metric_data: pd.DataFrame, colors, label, inset_legend=True):
    """Plot the inset for random seeds visualization"""
    final_values = []
    for idx, run_name in enumerate(metric_data['run_name'].unique()):
        run_data = metric_data[metric_data['run_name'] == run_name]
        ax_inset.plot(run_data['step'], run_data['value'], color=colors[idx], alpha=0.15, linewidth=1)
        ema = run_data['value'].ewm(span=20).mean()
        ax_inset.plot(run_data['step'], ema, color=colors[idx], alpha=0.7, linewidth=1)
        final_values.append(ema.iloc[-1])
    
    # Set zoom region
    x_range = run_data['step'].max() - run_data['step'].min()
    y_range = run_data['value'].max() - run_data['value'].min() 

    if 'data' in label:
        x_min_zoom = run_data['step'].max() - 0.08 * x_range
        y_min_zoom = run_data['value'].max() - 0.15 * y_range
    elif 'seed' in label:
        x_min_zoom = run_data['step'].max() - 0.08 * x_range
        y_min_zoom = run_data['value'].max() - 0.15 * y_range
    else:
        raise ValueError()
        
    x_max_zoom = run_data['step'].max()*1.045
    y_max_zoom = run_data['value'].max()*1.02

    # Center the y-axis around the mean value
    mean_value = np.mean(final_values)
    y_range = y_max_zoom - y_min_zoom
    y_min_zoom = mean_value - y_range/2
    y_max_zoom = mean_value + y_range/2
    
    ax_inset.set_xlim(x_min_zoom, x_max_zoom)
    ax_inset.set_ylim(y_min_zoom, y_max_zoom)
    ax_inset.set_xticklabels([])

    bracket_label = f'{label} noise'.replace(' ', '\n')
    
    add_bracket(ax_inset, run_data, bracket_label, final_values, inset=True)

    if inset_legend is True:
        ax_inset.legend([ax_inset.plot([], [], color='grey')[0]], [f'1B Run'], loc='lower right', fontsize=8)


def plot_datasets(ax: plt.Axes, plotted_task: str, metric: str, mixes: List[str], seed: int, df: pd.DataFrame):
    """Plot different datasets"""
    final_values = []
    for mix in mixes:
        curve_data = get_slice(df, mix=mix, task=plotted_task, size='1B', seed=seed)
        
        ax.plot(curve_data['step'], curve_data[metric], alpha=0.15, linewidth=1)
        ema = curve_data[metric].ewm(span=5).mean()
        ax.plot(curve_data['step'], ema, alpha=0.7, linewidth=1)
        final_values.append(curve_data[metric].iloc[-1])

    add_bracket(ax, curve_data, '$\\text{signal}$', final_values)
    
    ax.legend([ax.plot([], [], color='grey')[0]], ['1B Run (varying data)'], loc='upper left', fontsize=8)
    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0], current_xlim[1] * 1.11)

def add_bracket(ax: plt.Axes, data: pd.DataFrame, label: str, values=None, inset=False):
    """Add a bracket with label to show variation"""
    if values is None:
        y_min = data['value'].iloc[-50:].min()
        y_max = data['value'].iloc[-50:].max()
    else:
        y_min = min(values)
        y_max = max(values)
        
    y_mid = (y_min + y_max) / 2

    if inset:
        if 'seed' in label or 'data' in label:
            x_pos = data['step'].max() * 1.005
            bracket_width = x_pos * 0.003
            text_spacing = bracket_width / 2
        else:
            x_pos = data['step'].max() * 1.002
            bracket_width = x_pos * 0.001
            text_spacing = bracket_width
    else:
        x_pos = data['step'].max() * 1.02
        bracket_width = x_pos * 0.01
        text_spacing = bracket_width

    ax.plot([x_pos, x_pos], [y_min, y_max], color='black', linewidth=1)
    ax.plot([x_pos, x_pos - bracket_width], [y_min, y_min], color='black', linewidth=1)
    ax.plot([x_pos, x_pos - bracket_width], [y_max, y_max], color='black', linewidth=1)
    
    ax.annotate(label,
                xy=(x_pos + text_spacing, y_mid),
                xytext=(x_pos + text_spacing * 2, y_mid),
                ha='left', va='center')

def format_axes(axes: np.ndarray):
    """Format all axes with proper labels and styling"""
    def format_func(x, p):
        return f"{int(x/1000)}K"
        
    for ax in axes.flatten():
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.grid(True, alpha=0.3)


def plot_task_accuracy(ax: plt.Axes, two_class_results, task, sizes, show_legend=False, size_colors=SIZE_COLORS):
    # First plot all scatter points
    all_x = []
    all_y = []
    for size in list(size_colors.keys()):
        if size not in two_class_results.index.tolist():
            continue
        data = two_class_results.loc[size]
        x = np.array(two_class_results.columns, dtype=np.float64)
        y = np.array(data.values, dtype=np.float64)
        
        # Plot scatter points with consistent colors
        ax.scatter(x, y, marker='o', label=f'{size}', s=5, color=size_colors[size])
        
        # Collect valid points for overall spline
        mask = ~np.isnan(y) & ~np.isnan(x) & ~np.isneginf(y) & ~np.isneginf(x)
        all_x.extend(x[mask])
        all_y.extend(y[mask])
    
    # Add interpolating spline, ignoring nans
    mask = ~np.isnan(all_y) & ~np.isnan(all_x)
    if np.sum(mask) >= 3:  # Need at least 4 points for cubic spline
        all_x = np.array(np.array(all_x)[mask]) # exclude compute=0
        all_y = np.array(np.array(all_y)[mask]) # exclude compute=0

        x_nonzero = all_x != 0
        all_x = all_x[x_nonzero] # exclude x=0 values
        all_y = all_y[x_nonzero] # exclude x=0 values
        
        # Sort points by x value
        sort_idx = np.argsort(all_x)
        all_x = all_x[sort_idx]
        all_y = all_y[sort_idx]
        
        # Fit smoothed B-spline with high smoothing parameter
        x_smooth = np.logspace(np.log10(min(all_x)), np.log10(max(all_x)), len(all_x))
        # Use UnivariateSpline with high smoothing for a smoother fit
        spline = UnivariateSpline(np.log10(all_x), all_y, s=len(all_x))
        y_smooth = spline(np.log10(x_smooth))

        ax.plot(x_smooth, y_smooth, color='k', linestyle='--', label='spline', linewidth=1)
    
    # Add random baseline
    ax.axhline(y=0.5, color='r', linestyle='-', label='random', linewidth=0.5)
    
    ax.set_xlabel('Compute')
    ax.set_ylabel('2-class Accuracy')
    ax.set_title(f'{task}')
    ax.set_xscale('log', base=10)
    if show_legend: ax.legend(loc='lower right', fontsize=10, ncols=2)

    # Add vertical lines at specific FLOPS values with matching colors and accuracies
    for size in list(size_colors.keys()):
        if size not in two_class_results.index.tolist():
            continue
        try:
            flops = two_class_results.loc[size].dropna().index[0]
            acc = two_class_results.loc[size].get(np.float64(flops), np.nan)
            if not np.isnan(acc) and not np.isneginf(acc):
                ax.axvline(x=flops, color=size_colors[size], linestyle=':', alpha=0.7)
                ax.text(
                    flops, 0.98, ' ' + ('1.' if acc == 1 else f'{acc:.2f}').lstrip('0'), 
                    rotation=0, color=size_colors[size], ha='left', va='bottom', fontsize=8)
            else:
                # raise FileNotFoundError(f'Not all results found for task={task}, size={size}')
                raise FileNotFoundError(f'Not all results found for task={task}, size={size}')
        except Exception as e:
            # raise RuntimeError(f'Cant graph cheap decisions lines: {e}')
            print(f'Cant graph cheap decisions lines: {e}')


def plot_snr_scatter(ax, x_vals, y_vals, tasks, size, task_names, alpha=0.7, s=10):
    """Create scatter plot with task labels"""
    ax.scatter(x_vals, y_vals, alpha=alpha, label=size, s=s)
    texts = []
    for x, y, task in zip(x_vals, y_vals, tasks):
        pretty_name = task_names.get(task, task)
        texts.append(ax.text(x, y, pretty_name, fontsize=8, alpha=alpha))
    return texts


def config_snr_ax(ax, x_vals, y_vals, texts, xlabel, plot_fit=False, log_scale=False):
    """Configure axis properties"""
    def add_fit_line(ax, x_vals, y_vals):
        """Add line of best fit with confidence interval"""
        x_log = np.log10(x_vals)
        z = np.polyfit(x_log, y_vals, 1)
        p = np.poly1d(z)

        x_line = np.logspace(np.log10(min(x_vals)), np.log10(max(x_vals)), 100)
        y_line = p(np.log10(x_line))

        n = len(x_vals)
        x_mean = np.mean(x_log)
        s_err = np.sqrt(np.sum((y_vals - p(x_log)) ** 2) / (n - 2))
        x_new = np.log10(x_line)
        conf = (
            stats.t.ppf(0.975, n - 2)
            * s_err
            * np.sqrt(1 / n + (x_new - x_mean) ** 2 / np.sum((x_log - x_mean) ** 2))
        )

        r = np.corrcoef(x_log, y_vals)[0, 1]
        r2 = r**2
        stderr = s_err * np.sqrt((1 - r2) / (n - 2))

        ax.text(
            0.03,
            0.97,
            f"R = {r:.3f} ± {stderr:.3f}\nR² = {r2:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
        )
        ax.plot(x_line, y_line, "--", color="black", alpha=0.5)
        ax.fill_between(x_line, y_line - conf, y_line + conf, color="gray", alpha=0.2)

    ax.set_ylim(top=1)
    if plot_fit:
        add_fit_line(ax, x_vals, y_vals)
    if log_scale:
        ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.2f}".format(x)))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Decision Accuracy", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")
    # adjustText(ax, texts)
