from scipy import stats
from plot import adjustText
import numpy as np
import matplotlib.pyplot as plt

def create_scatter_plot(ax, x_vals, y_vals, tasks, size, task_names, alpha=0.7, s=10):
    """Create scatter plot with task labels"""
    ax.scatter(x_vals, y_vals, alpha=alpha, label=size, s=s)
    texts = []
    for x, y, task in zip(x_vals, y_vals, tasks):
        pretty_name = task_names.get(task, task)
        texts.append(ax.text(x, y, pretty_name, fontsize=8, alpha=alpha))
    return texts

def add_fit_line(ax, x_vals, y_vals):
    """Add line of best fit with confidence interval"""
    x_log = np.log10(x_vals)
    z = np.polyfit(x_log, y_vals, 1)
    p = np.poly1d(z)
    
    x_line = np.logspace(np.log10(min(x_vals)), np.log10(max(x_vals)), 100)
    y_line = p(np.log10(x_line))
    
    n = len(x_vals)
    x_mean = np.mean(x_log)
    s_err = np.sqrt(np.sum((y_vals - p(x_log))**2)/(n-2))
    x_new = np.log10(x_line)
    conf = stats.t.ppf(0.975, n-2) * s_err * np.sqrt(1/n + (x_new - x_mean)**2 / np.sum((x_log - x_mean)**2))
    
    r = np.corrcoef(x_log, y_vals)[0,1]
    r2 = r**2
    stderr = s_err * np.sqrt((1-r2)/(n-2))
    
    ax.text(0.03, 0.97, f'R = {r:.3f} ± {stderr:.3f}\nR² = {r2:.3f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.plot(x_line, y_line, '--', color='black', alpha=0.5)
    ax.fill_between(x_line, y_line-conf, y_line+conf, color='gray', alpha=0.2)

def configure_axis(ax, x_vals, y_vals, texts, xlabel, plot_fit=False, log_scale=False):
    """Configure axis properties"""
    ax.set_ylim(top=1)
    if plot_fit:
        add_fit_line(ax, x_vals, y_vals)
    if log_scale:
        ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Decision Accuracy', fontsize=12) 
    ax.grid(True, linestyle='--', alpha=0.3, which='both')
    # adjustText(ax, texts)
    