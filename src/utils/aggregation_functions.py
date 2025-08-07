import numpy as np
import scipy
from scipy.spatial import KDTree
from scipy.special import psi
from scipy.stats import median_abs_deviation

def rel_std_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    spread = np.mean(data_noise)
    noise = np.mean(step_noise)
    rel_spread = spread / np.mean(data_scores)
    rel_noise = noise / np.mean(data_scores_last_n)
    return rel_spread, rel_noise, rel_spread / rel_noise

def discrepancy_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    scores = data_scores
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    signal = scipy.stats.qmc.discrepancy(scores.reshape(-1, 1), iterative=False, method="CD")
    return signal, rel_noise, signal / rel_noise

def star_discrepancy_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    scores = data_scores
    scores_sorted = np.sort(scores)
    n = len(scores)
    t_vals = np.concatenate(([0], scores_sorted, [1]))
    empirical_cdf = np.arange(n+1) / n
    discrepancies = np.abs(empirical_cdf - t_vals[:-1])
    signal = np.mean(discrepancies)
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def star_discrepancy_shifted_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    scores = data_scores
    scores_shifted = (scores - scores.min()) / (scores.max() - scores.min())
    scores_sorted = np.sort(scores_shifted)
    n = len(scores_shifted)
    t_vals = np.concatenate(([0], scores_sorted, [1]))
    empirical_cdf = np.arange(n+1) / n
    discrepancies = np.abs(empirical_cdf - t_vals[:-1])
    signal = np.mean(discrepancies)
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def rel_star_discrepancy_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    scores = data_scores
    scores_sorted = np.sort(scores)
    n = len(scores)
    t_vals = np.concatenate(([0], scores_sorted, [1]))
    empirical_cdf = np.arange(n+1) / n
    discrepancies = np.abs(empirical_cdf - t_vals[:-1])
    star_discrepancy = np.mean(discrepancies)
    signal = star_discrepancy / np.mean(data_scores_last_n)
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def dispersion_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    scores = data_scores
    signal = np.max([np.abs(x - y) for i,x in enumerate(scores) for y in scores[i+1:]])
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def dispersion_shifted_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    scores = data_scores
    scores_shifted = (scores - scores.min()) / (scores.max() - scores.min())
    signal = np.max([np.abs(x - y) for i,x in enumerate(scores_shifted) for y in scores_shifted[i+1:]])
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def rel_dispersion_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    scores = data_scores
    dispersion = np.max([np.abs(x - y) for i,x in enumerate(scores) for y in scores[i+1:]])
    signal = dispersion / np.mean(data_scores_last_n)
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def mpd_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    scores = data_scores
    signal = np.mean([abs(x - y) for i,x in enumerate(scores) for y in scores[i+1:]])
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def mpsd_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    scores = data_scores
    signal = np.mean([(x - y)**2 for i,x in enumerate(scores) for y in scores[i+1:]])
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def rel_mpd_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    scores = data_scores
    mpd = np.mean([abs(x - y) for i,x in enumerate(scores) for y in scores[i+1:]])
    signal = mpd / np.mean(data_scores_last_n)
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def rel_mpsd_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    scores = data_scores
    mpd = np.mean([(x - y)**2 for i,x in enumerate(scores) for y in scores[i+1:]])
    signal = mpd / np.mean(data_scores_last_n)
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def mad_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    """Median Absolute Deviation SNR"""
    scores = data_scores
    median = np.median(scores)
    signal = np.median(np.abs(scores - median))
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def aad_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    """Average Absolute Deviation SNR"""
    scores = data_scores
    mean = np.mean(scores)
    signal = np.mean(np.abs(scores - mean))
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def dist_std_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    """Distance Standard Deviation SNR - std dev of pairwise distances"""
    scores = data_scores
    distances = [abs(x - y) for i,x in enumerate(scores) for y in scores[i+1:]]
    signal = np.std(distances)
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def gini_snr( step_noise, data_scores, data_noise, data_scores_last_n ):
    """Gini Coefficient SNR"""
    scores = data_scores
    # Normalize scores to [0,1]
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    n = len(scores)
    indices = np.argsort(scores)
    scores_sorted = scores[indices]
    cumsum = np.cumsum(scores_sorted)
    # Calculate Gini coefficient
    signal = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def entropy_snr(step_noise, data_scores, data_noise, data_scores_last_n):
    """Entropy-based SNR using histogram bins"""
    scores = data_scores
    # Use Freedman-Diaconis rule for bin width
    iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
    bin_width = 2 * iqr / (len(scores) ** (1/3)) if iqr > 0 else 0.1
    bins = int((max(scores) - min(scores)) / bin_width) if bin_width > 0 else 10
    bins = max(5, min(bins, len(scores))) # Keep bins between 5 and n
    hist, _ = np.histogram(scores, bins=bins, density=True)
    hist = hist[hist > 0] # Remove zero counts
    signal = -np.sum(hist * np.log(hist)) # Shannon entropy
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def quartile_deviation_snr(step_noise, data_scores, data_noise, data_scores_last_n):
    """Quartile Deviation (Semi-Interquartile Range) SNR"""
    scores = data_scores
    q75, q25 = np.percentile(scores, [75, 25])
    signal = (q75 - q25) / 2  # Semi-interquartile range
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def rms_deviation_snr(step_noise, data_scores, data_noise, data_scores_last_n):
    """Root Mean Square Deviation SNR"""
    scores = data_scores
    mean = np.mean(scores)
    signal = np.sqrt(np.mean((scores - mean) ** 2))
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def range_snr(step_noise, data_scores, data_noise, data_scores_last_n):
    """Range-based SNR"""
    scores = data_scores
    signal = np.max(scores) - np.min(scores)
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def robust_range_snr(step_noise, data_scores, data_noise, data_scores_last_n):
    """Robust Range SNR using 5th and 95th percentiles"""
    scores = data_scores
    p95, p5 = np.percentile(scores, [95, 5])
    signal = p95 - p5
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def differential_entropy_knn(step_noise, data_scores, data_noise, data_scores_last_n, k=1):
    """Differential entropy"""
    # We need to add a small perturbations because it will degenerate to 0 if two numbers are the same
    data_scores = np.array(data_scores)
    data_scores += 1e-10 * np.random.randn(*data_scores.shape)

    data_scores = np.asarray(data_scores).reshape(-1, 1)
    n = data_scores.shape[0]
    
    tree = KDTree(data_scores)
    epsilons, _ = tree.query(data_scores, k + 1)
    epsilons = epsilons[:, -1]  # k-th neighbor distance

    volume_unit_ball = 2  # in 1D
    entropy = psi(n) - psi(k) + np.log(volume_unit_ball) + np.mean(np.log(epsilons))

    entropy = np.exp(entropy) # make values strictly positive

    signal = entropy
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return signal, rel_noise, signal / rel_noise

def rel_differential_entropy_knn(step_noise, data_scores, data_noise, data_scores_last_n, k=1):
    """Relative differential entropy"""
    # We need to add a small perturbations because it will degenerate to 0 if two numbers are the same
    data_scores = np.array(data_scores)
    data_scores += 1e-10 * np.random.randn(*data_scores.shape)
    
    data_scores = np.asarray(data_scores).reshape(-1, 1)
    n = data_scores.shape[0]
    
    tree = KDTree(data_scores)
    epsilons, _ = tree.query(data_scores, k + 1)
    epsilons = epsilons[:, -1]  # k-th neighbor distance

    volume_unit_ball = 2  # in 1D
    entropy = psi(n) - psi(k) + np.log(volume_unit_ball) + np.mean(np.log(epsilons))

    entropy = np.exp(entropy) # make values strictly positive

    rel_signal = entropy / np.mean(data_scores_last_n)
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)
    return rel_signal, rel_noise, rel_signal / rel_noise

def iqr_snr(step_noise, data_scores, data_noise, data_scores_last_n):
    """Interquartile Range (IQR)"""
    data_scores = np.asarray(data_scores)
    q75, q25 = np.percentile(data_scores, [75, 25])
    iqr = q75 - q25

    signal = iqr
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(data_scores_last_n)

    rel_signal = signal / np.mean(data_scores_last_n)

    return rel_signal, rel_noise, rel_signal / rel_noise

def tukey_snr(step_noise, data_scores, data_noise, data_scores_last_n):
    data_scores = np.asarray(data_scores)
    n = len(data_scores)
    sorted_data = np.sort(data_scores)

    depths = np.minimum(
        np.searchsorted(sorted_data, data_scores, side='right') / n,
        1 - np.searchsorted(sorted_data, data_scores, side='left') / n
    )
    q75, q25 = np.percentile(depths, [75, 25])
    signal = q75 - q25

    data_scores_last_n = np.asarray(data_scores_last_n)
    depths_last = np.minimum(
        np.searchsorted(sorted_data, data_scores_last_n, side='right') / n,
        1 - np.searchsorted(sorted_data, data_scores_last_n, side='left') / n
    )
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(depths_last)

    rel_signal = signal / np.mean(data_scores_last_n)

    return rel_signal, rel_noise, rel_signal / rel_noise

def projection_snr(step_noise, data_scores, data_noise, data_scores_last_n):
    data_scores = np.asarray(data_scores)
    med = np.median(data_scores)
    mad = median_abs_deviation(data_scores, scale='normal')

    depths = 1 / (1 + np.abs(data_scores - med) / mad)
    q75, q25 = np.percentile(depths, [75, 25])
    signal = q75 - q25

    data_scores_last_n = np.asarray(data_scores_last_n)
    depths_last = 1 / (1 + np.abs(data_scores_last_n - med) / mad)
    noise = np.mean(step_noise)
    rel_noise = noise / np.mean(depths_last)

    rel_signal = signal / np.mean(data_scores_last_n)

    return rel_signal, rel_noise, rel_signal / rel_noise


AGGREGATION_FUNCTIONS = [
    {
        "title": "Rel. Std. Dev.",
        "latex": r"$\sigma/\mu$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Rel. Std",
        "snr_xlabel": "SNR = Data Rel. Std / Step Rel. Std",
        "func": rel_std_snr,
    },
    {
        "title": "Discrepancy",
        "latex": r"$\max_{c} |F_n(c) - F(c)|$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Discrepancy",
        "snr_xlabel": "SNR = Data Discrepancy / Step Rel. Std",
        "func": discrepancy_snr,
    },
    {
        "title": "Star Discrepancy",
        "latex": r"$\sup_{[0,c]} |F_n(t) - F(t)|$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Star Discrepancy",
        "snr_xlabel": "SNR = Data Star Discrepancy / Step Rel. Std",
        "func": star_discrepancy_snr,
    },
    {
        "title": "Star Discrepancy (Shift+Scale)",
        "latex": r"$\sup_{[0,c]} |F_n(t) - F(t)|$ with shifting",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Star Discrepancy (Shift+Scale)",
        "snr_xlabel": "SNR = Data Star Discrepancy / Step Rel. Std",
        "func": star_discrepancy_shifted_snr,
    },
    {
        "title": "Star Rel. Discrepancy",
        "latex": r"$\sup_{[0,c]} |F_n(t) - F(t)|/F(t)$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Star Rel. Discrepancy",
        "snr_xlabel": "SNR = Data Star Rel. Discrepancy / Step Rel. Std",
        "func": rel_star_discrepancy_snr,
    },
    {
        "title": "Dispersion",
        "latex": r"$\max_{i,j} |c_i - c_j|$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Dispersion",
        "snr_xlabel": "SNR = Data Dispersion / Step Rel. Std",
        "func": dispersion_snr,
    },
    {
        "title": "Dispersion (Shift+Scale)",
        "latex": r"$\max_{i,j} |c_i - c_j|$ with shifting",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Dispersion (Shift+Scale)",
        "snr_xlabel": "SNR = Data Dispersion / Step Rel. Std",
        "func": dispersion_shifted_snr,
    },
    {
        "title": "Rel. Dispersion",
        "latex": r"$\max_{i,j} |c_i - c_j|/\bar{c}$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Rel. Dispersion",
        "snr_xlabel": "SNR = Data Rel. Dispersion / Step Rel. Std",
        "func": rel_dispersion_snr,
    },
    {
        "title": "Mean Pairwise Distance",
        "latex": r"$\frac{1}{n^2}\sum_{i,j} |c_i - c_j|$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Mean Pairwise Distance",
        "snr_xlabel": "SNR = Data MPD / Step Rel. Std",
        "func": mpd_snr,
    },
    {
        "title": "Rel. Mean Pairwise Distance",
        "latex": r"$\frac{1}{n^2}\sum_{i,j} |c_i - c_j|/\bar{c}$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Rel. Mean Pairwise Distance",
        "snr_xlabel": "SNR = Data Rel. MPD / Step Rel. Std",
        "func": rel_mpd_snr,
    },
    {
        "title": "Mean Squared Pairwise Distance",
        "latex": r"$\frac{1}{n^2}\sum_{i,j} (c_i - c_j)^2$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Mean Squared Pairwise Distance",
        "snr_xlabel": "SNR = Data MSPD / Step Rel. Std",
        "func": mpsd_snr,
    },
    {
        "title": "Rel. Mean Squared Pairwise Distance",
        "latex": r"$\frac{1}{n^2}\sum_{i,j} (c_i - c_j)^2/\bar{c}^2$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Rel. Mean Squared Pairwise Distance",
        "snr_xlabel": "SNR = Data Rel. MSPD / Step Rel. Std",
        "func": rel_mpsd_snr,
    },
    {
        "title": "Median Absolute Deviation",
        "latex": r"$\text{median}(|c_i - \text{median}(c)|)$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe MAD",
        "snr_xlabel": "SNR = Data MAD / Step Rel. Std",
        "func": mad_snr,
    },
    {
        "title": "Average Absolute Deviation",
        "latex": r"$\frac{1}{n}\sum_i |c_i - \bar{c}|$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Avg Abs Dev",
        "snr_xlabel": "SNR = Data AAD / Step Rel. Std",
        "func": aad_snr,
    },
    {
        "title": "Distance Standard Deviation",
        "latex": r"$\frac{1}{n}\sum_i (c_i - \bar{c})$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Dist Std",
        "snr_xlabel": "SNR = Data Dist Std / Step Rel. Std",
        "func": dist_std_snr,
    },
    {
        "title": "Gini Coefficient",
        "latex": r"$\frac{1}{2n^2\mu}\sum_{i,j} |c_i - c_j|$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Gini",
        "snr_xlabel": "SNR = Data Gini / Step Rel. Std",
        "func": gini_snr,
    },
    {
        "title": "Quartile Deviation",
        "latex": r"$(Q_3 - Q_1)/2$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Quartile Dev",
        "snr_xlabel": "SNR = Data Quartile Dev / Step Rel. Std",
        "func": quartile_deviation_snr,
    },
    {
        "title": "RMS Deviation",
        "latex": r"$\sqrt{\frac{1}{n}\sum_i (c_i - \bar{c})^2}$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe RMS Dev",
        "snr_xlabel": "SNR = Data RMS Dev / Step Rel. Std",
        "func": rms_deviation_snr,
    },
    {
        "title": "Range",
        "latex": r"$\max(c) - \min(c)$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Range",
        "snr_xlabel": "SNR = Data Range / Step Rel. Std",
        "func": range_snr,
    },
    {
        "title": "Robust Range",
        "latex": r"$P_{95} - P_5$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Robust Range",
        "snr_xlabel": "SNR = Data Robust Range / Step Rel. Std",
        "func": robust_range_snr,
    },
    {
        "title": "Rel. Std. Dev.",
        "latex": r"$\sigma/\mu$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Rel. Std",
        "snr_xlabel": "SNR = Data Rel. Std / Step Rel. Std",
        "func": rel_std_snr,
    },
    {
        "title": "Rel. Dispersion",
        "latex": r"$\max_{i,j} |c_i - c_j|/\bar{c}$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Rel. Dispersion",
        "snr_xlabel": "SNR = Data Rel. Dispersion / Step Rel. Std",
        "func": rel_dispersion_snr,
    },
    {
        "title": "Median Absolute Deviation",
        "latex": r"$\text{median}(|c_i - \text{median}(c)|)$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe MAD",
        "snr_xlabel": "SNR = Data MAD / Step Rel. Std",
        "func": mad_snr,
    },
    {
        "title": "Interquartile Range",
        "latex": r"$Q_3 - Q_1$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe IQR",
        "snr_xlabel": "SNR = Data IQR / Step Rel. Std",
        "func": iqr_snr,
    },
    {
        "title": "Quartile Deviation",
        "latex": r"$(Q_3 - Q_1)/2$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Quartile Dev",
        "snr_xlabel": "SNR = Data Quartile Dev / Step Rel. Std",
        "func": quartile_deviation_snr,
    },
    {
        "title": "Robust Range",
        "latex": r"$P_{95} - P_5$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Robust Range",
        "snr_xlabel": "SNR = Data Robust Range / Step Rel. Std",
        "func": robust_range_snr,
    },
    {
        "title": "Halfspace Depth", # aka Tukey Depth 
        "latex": r"$\min\left( F_n(x),\ 1 - F_n(x) \right)$ where $F_n(x) = \frac{1}{n} \sum_{i=1}^n \mathbb{I}[c_i \leq x]$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Halfspace Depth",
        "snr_xlabel": "SNR = Data Halfspace Depth / Step Rel. Std",
        "func": tukey_snr,
    },
    {
        "title": "Projection Depth",
        "latex": r"$\left( 1 + \frac{|x - \text{med}(c)|}{\text{MAD}(c)} \right)^{-1}$",
        "signal_xlabel": "Step-to-Step Rel. Std",
        "noise_xlabel": "Data Recipe Projection Depth",
        "snr_xlabel": "SNR = Data Projection Depth / Step Rel. Std",
        "func": projection_snr,
    },
]
