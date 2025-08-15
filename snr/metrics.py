import numpy as np


def signal_to_noise_ratio(signal_scores: np.ndarray, noise_scores: np.ndarray) -> float:
    """
    signal = max_{j,k} |m_j - m_k| / m̄
    noise = σ_m / m̄
    snr = signal / noise
    """
    dispersion = np.max([np.abs(mj - mk) for mj in signal_scores for mk in signal_scores])
    signal = dispersion / np.mean(signal_scores)
    noise = np.std(noise_scores) / np.mean(noise_scores)
    snr = signal / noise
    return snr


def decision_acc_fast(scores_small: np.ndarray, scores_target: np.ndarray) -> float:
    scores_small = np.array(scores_small)
    scores_target = np.array(scores_target)
    small_diffs = scores_small[:, np.newaxis] > scores_small[np.newaxis, :]
    target_diffs = scores_target[:, np.newaxis] > scores_target[np.newaxis, :]
    mask = np.triu(np.ones_like(small_diffs), k=1).astype(bool)
    agreements = (small_diffs == target_diffs)[mask]
    return np.mean(agreements)