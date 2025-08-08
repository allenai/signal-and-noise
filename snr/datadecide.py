import numpy as np

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

FULL_SCHEDULE = {
    '4M': 5725,
    '20M': 14584,
    '60M': 29042,
    '90M': 29901,
    '150M': 38157,
    '300M': 45787,
    '530M': 57786,
    '750M': 63589,
    '1B': 69369,
}

MODEL_TO_BATCH = {
    '4M': 32, # batch_size=32, gpus=8
    '6M': 32,
    '8M': 32,
    '10M': 32,
    '14M': 32,
    '16M': 32,
    '20M': 64,
    '60M': 96,
    '90M': 160,
    '150M': 192,
    '300M': 320,
    '530M': 448,
    '750M': 576,
    '1B': 704
}

MODEL_TO_PARAMETERS = {
    '4M': 3_744_832,
    '6M': 6_010_464,
    '8M': 8_538_240,
    '10M': 9_900_432,
    '12M': 12_066_600,
    '14M': 14_380_224,
    '16M': 16_004_560,
    '20M': 19_101_888,
    '60M': 57_078_144,
    '90M': 97_946_640,
    '150M': 151898880,
    '300M': 319980544,
    '530M': 530074944,
    '750M': 681297408,
    '1B': 1_176_832_000
}


def get_compute(scale):
    return 2048 * 6 * MODEL_TO_BATCH[scale] * MODEL_TO_PARAMETERS[scale] * FULL_SCHEDULE[scale]


def compute_2_class(ranking_a, ranking_b):
    """ Compute 2-class accuracy """
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


def decision_acc_fast(scores_small, scores_target):
    scores_small = np.array(scores_small)
    scores_target = np.array(scores_target)
    small_diffs = scores_small[:, np.newaxis] > scores_small[np.newaxis, :]
    target_diffs = scores_target[:, np.newaxis] > scores_target[np.newaxis, :]
    mask = np.triu(np.ones_like(small_diffs), k=1).astype(bool)
    agreements = (small_diffs == target_diffs)[mask]
    return np.mean(agreements)


def get_slice(df, model, task):
    try:
        df = df.loc[(task, model)]
    except KeyError:
        return df.iloc[0:0]
    df = df.reset_index()
    return df
