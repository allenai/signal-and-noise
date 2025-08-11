DDOS_SIZES = ["4M", "20M", "60M", "90M", "150M", "300M", "530M", "750M", "1B"]

# FULL_SCHEDULE = {
#     '4M': 5725,
#     '20M': 14584,
#     '60M': 29042,
#     '90M': 29901,
#     '150M': 38157,
#     '300M': 45787,
#     '530M': 57786,
#     '750M': 63589,
#     '1B': 69369,
# }

# MODEL_TO_BATCH = {
#     '4M': 32, # batch_size=32, gpus=8
#     '6M': 32,
#     '8M': 32,
#     '10M': 32,
#     '14M': 32,
#     '16M': 32,
#     '20M': 64,
#     '60M': 96,
#     '90M': 160,
#     '150M': 192,
#     '300M': 320,
#     '530M': 448,
#     '750M': 576,
#     '1B': 704
# }

# MODEL_TO_PARAMETERS = {
#     '4M': 3_744_832,
#     '6M': 6_010_464,
#     '8M': 8_538_240,
#     '10M': 9_900_432,
#     '12M': 12_066_600,
#     '14M': 14_380_224,
#     '16M': 16_004_560,
#     '20M': 19_101_888,
#     '60M': 57_078_144,
#     '90M': 97_946_640,
#     '150M': 151898880,
#     '300M': 319980544,
#     '530M': 530074944,
#     '750M': 681297408,
#     '1B': 1_176_832_000
# }

# def get_toks_params(scale):
#     toks = 2048 * MODEL_TO_BATCH[scale] * FULL_SCHEDULE[scale]
#     params = MODEL_TO_PARAMETERS[scale]
#     return toks, params

# def get_compute(scale):
#     toks, params = get_toks_params(scale)
#     return 6 * toks * params

# DDOS_COMPUTE_SIZES = tuple(get_compute(size) for size in DDOS_SIZES)

DDOS_MODEL_NAMES = [
    "DCLM-baseline",
    "dolma17",
    "c4",
    "dclm_ft7percentile_fw2",
    "dclm_ft7percentile_fw3",
    "dclm_fw_top10",
    "dclm_fw_top3",
    "dolma-v1-6-and-sources-baseline",
    "dolma17-25p-DCLM-baseline-75p",
    "dolma17-50p-DCLM-baseline-50p",
    "dolma17-75p-DCLM-baseline-25p",
    "falcon",
    "falcon_and_cc",
    "falcon_and_cc_eli5_oh_top10p",
    "falcon_and_cc_eli5_oh_top20p",
    "falcon_and_cc_og_eli5_oh_top10p",
    "falcon_and_cc_tulu_qc_top10",
    "fineweb_edu_dedup",
    "no_code",
    "no_flan",
    "no_math_no_code",
    "no_reddit",
    "prox_fineweb_pro",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p",
]
