from snr.constants.tasks import OLMES, OLMES_GEN, MULTITASK_MATH, MULTITASK_CODE, PALOMA, CUSTOM_LOSS

TASKS = (
    OLMES
    + ["mmlu", "mmlu_pro" "minerva", "agi_eval", "medmcqa", "autobencher"]
    + OLMES_GEN
    + MULTITASK_CODE
    + MULTITASK_MATH
)

