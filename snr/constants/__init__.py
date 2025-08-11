import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'analysis', 'data')
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'img')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

from snr.constants.plot import PRETTY_TASK_NAMES, SHORT_TASK_NAME
from snr.constants.tasks import MMLU, MINERVA, MMLU_PRO, OLMES, OLMES_GEN, AGI_EVAL, BBH, MMLU_MC, OLMES_MC, PALOMA, CUSTOM_LOSS, MULTITASK_MATH, MULTITASK_CODE, MULTITASK_KNOWLEDGE, MULTITASK, OLMES_ALL, OLMES_ALL, MULTITASK


def get_selected_tasks():
    selected_tasks = \
        [OLMES, MINERVA, OLMES_GEN, MMLU, MMLU_PRO, AGI_EVAL, BBH] + \
        OLMES + OLMES_GEN + \
        ['mbpp', 'mbppplus', 'codex_humaneval', 'codex_humanevalplus'] + \
        ['autobencher'] + \
        ["gsm_plus", "gsm_symbolic_main", "gsm_symbolic_p1", "gsm_symbolic_p2", "medmcqa", "minerva_math_500"]
    
        # multitask_math, multitask_code, multitask_knowledge, multitask, olmes_all
    
    return selected_tasks


def get_title_from_task(task):
    if isinstance(task, list):
        assert len(task) > 0, f'Seeing empty array passed as a task: {task}'
        if len(task) == 1:
            return task[0]
        for key, title in SHORT_TASK_NAME.items():
            if key in task[0]:
                return title
        return 'aggregate'
    return task


def get_pretty_task_name(task):
    """Map task names to prettier display names"""
    task = get_title_from_task(task)
    if task not in PRETTY_TASK_NAMES:
        print(f"Task does not have pretty name: {task}")
    return PRETTY_TASK_NAMES.get(task, task)
