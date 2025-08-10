import os
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'analysis', 'data')
PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'img')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def get_selected_tasks(tasks):
    mmlu      = [t for t in tasks if 'mmlu' in t and ':' not in t and '_pro_' not in t]
    minerva   = [t for t in tasks if 'minerva' in t and ':' not in t and 'math_500' not in t and t != 'minerva']
    mmlu_pro  = [t for t in tasks if '_pro_' in t and ':rc' in t]
    olmes     = ['arc_challenge', 'arc_easy', 'boolq', 'csqa', 'hellaswag', 'openbookqa', 'piqa', 'socialiqa', 'winogrande']
    olmes_gen = ['drop', 'gsm8k', 'jeopardy', 'squad', 'triviaqa'] # naturalqs
    agi_eval  = [t for t in tasks if 'agi_eval' in t and ':rc' in t]
    bbh       = [t for t in tasks if 'bbh' in t and ':' not in t]
    
    mmlu_mc   = [t for t in tasks if 'mmlu' in t and ':mc' in t and '_pro_' not in t]
    olmes_mc  = [f'{task}:mc' for task in olmes]
    paloma    = [t for t in tasks if 'paloma' in t]
    llm_compression = [t for t in tasks if 'llm_compression' in t]
    custom_loss = [t for t in tasks if 'custom_loss' in t]

    # task suites
    multitask_math = ["gsm_plus", "gsm_symbolic_main", "gsm_symbolic_p1", "gsm_symbolic_p2", "minerva_math_500", "aime"] # 6
    multitask_code = ['mbpp', 'mbppplus', 'codex_humaneval', 'codex_humanevalplus'] # 4
    multitask_knowledge = ["medmcqa", 'autobencher'] + olmes + mmlu + olmes_gen + mmlu_pro + agi_eval # 19
    multitask = multitask_knowledge + multitask_math + multitask_code + bbh # 30
    olmes_all = olmes + mmlu + olmes_gen

    # Re-order tasks so that the title logic works (a bit hacky, yes)
    olmes_all = ['jeopardy'] + list(set(olmes_all) - {'jeopardy'})
    multitask = ['boolq'] + list(set(multitask) - {'boolq'})

    # [minerva, mmlu, mmlu_pro, agi_eval] + \
    selected_tasks = \
        [olmes, minerva, olmes_gen, mmlu, mmlu_pro, agi_eval, bbh] + \
        olmes + olmes_gen + \
        ['mbpp', 'mbppplus', 'codex_humaneval', 'codex_humanevalplus'] + \
        ['autobencher'] + \
        ["gsm_plus", "gsm_symbolic_main", "gsm_symbolic_p1", "gsm_symbolic_p2", "medmcqa", "minerva_math_500"] # "aime"
        # [] + \
        # multitask_math, multitask_code, multitask_knowledge, multitask, olmes_all
    
    return selected_tasks

def get_title_from_task(task):
    if isinstance(task, list):
        assert len(task) > 0, f'Seeing empty array passed as a task: {task}'
        if len(task) == 1:
            return task[0]
        title_mapping = {
            'jeopardy': 'olmes_all',
            'boolq': 'multitask_all',
            'gsm_plus': 'multitask_math',
            'mbpp': 'multitask_code',
            'medmcqa': 'multitask_knowledge',
            'mmlu_pro_': 'mmlu_pro',
            'mmlu_abstract_algebra:mc': 'mmlu_mc',
            'mmlu': 'mmlu',
            'minerva': 'minerva',
            'agi_eval': 'agi_eval',
            'bbh': 'bbh',
            'arc_challenge:para': 'olmes_core9_para',
            'arc_challenge:distractors': 'olmes_core9_distractors',
            'arc_challenge:enlarge': 'olmes_core9_enlarge',
            'arc_challenge:mc': 'olmes_core9_mc',
            'arc_challenge': 'olmes_core9',
            'drop': 'olmes_gen',
        }
        for key, title in title_mapping.items():
            if key in task[0]:
                return title
        return 'aggregate'
    return task

def get_pretty_task_name(task):
    """Map task names to prettier display names"""
    # print(task)
    # if isinstance(task, list):
    #     return ','.join(task)
    task = get_title_from_task(task)
    mapping = {
        'arc_challenge:mc': 'ARC Challenge MC',
        'arc_challenge': 'ARC Challenge',
        'arc_easy:mc': 'ARC Easy MC', 
        'arc_easy': 'ARC Easy', 
        'autobencher:mc': 'Autobencher MC',
        'autobencher': 'AutoBencher',
        'boolq:mc': 'BoolQ MC',
        'boolq': 'BoolQ',
        'codex_humaneval': 'HumanEval',
        'codex_humanevalplus': 'HumanEval+',
        'csqa:mc': 'CommonsenseQA MC',
        'csqa': 'CommonsenseQA',
        'drop': 'DROP',
        'gsm8k': 'GSM8K',
        'hellaswag:mc': 'HellaSwag MC',
        'hellaswag': 'HellaSwag',
        'jeopardy': 'Jeopardy',
        'mbpp': 'MBPP',
        'mbppplus': 'MBPP+',
        'minerva': 'Minerva MATH',
        'mmlu_mc': 'MMLU MC',
        'mmlu': 'MMLU',
        'olmes_core9_mc': 'OLMES Core 9 MC',
        'olmes_core9': 'OLMES Core 9',
        'olmes_gen': 'OLMES Gen',
        'openbookqa:mc': 'OpenBookQA MC',
        'openbookqa': 'OpenBookQA',
        'paloma_c4_en': 'Paloma C4',
        'paloma_m2d2_s2orc_unsplit': 'Paloma M2D2',
        'piqa:mc': 'PIQA MC',
        'piqa': 'PIQA',
        'socialiqa:mc': 'SocialIQA MC',
        'socialiqa': 'SocialIQA', 
        'squad': 'SQuAD',
        'triviaqa': 'TriviaQA',
        'winogrande:mc': 'WinoGrande MC',
        'winogrande': 'WinoGrande',
        'agi_eval': 'AGI Eval',
        'aime': 'AIME',
        'bbh': 'BBH',
        'gsm_plus': 'GSM+',
        'gsm_symbolic_main': 'GSM Symbolic',
        'gsm_symbolic_p1': 'GSM Symbolic P1',
        'gsm_symbolic_p2': 'GSM Symbolic P2', 
        'medmcqa': 'MedMCQA',
        'minerva_math_500': 'Minerva MATH 500',
        'mmlu_pro': 'MMLU Pro',
        'olmes_10_macro_avg': 'OLMES 10 Avg.',
        'multitask_math': 'Math Tasks',
        'multitask_code': 'Code Tasks',
        'multitask_knowledge': 'Knowledge Tasks',
        'olmes_all': 'OLMES + Gen',
        'multitask_all': 'All Tasks',
    }
    if task not in mapping:
        print(f"Task does not have pretty name: {task}")
    return mapping.get(task, task)

def weka_to_gcs(model_name):
    if 'weka://' in model_name:
        return f"gs://ai2-llm/checkpoints/davidh/{model_name.split('checkpoints/')[1]}"
    else:
        return model_name

def fix_model_path(name):
    if name.endswith('peteish7/step928646-hf'):
        name = 'peteish7/step928646-hf-vllm-2'
    name = name.replace('OLMoE-1B-7B-0924', 'olmoe-1b-7b-0924')
    name = name.replace('OLMo-7B-hf', 'olmo-7b')
    name = name.replace('phi-1.5', 'phi-1_5')
    name = name.replace('Qwen2-1.5B', 'qwen2-1.5b')
    name = name.replace('Qwen2.5-3B', 'qwen2.5-3b')
    name = name.replace('deepseek-7b', 'deepseek-llm-7b-base')
    return name