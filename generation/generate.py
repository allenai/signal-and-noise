import json, os

from utils.gpt import openai_init
from utils.__init__ import DATA_DIR, ROOT_DIR

from methods.paraphrase import paraphrase_task, paraphrase_few_shot
from methods.enlarge import enlarge_task, enlarge_task_few_shot
from methods.distractors import add_distractors_task, add_distractors_task_few_shot
from methods.cloze import convert_mc_to_rc, convert_mc_to_rc_few_shot
from methods.cot_perturb import perturb_cot_task, perturb_cot_task_few_shot
from methods.rc_perturb import perturb_rc_task, perturb_rc_task_few_shot

from oe_eval.tasks.base_task import Task
from oe_eval.tasks.fewshot_sources import FEWSHOT_SOURCES
from oe_eval.tasks.oe_eval_tasks.paraphrased import PARAPHRASED_TASK_REGISTRY
from oe_eval.tasks.oe_eval_tasks.synthetic import NEW_TASK_REGISTRY
from oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY
from oe_eval.configs.tasks import TASK_CONFIGS

import datasets

ENLARGE_SIZE = 10_000
LIMIT = None


def sanity_check_length(task: Task):
    """ check if there are the same number of generated and original choices """
    task.download()
    docs = task.get_eval_docs(limit=1)

    for doc in docs:
        try:
            assert len(doc['choices']) == len(doc['paraphrased_choices']), f"Lengths do not match: {len(doc['choices'])} != {len(doc['paraphrased_choices'])}"
        except AssertionError as e:
            raise RuntimeError(f'{doc["idx"]}: {e}')


def _test_dataloader(task_name: str, task_cls: Task, config_name: str):
    print(f'Testing dataloader for "{task_name}" on config "{config_name}"')

    try:
        olmes_config = TASK_CONFIGS[config_name]
    except KeyError as e:
        raise KeyError(f'{e} not found in task configs, seeing: {TASK_CONFIGS.keys()}')

    # rm -rf ~/.cache/huggingface/datasets/_Users_*
    # rm -rf ~/.cache/huggingface/datasets/dataloader/
    task: Task = task_cls(
        task_name = olmes_config['task_name'], 
        task_config = olmes_config,
        download_mode=datasets.DownloadMode.REUSE_CACHE_IF_EXISTS
    )

    if ':mc' in task_name: assert 'MC' in task.__class__.__name__, f'Task name {task_name} but class name is {task.__class__.__name__}. Make sure the task_cls was set properly.'

    task.download()
    # sanity_check_length(task)

    try:
        docs = task.get_eval_docs(limit=1)

        # print(json.dumps(docs, indent=4))

        # For some reason, this is not set properly in task __init__
        for field in ['metric_kwargs', 'context_kwargs', 'generation_kwargs']:
            if task.task_config[field] is None:
                task.task_config[field] = {}

        task.build_all_requests()
        task.build_all_requests() # need to test just in case. this is called multiple times if different variations are used, and occasionally breaks the dataloader

        task_instances = task._instances
        example_instances = [task_instances[0]]
        example_instances = task_instances[:3]

        # print(example_instance)

        # Color the data examples
        if ':para' in task_name:
            c1, c2 = str(34), str(35)
        elif ':rc' in task_name or ':perturb_' in task_name:
            c1, c2 = str(36), str(31)
        elif ':mc' in task_name or ':qa' in task_name:
            c1, c2 = str(32), str(31)
        elif ':cot' in task_name:
            c1, c2 = str(31), str(32)
        else:
            c1, c2 = str(31), str(36)

        for example_instance in example_instances:
            if example_instance.request_type == 'loglikelihood':
                print(f'EXAMPLE REQUEST ({task_name}):\n\033[{c1}m{example_instance.request.context}\033[0m\033[{c2}m{example_instance.request.continuation}\033[0m')
            elif example_instance.request_type == 'generate_until':
                print(f'EXAMPLE REQUEST ({task_name}):\n\033[{c1}m{example_instance.request.context}\033[0m\033[{c2}m {example_instance.label}\033[0m')
            elif example_instance.request_type == 'generate_until_and_loglikelihood':
                # print(f'EXAMPLE REQUEST ({task_name}):\n\033[{c1}m{example_instance.request.context}\033[0m\033[{c2}m {example_instance.label}\033[0m')
                print(f'EXAMPLE REQUEST ({task_name}):\n\033[{c1}m{example_instance.request.context}\033[0m\033[{c2}m{example_instance.request.continuation}\033[0m')
            else:
                raise ValueError(example_instance.request_type)

    except Exception as e:
        raise RuntimeError(f"{task_name} failed on error: {e}")


def test_dataloader():
    """ Test whether the new datasets can be loaded in oe-eval """
    # # for task_name, task_cls in PARAPHRASED_TASK_REGISTRY.items():
    # for task_name, task_cls in SYNTHETIC_TASK_REGISTRY.items():
    #     TASKS_TO_INCLUDE = ['arc_easy', 'arc_challenge'] # mmlu, mmlu_computer_security, 'boolq', 'openbookqa', 'winogrande'
    #     if not any(task in task_name for task in TASKS_TO_INCLUDE): continue

    #     task_root = task_name.split(':')[0]

    #     # Find RC task equivalent
    #     rc_task_name = task_root
    #     rc_task_cls = TASK_REGISTRY[rc_task_name]
    #     rc_task_name += ':rc'
        
    #     # Find MC task equivalent
    #     mc_task_name = f'{task_root}:mc'
    #     mc_task_cls = TASK_REGISTRY[mc_task_name]
    #     mc_config_name = f'{mc_task_name}::olmes'

    #     _test_dataloader(mc_task_name, mc_task_cls, mc_config_name)

    #     for version in ['']: # , ':show_options', ':show_options:full_io'
    #         rc_config_name = f'{rc_task_name}{version}::olmes'
    #         _test_dataloader(rc_task_name, rc_task_cls, rc_config_name)

    #     for version in ['prefix_para']: # 'prefix_para', 'suffix_para', 'full_io'
    #         # Get Para task name
    #         config_name = f'{task_name}:{version}::olmes'
    #         _test_dataloader(task_name, task_cls, config_name)

    # Test non-OLMES core9mcqa
    for task_name, task_cls in TASK_REGISTRY.items():
        TASKS_TO_INCLUDE = ['drop', 'gsm8k', 'jeopardy', 'naturalqs', 'squad', 'triviaqa', 'bbh', 'mmlu_pro_', 'agi_eval'] # minerva perturb_rc enlarge distractors perturb_cot, mmlu_pro, mmlu, mmlu_computer_security, 'boolq', 'openbookqa', 'winogrande'
        if not any(task in task_name for task in TASKS_TO_INCLUDE): continue
        if '_selfc' in task_name: continue # no self-consistency tasks

        task_cls = TASK_REGISTRY[task_name]
        task_root = task_name.split(':')[0]

        task_root = task_root.replace('naturalqa_open', 'naturalqa') # exception for just this task

        if 'zero_scrolls' in task_root: # long context benchmark (for tulu)
            continue

        # Select the relevant version of each benchmark
        if 'perturb_cot' in task_name:
            config_name = f'{task_root}:perturb_cot::olmes'
        elif 'bbh' in task_root:
            task_root = f'{task_root}:qa'
            config_name = f'{task_root}::none'
        elif 'mmlu_pro_' in task_root:
            if ':mc' in task_name:
                task_root = f'{task_root}:mc'
                config_name = f'{task_root}::none'
            elif ':rc' in task_name:
                task_root = f'{task_root}:rc'
                config_name = f'{task_root}::none'
            else:
                continue
        elif task_root in ['ifeval', 'truthfulqa', 'tydiqa_english', 'alpaca_eval'] or 'deepmind' in task_root:
            config_name = f'{task_root}::tulu'
        elif 'enlarge' in task_name:
            config_name = f'{task_root}:enlarge::olmes'
        elif 'distractors' in task_name:
            config_name = f'{task_root}:distractors::olmes'
        elif 'perturb_rc' in task_name:
            config_name = f'{task_root}:perturb_rc::olmes'
        else:
            config_name = f'{task_root}::olmes'

        try:
            _test_dataloader(task_name, task_cls, config_name)
        except KeyError as e:
            # print(f'Skipping: {task_name} {task_root} {config_name}. {e}')
            raise KeyError(f'{task_name} {task_root} {config_name}.')


def get_few_shot_examples(task_name, config, transform):
    # Manual overrides to get the correct few shot example set
    if 'bbh' in task_name:
        new_few_shot_name = task_name.split(':')[0]
        examples = FEWSHOT_SOURCES[f'STD:{new_few_shot_name}']
    elif 'agi_eval' in task_name:
        new_few_shot_name = task_name.split(':')[0]
        new_few_shot_name = new_few_shot_name[::-1].replace('_', ':', 1)[::-1] # replace last _ with : (agi_eval_sat-math -> agi_eval:sat-math)
        examples = FEWSHOT_SOURCES[new_few_shot_name]
    elif 'minerva' in task_name:
        new_few_shot_name = task_name
        examples = FEWSHOT_SOURCES['Minerva:MATH:fixed']
    elif 'gsm' in task_name and transform == 'perturb_cot':
        new_few_shot_name = task_name
        examples = FEWSHOT_SOURCES['STD:GSM8k']
    elif 'gsm' in task_name and transform == 'perturb_rc':
        new_few_shot_name = 'STD:GSM8k'
        examples = FEWSHOT_SOURCES['STD:GSM8k']
    elif 'ifeval' in task_name:
        raise NotImplementedError()
    elif 'coqa' in task_name:
        raise NotImplementedError('Coqa does not have few shot examples')
    else:
        new_few_shot_name = config['fewshot_source'].replace(f':{transform.capitalize()}', '')
        examples = FEWSHOT_SOURCES[new_few_shot_name]
    return new_few_shot_name, examples


def main(transform):
    openai_init()

    NEW_FEWSHOT_SOURCES = {}

    if transform == 'para':
        task_registry = PARAPHRASED_TASK_REGISTRY
    elif transform == 'enlarge':
        task_registry = TASK_REGISTRY
        TASKS_TO_INCLUDE = ["arc_challenge", "arc_easy", "boolq", "csqa", "hellaswag", "piqa", "socialiqa", "openbookqa"] # "winogrande"
    elif transform == 'distractors':
        task_registry = TASK_REGISTRY
        TASKS_TO_INCLUDE = ["arc_challenge", "arc_easy", "boolq", "csqa", "hellaswag", "piqa", "socialiqa", "openbookqa"] # "winogrande"
    elif transform == 'cloze':
        task_registry = TASK_REGISTRY
        TASKS_TO_INCLUDE = ['mmlu_pro_']
    elif transform == 'perturb_rc':
        task_registry = TASK_REGISTRY
        TASKS_TO_INCLUDE = ['drop', 'gsm8k', 'jeopardy', 'naturalqs', 'squad', 'triviaqa'] # 'coqa'
    elif transform == 'perturb_cot':
        task_registry = TASK_REGISTRY
        TASKS_TO_INCLUDE = ['gsm', 'agi_eval', 'minerva', 'bbh'] # ifeval, humaneval, bigcodebench
    else:
        raise ValueError(transform)

    for task_name, task_cls in task_registry.items():
        if not any(task in task_name for task in TASKS_TO_INCLUDE): continue

        cot_prompt_config = None # If we need to prompt GPT for a Gold CoT
        if transform == 'para':
            # We use the abstract parent class of the paraphrased task class
            task_cls = task_cls.__base__
            olmes_config = TASK_CONFIGS[f'{task_name}::olmes']
        elif transform == 'enlarge' or transform == 'distractors':
            if ':' in task_name: continue # only get the root RC task
            olmes_config = TASK_CONFIGS[f'{task_name}:rc::olmes']
        elif transform == 'cloze':
            if ':' in task_name: continue # only one root of the task

            if 'mmlu_pro_' in task_name: 
                # olmes_config = TASK_CONFIGS[f'{task_name}:mc::none']
                olmes_config = TASK_CONFIGS[f'{task_name}:rc::none']
                raise NotImplementedError('RC is now implemented natively.')
            elif 'bbh' in task_name: 
                olmes_config = TASK_CONFIGS[f'{task_name}:qa::none']
            else: 
                raise ValueError(task)
            
            if 'bbh' in task_name:
                olmes_config["context_kwargs"]["short_prefix"] = False
        elif transform == 'perturb_cot':
            special_case = ('agi_eval' in task_name and ':cot::' in task_name) # AGI evals have different format keys (we only want the :cot keys)
            if ':' in task_name and not special_case: continue # only one root of the task
            if '_selfc' in task_name: continue # no self-consistency tasks

            cot_prompt_config = None
            if 'bbh' in task_name: 
                olmes_config = TASK_CONFIGS[f'{task_name}:cot::olmes']
                cot_prompt_config = TASK_CONFIGS[f'{task_name}:cot::olmes']
            elif 'ifeval' in task_name: 
                olmes_config = TASK_CONFIGS[f'{task_name}::tulu']
            elif 'agi_eval' in task_name:
                olmes_config = TASK_CONFIGS[f'{task_name}::none'] # task name already has :cot
                cot_prompt_config = TASK_CONFIGS[f'{task_name}::none']
            elif 'gsm' in task_name or 'minerva' in task_name:
                # Gold CoT provided for these tasks. Do not need to generate
                olmes_config = TASK_CONFIGS[f'{task_name}::olmes']
            else:
                raise ValueError(task_name)
        elif transform == 'perturb_rc':
            if ':' in task_name: continue # only one root of the task
            if '_selfc' in task_name: continue # no self-consistency tasks

            if 'naturalqs' in task_name:
                task_name = task_name.replace('naturalqs_open', 'naturalqs')
            elif 'squad2' in task_name:
                continue # only squad 1 for now

            olmes_config = TASK_CONFIGS[f'{task_name}::olmes']
        else:
            raise ValueError(transform)

        print(f'Processing {task_name}')

        # Manual override for ARC (since there are two ARC classes to inherit)
        if 'arc' in task_name and 'challenge' in task_name:
            olmes_config['dataset_name'] = 'ARC-Challenge'
        elif 'arc' in task_name and 'easy' in task_name:
            olmes_config['dataset_name'] = 'ARC-Easy'

        task: Task = task_cls(
            task_name = olmes_config['task_name'], 
            task_config = olmes_config,
            download_mode=datasets.DownloadMode.REUSE_CACHE_IF_EXISTS
        )

        cot_task = None
        if cot_prompt_config is not None:
            cot_task: Task = task_cls(
                task_name = cot_prompt_config['task_name'], 
                task_config = cot_prompt_config,
                download_mode=datasets.DownloadMode.REUSE_CACHE_IF_EXISTS
            )

        if transform == 'para':
            paraphrase_task(task, limit=LIMIT)
            few_shot_f = paraphrase_few_shot
        elif transform == 'enlarge':
            enlarge_task(task, limit=ENLARGE_SIZE)
            few_shot_f = enlarge_task_few_shot
        elif transform == 'distractors':
            add_distractors_task(task, n_new_distractors=4, limit=LIMIT)
            few_shot_f = add_distractors_task_few_shot
        elif transform == 'cloze':
            convert_mc_to_rc(task, limit=LIMIT)
            few_shot_f = convert_mc_to_rc_few_shot
        elif transform == 'perturb_cot':
            perturb_cot_task(task, cot_task, n_cots=4, limit=LIMIT)
            few_shot_f = perturb_cot_task_few_shot
        elif transform == 'perturb_rc':
            # perturb_rc_task(task, n_choices=4, limit=LIMIT, process_docs=('coqa' in task_name))
            few_shot_f = perturb_rc_task_few_shot
        else:
            raise ValueError(transform)

        if 'mmlu' in task_name:
            # MMLU few shot examples are saved as the dev set
            examples = task.fewshot_examples(1000000, None, None)
            docs = few_shot_f(task, examples)
            if 'mmlu_pro_' in task_name:
                file_name = 'val' 
                task_name = task_name.replace(':mc', ':rc')
            else:
                file_name = 'dev' 
            filepath = f'{DATA_DIR}/{transform}/{task_name}/{file_name}.json'
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(docs, f, ensure_ascii=False, indent=4)
        else:
            new_few_shot_name, examples = get_few_shot_examples(task_name, olmes_config, transform)

            if transform == 'perturb_cot':
                few_shot = few_shot_f(task, cot_task, examples)
            else:
                few_shot = few_shot_f(task, examples)

            NEW_FEWSHOT_SOURCES[f'{new_few_shot_name}:{transform.capitalize()}'] = few_shot

    # save few shot sources
    output_files = [
        f'{DATA_DIR}/{transform}/fewshot_sources.json',
        f'{ROOT_DIR}/olmo-repos/oe-eval-internal/oe_eval/tasks/oe_eval_tasks/synthetic/fewshot_sources_{transform}.json'
    ]
    for output_file in output_files:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(NEW_FEWSHOT_SOURCES, f, indent=4, ensure_ascii=False)


if __name__ == '__main__': 
    # main(transform='para')
    # main(transform='enlarge')
    # main(transform='distractors')
    # main(transform='cloze')
    # main(transform='perturb_cot')
    # main(transform='perturb_rc')

    test_dataloader()
