import json
from pathlib import Path

from utils.__init__ import DATA_DIR

from oe_eval.tasks.base_task import Task


def convert_mc_to_rc(task: Task, limit=None):
    task.download()

    docs = task.get_eval_docs(limit=limit)

    docs = _run_convert_rc(task, docs)

    filepath = f'{DATA_DIR}/cloze/{task.task_config["task_name"]}:rc/{task.task_config["split"]}.json'

    # save reformatted benchmark
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)


def convert_mc_to_rc_few_shot(task: Task, few_shot_examples: list[dict]): 
    docs = [task._process_doc(doc) for doc in few_shot_examples]

    few_shot_examples = _run_convert_rc(task, docs)

    return few_shot_examples


def _run_convert_rc(task: Task, docs: dict):
    """ Output doc must have keys: ['id', 'query', 'choices', 'gold'] """
    for i, doc in enumerate(docs):
        # mmlu pro
        if 'mmlu' in task.task_name:
            docs[i]['id'] = docs[i]['question_id']
            docs[i]['query'] = docs[i]['query']
            docs[i]['gold'] = docs[i]['answer_index']
            docs[i]['choices'] = docs[i]['options']
        else:
            raise NotImplementedError()

    return docs
