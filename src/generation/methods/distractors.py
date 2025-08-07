import re
import json
import random
import copy

from pathlib import Path

from utils.gpt import generate_gpt, print_estimate_cost
from utils.parser import _parse_choices
from utils import DATA_DIR

from oe_eval.tasks.base_task import Task

random.seed(42)


LLM_PROMPT = """You are given examples of QUESTIONS, with four CHOICES and an ANSWER (0, 1, 2, 3). Please generate {n_new_distractors} new INCORRECT DISTRACTOR CHOICES that are incorrect, but plausible answers to the question. Make sure the INCORRECT DISTRACTOR CHOICES are adequately different than existing options, and ensure that the new DISTRACTOR CHOICES are incorrect. Please add an explanation as to why they are incorrect using a [REASON][/REASON] tag.

For example:

{few_shot_examples}

Now you try!

QUESTION: {question}

CHOICES: {choice_text}

ANSWER: {answer}

NEW DISTRACTOR CHOICES:"""

FEW_SHOT_TEMPLATE = """QUESTION: {example_question}

CHOICES: {example_choices}

ANSWER: {example_answer}

NEW INCORRECT DISTRACTOR CHOICES: {example_distractors}"""


# This can be overridden for specific tasks!
DEFAULT_EXAMPLE_QUESTION = ["Data in tables may also be presented in graphs. Which type of data would best be displayed on a circle graph?"]

DEFAULT_EXAMPLE_CHOICES = ["""
- the distance of the planets from the sun
- the depths of the major oceans on Earth
- the amount of rainfall each day for a month
- the percent of various materials in solid waste
"""]

DEFAULT_EXAMPLE_ANSWER = [3]

DEFAULT_EXAMPLE_DISTRACTORS = ["""
- the price of Apple stock over time [REASON] stock prices are time series data, so would best be represented in a line graph [/REASON]
- results of a presidential election different geographies [REASON] since this graph is communicating the results of different areas, it would be best displayed on a map [/REASON]
- the distribution of a null hypothesis [REASON] a distribution is best represented with a histogram [/REASON]
- a linear regression for the relationship between two variables [REASON] this is best expressed using a scatter plot to show different values of points [/REASON]
"""]



def add_distractors_task(task: Task, n_new_distractors: int, limit=None):
    task.download()

    docs = task.get_eval_docs(limit=limit)

    docs = _run_add_distractors_task(n_new_distractors, docs)

    filepath = f'{DATA_DIR}/distractors/{task.task_config["task_name"]}:distractors/{task.task_config["split"]}.json'

    # save reformatted benchmark
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)


def add_distractors_task_few_shot(task: Task, few_shot_examples: list[dict], n_new_distractors: int = 4): 
    docs = [task._process_doc(doc) for doc in few_shot_examples]

    docs = _run_add_distractors_task(n_new_distractors, docs)
    for i, example in enumerate(few_shot_examples):
        few_shot_examples[i]['choices'] = docs[i]['choices']

    few_shot_examples = docs

    return few_shot_examples


def get_id(doc):
    if 'ind' in doc:
        key = 'ind'
        id = doc['ind']
    elif 'idx' in doc:
        key = 'idx'
        id = doc['idx']
    elif 'index' in doc:
        key = 'index'
        id = doc['index']
    elif 'id' in doc:
        key = 'id'
        id = doc['id']
    else:
        raise KeyError(doc)
    return id, key


def _run_add_distractors_task(n_new_distractors: int, docs: dict, override_idx: bool=True):
    prompts = []

    c1, c2 = str(34), str(35)
    print(f'\033[{c1}mExample doc: \033[0m\033[{c2}m{docs[0]}\033[0m')

    assert n_new_distractors == 4, 'Changing number of distractors not implemented yet'
    # I'd need to change the number of distractors in the example and change the few_shot function input

    for i, doc in enumerate(docs):
        try:
            id, key = get_id(doc)
            question = doc['query']
            choices  = doc['choices']
            answer   = doc['gold']
        except KeyError as e:
            raise KeyError(f'{e}: ' + str(doc))

        choice_text = '\n- ' + '\n- '.join(choices)

        # construct few show examples
        few_shot_question, few_shot_choices, few_shot_answers, few_shot_distractors = None, None, None, None

        if few_shot_question is None:    few_shot_question = DEFAULT_EXAMPLE_QUESTION
        if few_shot_choices is None:     few_shot_choices = DEFAULT_EXAMPLE_CHOICES
        if few_shot_answers is None:     few_shot_answers = DEFAULT_EXAMPLE_ANSWER
        if few_shot_distractors is None: few_shot_distractors = DEFAULT_EXAMPLE_DISTRACTORS

        few_shot_text = '\n\n'.join([
            FEW_SHOT_TEMPLATE.format(
                example_question=fs_q,
                example_choices=fs_i.rstrip(),
                example_answer=fs_a,
                example_distractors=fs_d
            ) for fs_q, fs_i, fs_a, fs_d in zip(few_shot_question, few_shot_choices, few_shot_answers, few_shot_distractors)
        ])

        # construct GPT prompt
        prompt = LLM_PROMPT.format(
            n_new_distractors=n_new_distractors,
            few_shot_examples=few_shot_text,
            question=question,
            choice_text=choice_text,
            answer=answer,
        )

        if i == 0: print("\033[94m" + prompt + "\033[0m")

        prompts += [prompt]

    print_estimate_cost(prompts, model='gpt-4o-mini', input_cost=0.15, output_cost=0.6)
    # print_estimate_cost(prompts, model='gpt-4o', input_cost=2.5, output_cost=10)

    responses = generate_gpt(prompts, model='gpt-4o-mini', max_tokens=1024)

    N_RETRIES = 5
    
    # Attempt parsing responses, with retries on failure
    new_docs = []
    for i, (prompt, response, doc) in enumerate(zip(prompts, responses, docs)):
        n_retries = 0
        while n_retries < N_RETRIES:
            try:
                response = response.replace('NEW INCORRECT DISTRACTOR CHOICES:', '')
                response = response.lstrip()

                # Parse answer choices
                response_choices = _parse_choices(response, n_choices=n_new_distractors)

                # remove reason text
                def remove_reason_tags(text):
                    return re.sub(r'\[REASON\].*?\[/REASON\]', '', text, flags=re.DOTALL).rstrip()
                response_choices = [remove_reason_tags(r) for r in response_choices]
            except (IndexError, AttributeError, AssertionError, ValueError, TypeError) as e:
                print(f"Error parsing response: {e}\nResponse:\n{repr(response)}")
                response_choices = None

            if response_choices is None:
                # Parsing failed, attempt to retry
                print(f'Parsing failed, retrying... ({n_retries}/{N_RETRIES})')
                response = generate_gpt([prompt], model='gpt-4o-mini', max_tokens=1024)
            else:
                break
            n_retries += 1

        if response_choices is not None:
            new_doc = copy.deepcopy(docs[i])

            id, key = get_id(doc)
            gold_choice = new_doc['choices'][new_doc['gold']]
            new_choices = new_doc['choices'] + response_choices

            # reshuffle the distractors and gold choice
            random.shuffle(new_choices)

            new_doc['choices'] = new_choices
            new_doc['gold'] = new_choices.index(gold_choice)
            if override_idx:
                new_doc[key] = f'distractors_{id}'
                new_doc['id'] = f'distractors_{id}'

            new_docs += [new_doc]

    return new_docs
