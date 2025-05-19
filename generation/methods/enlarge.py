import json
import random

from pathlib import Path

from utils.gpt import generate_gpt, print_estimate_cost
from utils.parser import _parse_choices
from utils.__init__ import DATA_DIR

from oe_eval.tasks.base_task import Task

random.seed(42)


LLM_PROMPT = """You are given examples of QUESTIONS, please generate a new question similar in difficulty and general topic to the given QUESTIONS, but make sure it tests some new knowledge not covered by the existing questions. Please provide the QUESTION, {n_choices} answer CHOICES, and the zero-indexed index of the answer in ANSWER.

For example:

{few_shot_examples}

Now you try!

NEW QUESTION:"""

FEW_SHOT_TEMPLATE = """QUESTION: {example_question}

CHOICES: {example_choices}

ANSWER: {example_answer}"""


def enlarge_task(task: Task, limit=None):
    task.download()

    docs = task.get_eval_docs()

    docs = _run_enlarge_task(limit, docs)

    filepath = f'{DATA_DIR}/enlarge/{task.task_config["task_name"]}:enlarge/{task.task_config["split"]}.json'

    # save reformatted benchmark
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)


def enlarge_task_few_shot(task: Task, few_shot_examples: list[dict]): 
    docs = [task._process_doc(doc) for doc in few_shot_examples]

    # docs = _run_enlarge_task(task, docs)
    # for i, example in enumerate(few_shot_examples):
    #     few_shot_examples[i]['paraphrased_choices'] = docs[i]['paraphrased_choices']

    few_shot_examples = docs

    return few_shot_examples


def _get_task_examples(docs: dict, n_examples=4):
    """ Randomly sample from docs dict """
    sampled_docs = random.sample(docs, n_examples)

    try:
        questions = [doc['query'] for doc in sampled_docs]
        choices   = [doc['choices'] for doc in sampled_docs]
        answers   = [doc['gold'] for doc in sampled_docs]
    except KeyError as e:
        raise KeyError(f'{e}: ' + str(sampled_docs[0]))

    choices   = ['\n- ' + '\n- '.join(c) for c in choices]

    # A bit hacky...
    questions = [q.replace('Question: ', '').replace('\nAnswer:', '') for q in questions]

    return questions, choices, answers


def _run_enlarge_task(n_new_instances: int, docs: dict):
    prompts = []

    c1, c2 = str(34), str(35)
    print(f'\033[{c1}mExample doc: \033[0m\033[{c2}m{docs[0]}\033[0m')

    if 'choices' in docs[0]:
        n_choices = len(docs[0]['choices'])
    else:
        raise KeyError(docs[0])

    for i in range(n_new_instances):
        # construct few show examples
        few_shot_question, few_shot_choices, few_shot_answers = _get_task_examples(docs)

        few_shot_text = '\n\n'.join([
            FEW_SHOT_TEMPLATE.format(
                example_question=fs_q,
                example_choices=fs_i.rstrip(),
                example_answer=fs_a
            ) for fs_q, fs_i, fs_a in zip(few_shot_question, few_shot_choices, few_shot_answers)
        ])

        # construct GPT prompt
        prompt = LLM_PROMPT.format(
            few_shot_examples=few_shot_text,
            n_choices=n_choices
        )

        if i == 0: print("\033[94m" + prompt + "\033[0m")

        prompts += [prompt]

    print_estimate_cost(prompts, model='gpt-4o-mini', input_cost=0.15, output_cost=0.6)
    # print_estimate_cost(prompts, model='gpt-4o', input_cost=2.5, output_cost=10)

    responses = generate_gpt(prompts, model='gpt-4o-mini', max_tokens=1024)

    N_RETRIES = 5
    
    # Attempt parsing responses, with retries on failure
    new_docs = []
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        n_retries = 0
        while n_retries < N_RETRIES:
            try:
                response = response.split('QUESTION: ')[-1]
                response_question, suffix = response.split('\n\nCHOICES:')
                response_choices, response_answer = suffix.split('\n\nANSWER:')

                # Clean up artifacts
                response_question = response_question.replace('\nAnswer:', '').rstrip() 
                response_choices = response_choices.lstrip()
                response_answer = int(response_answer.strip())

                if response_answer >= len(response_choices):
                    raise IndexError(f'Recieved answer index of {response_answer}, but there are only {response_choices} choices!')

                # Parse answer choices
                response_choices = _parse_choices(response_choices, n_choices=n_choices)

                # Add back scaffolding
                response_question = f'Question: {response_question}\nAnswer:'
            except (IndexError, AttributeError, AssertionError, ValueError) as e:
                print(f"Error parsing response: {e}\nResponse:\n{repr(response)}")
                response_choices = None

            if response_choices is None:
                # Parsing failed, attempt to retry
                print(f'Parsing failed, retrying... ({n_retries}/{N_RETRIES})')
                response = generate_gpt([prompt], model='gpt-4o-mini', max_tokens=1024)
            else:
                new_docs += [{
                    'id': f'synthetic_{i}',
                    'query': response_question,
                    'choices': response_choices,
                    'gold': response_answer,
                }]
                break
            n_retries += 1

    return new_docs
