from collections import defaultdict
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

SEP = '>>>'
N_RETRIES = 5


LLM_PROMPT = """You are given examples of QUESTIONS, with an ANSWER. Please generate {n_choices} new INCORRECT DISTRACTOR ANSWERS that are incorrect, but plausible answers to the question. Make sure the INCORRECT DISTRACTOR ANSWERS are adequately different than existing options, and ensure that the new DISTRACTOR ANSWERS are incorrect. Please add an explanation as to why they are incorrect using a [REASON][/REASON] tag.

For example:

{few_shot_examples}


Now you try!

QUESTION: {question}

CORRECT ANSWER: {answer}

NEW INCORRECT ANSWERS:"""

FEW_SHOT_TEMPLATE = """QUESTION: {example_question}

CORRECT ANSWER: {example_answer}

NEW INCORRECT ANSWERS: {example_incorrect_answers}"""


# This is the first question of GSM (probably should take the example from elsewhere)
DEFAULT_EXAMPLE_QUESTION = ["Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"]

DEFAULT_EXAMPLE_COT = ["Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18"]

DEFAULT_EXAMPLE_ANSWER = [18]

DEFAULT_EXAMPLE_INCORRECT_ANSWERS = [[
    "26 [REASON] Incorrectly ignores the 4 eggs used for muffins in the calculation of the remaining eggs sold.[/REASON]",
    "8 [REASON] Miscalculates the number of eggs available for sale by misinterpreting how many are used for baking versus how many are sold.[/REASON]",
    "16 [REASON] The calculation is correct but misunderstands the problem; the total eggs used for breakfast and baking should be subtracted from 16, not counted separately.[/REASON]",
    "26 [REASON] Assumes all 16 eggs can be sold despite consuming some, leading to an inflated total.[/REASON]"
]]



def perturb_rc_task(task: Task, n_choices: int, process_docs=False, limit=None):
    task.download()
    docs = task.get_eval_docs(limit=limit)

    if process_docs: # 'coqa'
        docs = [task._process_doc(doc) for doc in docs]

    docs = _run_perturb_rc_task(task, docs, n_choices)

    filepath = f'{DATA_DIR}/perturb_rc/{task.task_config["task_name"]}:perturb_rc/{task.task_config["split"]}.json'

    # save reformatted benchmark
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)


def perturb_rc_task_few_shot(task: Task, few_shot_examples: list[dict], n_choices: int=4): 
    few_shot_examples = [task._process_doc(doc) for doc in few_shot_examples]

    c1, c2 = str(31), str(32)
    print(f'\033[{c1}mExample few shot doc: \033[0m\033[{c2}m{few_shot_examples[0]}\033[0m')

    few_shot_parsed = _run_perturb_rc_task(task, few_shot_examples, n_choices)

    return few_shot_parsed


def parse_doc(doc):
    # Get question
    if 'query' in doc:
        question = doc['query']
        question = question.replace('Passage: ', '').replace("\nAnswer:", "")
    elif 'question' in doc:
        question = doc['question']
    elif 'problem' in doc:
        question = doc['problem']
    elif 'prompt' in doc:
        question = doc['prompt']
    else:
        raise KeyError(doc)
    
    # Get gold answer
    if 'short_answer' in doc:
        answer = doc['short_answer']
    elif 'answers' in doc:
        answer = doc['answers']
    elif 'answer_value' in doc:
        answer = doc['answer_value']
    elif 'answers_text' in doc:
        answer = doc['answers_text']
    elif 'answer' in doc:
        answer = doc['answer']
    elif 'label' in doc:
        answer = doc['label']
    else:
        raise KeyError(doc)
    
    # Since we are converting to RC, some benchmarks may have multiple answers, but we will
    # just take the first answer
    if isinstance(answer, list) or isinstance(answer, tuple):
        answer = answer[0]
        if isinstance(answer, list) or isinstance(answer, tuple):
            answer = answer[0] # no lists in this house! (looking at you DROP)
            if isinstance(answer, list) or isinstance(answer, tuple):
                answer = answer[0] # no lists in this house!
    
    return question, answer


def _run_perturb_rc_task(task: Task, docs: dict, n_choices: int = 4):
    prompts = []

    for i, doc in enumerate(docs):
        if i == 0: 
            c1, c2 = str(31), str(32)
            print(f'\033[{c1}mExample doc: \033[0m\033[{c2}m{doc}\033[0m')

        question, answer = parse_doc(doc)

        # construct few show examples
        few_shot_question, few_shot_answers, few_shot_incorrect_answers = None, None, None

        if few_shot_question is None:          few_shot_question = DEFAULT_EXAMPLE_QUESTION
        if few_shot_answers is None:           few_shot_answers = DEFAULT_EXAMPLE_ANSWER
        if few_shot_incorrect_answers is None: few_shot_incorrect_answers = DEFAULT_EXAMPLE_INCORRECT_ANSWERS

        few_shot_incorrect_answers = [f'\n{SEP} ' + f'\n{SEP} '.join([str(ic) for ic in fs_ic]) for fs_ic in few_shot_incorrect_answers]

        few_shot_text = '\n\n'.join([
            FEW_SHOT_TEMPLATE.format(
                example_question=fs_q,
                example_answer=fs_a,
                example_incorrect_answers=fs_ia
            ) for fs_q, fs_a, fs_ia in zip(few_shot_question, few_shot_answers, few_shot_incorrect_answers)
        ])

        # construct GPT prompt
        prompt = LLM_PROMPT.format(
            n_choices=n_choices-1,
            few_shot_examples=few_shot_text,
            question=question,
            answer=answer,
        )

        if i == 0: print("\033[94m" + prompt + "\033[0m")

        prompts += [prompt]

    print_estimate_cost(prompts, model='gpt-4o-mini', input_cost=0.15, output_cost=0.6)
    # print_estimate_cost(prompts, model='gpt-4o', input_cost=2.5, output_cost=10)

    responses = generate_gpt(prompts, model='gpt-4o-mini', max_tokens=4096)
    
    # Attempt parsing responses, with retries on failure
    new_docs = []
    for i, (prompt, response, doc) in enumerate(zip(prompts, responses, docs)):
        n_retries = 0
        while n_retries < N_RETRIES:
            try:
                if isinstance(response, list) and len(response) == 1: response = response[0] # collapse a list of a single string to a string

                response = response.replace('NEW INCORRECT ANSWERS:', '')
                response = response.lstrip()

                # Parse answer choices (we can get away with n-1 answers)
                response_choices = _parse_choices(response, n_choices=[n_choices, n_choices-1], sep=SEP)

                # TODO: VERIFY ANSWERS ARE INCORRECT

                # remove reason text
                def remove_reason_tags(text):
                    return re.sub(r'\[REASON\].*?\[/REASON\]', '', text, flags=re.DOTALL).rstrip()
                response_choices = [remove_reason_tags(r) for r in response_choices]
            except (IndexError, AttributeError, AssertionError, ValueError, TypeError) as e:
                # raise RuntimeError(repr(response))
                print(f"Error parsing response: {e}\nResponse:\n{repr(response)}")
                response_choices = None

            if response_choices is None:
                # Parsing failed, attempt to retry
                c1 = str(31)
                print(f'\033[{c1}mParsing failed, retrying... ({n_retries}/{N_RETRIES})\033[0m')
                response = generate_gpt([prompt], model='gpt-4o-mini', max_tokens=4096)
            else:
                break
            n_retries += 1

        if response_choices is not None:
            new_doc = copy.deepcopy(docs[i])

            # TODO: verify incorrect cots are incorrect

            id = i # use original doc id?
            _, answer = parse_doc(doc)
            query = task.doc_to_text(doc)

            # shuffle perturbed cot
            choices = [answer] + response_choices[:n_choices]
            random.shuffle(choices)

            # construct final query
            new_doc['choices'] = choices
            new_doc['gold'] = choices.index(answer)
            new_doc['query'] = query
            new_doc['id'] = f'cot_perturb_{id}'

            new_docs += [new_doc]

    return new_docs
