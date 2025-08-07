import json
from pathlib import Path

from utils.gpt import generate_gpt, print_estimate_cost
from utils.parser import _parse_choices
from utils.__init__ import DATA_DIR

from oe_eval.tasks.base_task import Task


LLM_PROMPT = """You are given a QUESTION and a set of CHOICES, we want to convert each CHOICE into a standalone statement (NEW_CHOICES). We want the NEW_CHOICES to be as similar as possible, where the only different text is the meaning of the original CHOICE, and we want the different text to occur at the end of the statement. 

Make sure the number of NEW_CHOICES matches the number of provided CHOICES.

For example:

{few_shot_examples}

Now you try!

QUESTION: {question}

CHOICES: {choice_text}

NEW CHOICES:"""

FEW_SHOT_TEMPLATE = """QUESTION: {example_question}

CHOICES: {example_choices_input}

NEW CHOICES: {example_choices_output}"""


# This can be overridden for specific tasks!
DEFAULT_EXAMPLE_QUESTION = ["Data in tables may also be presented in graphs. Which type of data would best be displayed on a circle graph?"]

DEFAULT_EXAMPLE_CHOICES_INPUT = ["""
- the distance of the planets from the sun
- the depths of the major oceans on Earth
- the amount of rainfall each day for a month
- the percent of various materials in solid waste
"""]

DEFAULT_EXAMPLE_CHOICES_OUTPUT = ["""
- Data in tables may also be presented in graphs. The distance of the planets from the sun is best displayed on a circle graph.
- Data in tables may also be presented in graphs. The depths of the major oceans on Earth is best displayed on a circle graph.
- Data in tables may also be presented in graphs. The amount of rainfall each day for a month is best displayed on a circle graph.
- Data in tables may also be presented in graphs. The percent of various materials in solid waste is best displayed on a circle graph.
"""]


def paraphrase_task(task: Task, limit=None):
    task.download()

    docs = task.get_eval_docs(limit=limit)

    docs = _run_paraphrase(task, docs)

    filepath = f'{DATA_DIR}/paraphrased/{task.task_config["task_name"]}/{task.task_config["split"]}.json'

    # save reformatted benchmark
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)


def paraphrase_few_shot(task: Task, few_shot_examples: list[dict]): 
    docs = [task._process_doc(doc) for doc in few_shot_examples]

    docs = _run_paraphrase(task, docs)

    for i, example in enumerate(few_shot_examples):
        few_shot_examples[i]['paraphrased_choices'] = docs[i]['paraphrased_choices']

    return few_shot_examples


def _run_paraphrase(task: Task, docs: dict):
    prompts = []

    for i, doc in enumerate(docs):
        question = doc['p_query']
        choices  = doc['p_choices']

        choice_text = '\n- ' + '\n- '.join(choices)

        # construct few show examples
        few_shot_question, few_shot_input, few_shot_output = task.get_examples()

        if few_shot_question is None: few_shot_question = DEFAULT_EXAMPLE_QUESTION
        if few_shot_input is None:    few_shot_input = DEFAULT_EXAMPLE_CHOICES_INPUT
        if few_shot_output is None:   few_shot_output = DEFAULT_EXAMPLE_CHOICES_OUTPUT

        few_shot_text = '\n\n'.join([
            FEW_SHOT_TEMPLATE.format(
                example_question=fs_q,
                example_choices_input=fs_i.rstrip(),
                example_choices_output=fs_o.rstrip(),
            ) for fs_q, fs_i, fs_o in zip(few_shot_question, few_shot_input, few_shot_output)
        ])

        # construct GPT prompt
        prompt = LLM_PROMPT.format(
            few_shot_examples=few_shot_text,
            question=question,
            choice_text=choice_text,
        )

        if i == 0: print("\033[94m" + prompt + "\033[0m")

        prompts += [prompt]

    print_estimate_cost(prompts, model='gpt-4o-mini', input_cost=0.15, output_cost=0.6)
    # print_estimate_cost(prompts, model='gpt-4o', input_cost=2.5, output_cost=10)

    responses = generate_gpt(prompts, model='gpt-4o-mini', max_tokens=1024)

    N_RETRIES = 5
    
    # Attempt parsing responses, with retries on failure
    for i, (prompt, response, doc) in enumerate(zip(prompts, responses, docs)):
        n_retries = 0
        while n_retries < N_RETRIES:
            choices = doc['p_choices']
            parsed = _parse_choices(response, len(choices))

            if parsed is None:
                # Parsing failed, attempt to retry
                print(f'Parsing failed, retrying... ({n_retries}/{N_RETRIES})')
                response = generate_gpt([prompt], model='gpt-4o-mini', max_tokens=1024)
            else:
                paraphrased_choices = parsed
                break
            n_retries += 1

        if parsed is None:
            print(f'Parsing failed: {doc}')
            paraphrased_choices = doc['p_choices']

        if 'p_context' in doc:
            paraphrased_choices = [doc['p_context'] + c for c in paraphrased_choices]

        docs[i]['paraphrased_choices'] = paraphrased_choices

    # Remove p_ fields
    for doc in docs:
        del doc['p_query']
        del doc['p_choices']
        if 'p_context' in doc: del doc['p_context']

    return docs
