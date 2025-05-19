import os, argparse, json, tqdm
import copy
from collections import defaultdict
from util import gen_from_prompt, process_args_for_models, helm_process_args
from tool_util import (
    _generate_lm_answers,
    extract_json_v2,
    search_related_pages,
    search_step,
    get_pageviews,
)

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm

from prompts.wiki import (
    DEFAULT_JSON_MESSAGE,
    COMPARE_ANSWERS_STR,
    GEN_QA_WITHOUT_DOCS_CONTEXT,
    QA_PAIRS_AUGMENTED_CONTEXT,
    REFINE_CATEGORIES_RANDOM_AUGMENTED_CONTEXT,
    REFINE_CATEGORIES_TARGETACC_AUGMENTED,
    GENERATE_CATEGORIES_TARGETACC_AUGMENTED,
    GENERATE_CATEGORIES_RANDOM_AUGMENTED,
    ASK_QUESTION,
)


DEFAULT_JSON_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your reasoning and language skills.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
Reply "TERMINATE" in the end when everything is done.
"""


def get_summary_of_results(json_dict, gold_key="python_answer", verbose=False):
    # a summary of the results.
    # summarize by each category.
    category2correct_count = defaultdict(list)
    category2question = defaultdict(list)
    str_summary = "In the following, we summarize the evaluation results by each category in this agent iteration. \n We will report the accuracy for each category, and list the questions that are answered correctly and incorrectly. \n"
    for line in json_dict:
        line["category2"] = (
            f"{line['category']} || {line['wiki_entity']} [{line['additional_requirement']}]"
            if "additional_requirement" in line
            else line["category"]
        )
        category2correct_count[line["category2"]].append(line["is_correct"])
        category2question[(line["category2"], line["is_correct"])].append(line)
    for category in category2correct_count:
        acc_temp = sum(
            [1 if x == "true" else 0 for x in category2correct_count[category]]
        ) / len(category2correct_count[category])
        str_summary += (
            f"category: {category}, accuracy: {round(acc_temp, 3)} "
            f"|| {sum([1 if x == 'true' else 0 for x in category2correct_count[category]])} out of {len(category2correct_count[category])}"
            + "\n"
        )
        if verbose:
            str_summary += "# Questions answered correctly:\n"
            for qq in category2question[(category, "true")]:
                str_summary += (
                    f"{qq['question']} || gold: {qq[gold_key]} || pred: {qq['test_taker_answer']}"
                    + "\n"
                )

                # str_summary += f"{qq['question']} || {qq['difficulty']} || gold: {qq['python_answer']} || pred: {qq['test_taker_answer']}" + "\n"
            str_summary += "# Questions answered incorrectly:\n"
            for qq in category2question[(category, "false")]:
                str_summary += (
                    f"{qq['question']} || gold: {qq[gold_key]} || pred: {qq['test_taker_answer']}"
                    + "\n"
                )
            str_summary += "\n + ------------------------------------ + \n"
    # print(str_summary)
    return str_summary


def summarize_over_history(history_json_dict, gold_key="python_answer", verbose=True):
    """
    :param history: a list of dictionaries. Each dictionary corresponds to a run.
    :return: a summary of the results.
    """
    # augment each line of the dictionary with the iteration number.
    for idx, json_dict in enumerate(history_json_dict):
        for line in json_dict:
            line["iteration"] = idx
    # concatenate the dictionaries.
    json_dict = [line for json_dict in history_json_dict for line in json_dict]
    # a summary of the results.
    str_summary = get_summary_of_results(json_dict, gold_key=gold_key, verbose=verbose)
    # print(str_summary)
    return str_summary


def get_acc_lst(json_dict, gold_key="python_answer"):
    # a summary of the results.
    # summarize by each category.
    category2correct_count = defaultdict(list)
    for line in json_dict:
        category2correct_count[line["category"]].append(line["is_correct"])
    acc_lst = []
    for category in category2correct_count:
        acc = sum(
            [1 if x == "true" else 0 for x in category2correct_count[category]]
        ) / len(category2correct_count[category])
        acc_lst.append(acc)
    return acc_lst


def solve_and_compare_questions(
    test_taker_info,
    agent_info,
    question_json,
    gold_answer,
    outfile_prefix,
    gold_ans_key="gold_answer",
):
    test_taker_output = _generate_lm_answers(
        question_json, test_taker_info, agent_info, outfile_prefix=outfile_prefix
    )
    summary_prev_iteration, history_json = fast_compare_answers(
        gold_answer,
        test_taker_output,
        agent_info,
        outfile_prefix=outfile_prefix,
        gold_ans_key=gold_ans_key,
    )

    return history_json

def process_line(idx, line_gold, line_pred, agent_model_info, context_str, gold_ans_key, test_taker_output):
    agent_lm, agent_tokenizer, agent_client = agent_model_info
    line = {
        "id": str(idx + 1),
        "question": line_gold["question"],
        "gold_answer": line_gold[gold_ans_key],
        "test_taker_answer": line_pred["test_taker_response"],
    }
    for k, v in line_gold.items():
        if k not in line:
            line[k] = v

    pred = line_pred["test_taker_response"].strip()
    gold = line_gold[gold_ans_key].strip()
    q_str = f"Question {idx+1}: {line_gold['question']}\npred={pred} || gold={gold}\nreason:"
    context = context_str + q_str

    request_result = gen_from_prompt(
        model=agent_lm,
        tokenizer=agent_tokenizer,
        prompt=[context],
        echo_prompt=False,
        temperature=0.0,
        max_tokens=3000,
        process_func=None,
        service=agent_client,
        terminate_by_linebreak="no",
        verbose=False,
    )
    response = request_result.completions[0].text
    line["reasons"] = response.strip()
    line["is_correct"] = response.strip().split("##")[-1].strip()
    test_taker_line = test_taker_output[idx]
    line["question"] = test_taker_line["question"]
    line["category"] = test_taker_line.get("category", "None")
    line["difficulty"] = test_taker_line.get("difficulty")
    
    return line


def fast_compare_answers(
    gold_output,
    test_taker_output,
    agent_model_info,
    outfile_prefix="att1",
    gold_ans_key="gold_answer",
):
    if os.path.exists(f"{outfile_prefix}.compare_answers.json"):
        print("FOUND compare_answers.json")
        json_dict = json.load(open(f"{outfile_prefix}.compare_answers.json", "r"))
        str_summary = get_summary_of_results(json_dict, gold_key="gold_answer")
        return str_summary, json_dict

    # print("Comparing the answers generated by the python code and the test taker...")
    assert len(gold_output) == len(test_taker_output)
    agent_lm, agent_tokenizer, agent_client = agent_model_info
    context_str = COMPARE_ANSWERS_STR

    with open(f"{outfile_prefix}.compare_answers.jsonl", "w") as out_handle:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_line,
                    idx,
                    line_gold,
                    line_pred,
                    agent_model_info,
                    context_str,
                    gold_ans_key,
                    test_taker_output,
                )
                for idx, (line_gold, line_pred) in enumerate(zip(gold_output, test_taker_output))
            ]

            correct_count2 = 0
            final_lst = []
            
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc='Comparing answers'):
                line = future.result()
                if line["is_correct"] == "true":
                    correct_count2 += 1
                print(json.dumps(line), file=out_handle)
                final_lst.append(line)

    if len(final_lst) == 0:
        return "", {}

    json_dict = final_lst
    accuracy = correct_count2 / len(json_dict)
    print("accuracy: ", accuracy)
    assert len(json_dict) == len(test_taker_output)
    out_handle.close()

    with open(f"{outfile_prefix}.compare_answers.json", "w") as out_handle:
        json.dump(json_dict, out_handle, indent=2)

    str_summary = get_summary_of_results(json_dict, gold_key="gold_answer")
    return str_summary, json_dict


def gen_qa_without_docs(topic, agent_info, additional_req):
    context = GEN_QA_WITHOUT_DOCS_CONTEXT
    context.replace("{{TOPIC}}", topic)
    agent_lm, agent_tokenizer, agent_client = agent_info

    context += f"Topic: {topic}\nAdditional requirements: {additional_req}\n"
    # extract the json file from the message
    request_result = gen_from_prompt(
        model=agent_lm,
        tokenizer=agent_tokenizer,
        prompt=[context],
        echo_prompt=False,
        temperature=0.0,
        max_tokens=4096,
        process_func=None,
        service=agent_client,
        terminate_by_linebreak="no",
        verbose=False,
    )
    response = request_result.completions[0].text

    extracted_json = extract_json_v2(response, None)
    extracted_json = extracted_json[0]
    return extracted_json


def gen_qa_pairs_augmented(paragraph, agent_info, additional_req):
    context = QA_PAIRS_AUGMENTED_CONTEXT
    agent_lm, agent_tokenizer, agent_client = agent_info

    context += (
        f"Wiki paragraph: {paragraph}\nAdditional requirements: {additional_req}\n"
    )
    # extract the json file from the message
    request_result = gen_from_prompt(
        model=agent_lm,
        tokenizer=agent_tokenizer,
        prompt=[context],
        echo_prompt=False,
        temperature=0.0,
        max_tokens=2000,
        process_func=None,
        service=agent_client,
        terminate_by_linebreak="no",
        verbose=False,
    )
    response = request_result.completions[0].text

    extracted_json = extract_json_v2(response, None)
    extracted_json = extracted_json[0]
    return extracted_json


def _refine_categories_random_augmented(
    theme, agent_info, history, iters, outfile_prefix="att1", acc_target="0.3--0.5"
):
    category_json = _generate_categories_random_augmented(
        theme,
        agent_info,
        history,
        iters,
        outfile_prefix=outfile_prefix + ".brainstorm",
        acc_target=acc_target,
    )
    # given the json_lst, refine the categories to achieve the target accuracy.
    full_cat_lst = []
    for line in category_json:
        cat_lst = search_related_pages(line["category"])
        full_cat_lst.extend(cat_lst)
    context = REFINE_CATEGORIES_RANDOM_AUGMENTED_CONTEXT
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _refine_categories(
        theme,
        context,
        agent_info,
        history,
        iters,
        full_cat_lst,
        outfile_prefix=outfile_prefix + ".refine",
    )


def _refine_categories_targetacc_augmented(
    theme, agent_info, history, iters, outfile_prefix="att1", acc_target="0.3--0.5"
):
    category_json = _generate_categories_targetacc_augmented(
        theme,
        agent_info,
        history,
        iters,
        outfile_prefix=outfile_prefix + ".brainstorm",
        acc_target=acc_target,
    )
    # given the json_lst, refine the categories to achieve the target accuracy.
    full_cat_lst = []
    for line in category_json:
        cat_lst = search_related_pages(line["category"])
        full_cat_lst.extend(cat_lst)
    context = REFINE_CATEGORIES_TARGETACC_AUGMENTED
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _refine_categories(
        theme,
        context,
        agent_info,
        history,
        iters,
        full_cat_lst,
        outfile_prefix=outfile_prefix + ".refine",
    )


def _generate_categories_targetacc_augmented(
    theme, agent_info, history, iters, outfile_prefix="att1", acc_target="0.3--0.5"
):
    context = GENERATE_CATEGORIES_TARGETACC_AUGMENTED
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _generate_categories(
        theme, context, agent_info, history, iters, outfile_prefix=outfile_prefix
    )


def _generate_categories_random_augmented(
    theme, agent_info, history, iters, outfile_prefix="att1", acc_target="0.3--0.5"
):
    context = GENERATE_CATEGORIES_RANDOM_AUGMENTED
    context = context.replace("{ACC_TARGET}", str(acc_target))
    return _generate_categories(
        theme, context, agent_info, history, iters, outfile_prefix=outfile_prefix
    )


def _refine_categories(
    theme, context, agent_info, history, iters, candidate_lst, outfile_prefix="att1"
):
    if os.path.exists(f"{outfile_prefix}.categories.json"):
        print("FOUND categories.json")
        return json.load(open(f"{outfile_prefix}.categories.json", "r"))[0]
    agent_lm, agent_tokenizer, agent_client = agent_info
    context = context.replace("THEME", theme)
    if iters is None:
        iters = len(history) + 1
    if iters == 1:
        context += (
            "Please start with iteration 1."
            + "Here are the category candidates to select from (delimited by ||): "
            + " || ".join(candidate_lst)
            + "\n"
        )
    else:
        context += (
            "\n".join(history)
            + "Please start with iteration {}.".format(iters)
            + "Here are the category candidates to select from (delimited by ||): "
            + "||".join(candidate_lst)
            + "\n"
        )
    context = DEFAULT_JSON_MESSAGE + context
    # extract the json file from the message
    request_result = gen_from_prompt(
        model=agent_lm,
        tokenizer=agent_tokenizer,
        prompt=[context],
        echo_prompt=False,
        temperature=0.0,
        max_tokens=2000,
        process_func=None,
        service=agent_client,
        terminate_by_linebreak="no",
    )
    response = request_result.completions[0].text

    with open(
        f"{outfile_prefix}.full_thoughts.txt", "w", encoding="utf-8"
    ) as out_handle:
        out_handle.write(context)
        out_handle.write(
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )
        out_handle.write(response)

    extracted_json = extract_json_v2(response, f"{outfile_prefix}.categories.json")
    if len(extracted_json) == 1:
        extracted_json = extracted_json[0]
    return extracted_json


def _generate_categories(
    theme, context, agent_info, history, iters, outfile_prefix="att1"
):
    if os.path.exists(f"{outfile_prefix}.categories.json"):
        print("FOUND categories.json")
        return json.load(open(f"{outfile_prefix}.categories.json", "r"))[0]
    agent_lm, agent_tokenizer, agent_client = agent_info
    context = context.replace("THEME", theme)
    if iters is None:
        iters = len(history) + 1
    if iters == 1:
        context += "Please start with iteration 1."
    else:
        context += "\n".join(history) + "Please start with iteration {}.".format(iters)
    context = DEFAULT_JSON_MESSAGE + context
    # extract the json file from the message
    request_result = gen_from_prompt(
        model=agent_lm,
        tokenizer=agent_tokenizer,
        prompt=[context],
        echo_prompt=False,
        temperature=0.0,
        max_tokens=2000,
        process_func=None,
        service=agent_client,
        terminate_by_linebreak="no",
    )
    response = request_result.completions[0].text

    with open(
        f"{outfile_prefix}.full_thoughts.txt", "w", encoding="utf-8"
    ) as out_handle:
        out_handle.write(context)
        out_handle.write(
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )
        out_handle.write(response)

    extracted_json = extract_json_v2(response, f"{outfile_prefix}.categories.json")
    if len(extracted_json) == 1:
        extracted_json = extracted_json[0]
    return extracted_json


def _ask_question(theme, agent_info, history, iters, outfile_prefix="att1"):
    if os.path.exists(f"{outfile_prefix}.questions.json"):
        print("FOUND questions.json")
        return json.load(open(f"{outfile_prefix}.questions.json", "r"))
    agent_lm, agent_tokenizer, agent_client = agent_info
    context = ASK_QUESTION
    context = context.replace("THEME", theme)
    if iters is None:
        iters = len(history) + 1
    if iters == 1:
        context += "Please start with iteration 1."
    else:
        context += "\n".join(
            history
        ) + "Please start with iteration {}. Remember: Always output 30 questions, DO NOT just terminate directly".format(
            iters
        )
    context = DEFAULT_JSON_MESSAGE + context
    # extract the json file from the message
    request_result = gen_from_prompt(
        model=agent_lm,
        tokenizer=agent_tokenizer,
        prompt=[context],
        echo_prompt=False,
        temperature=0.0,
        max_tokens=2000,
        process_func=None,
        service=agent_client,
        terminate_by_linebreak="no",
    )
    response = request_result.completions[0].text

    extracted_json = extract_json_v2(response, f"{outfile_prefix}.questions.json")
    return extracted_json


def generate_dataset_without_docs(
    line_, agent_info, outfile_prefix, total_num_questions=50
):
    if os.path.exists(f"{outfile_prefix}.KI_questions.json"):
        print("found ", f"{outfile_prefix}.KI_questions.json")
        full_lst = []
        with open(f"{outfile_prefix}.KI_questions.json", "r") as f:
            for line in f:
                line = json.loads(line)
                full_lst.append(line)
        return full_lst

    f = open(f"{outfile_prefix}.KI_questions.json", "w")

    full_lst = []

    try:
        json_questions = gen_qa_without_docs(
            line_["category"], agent_info, line_["additional_requirement"]
        )
    except Exception as e:
        print(e)
        print("error in generating more questions, skipping...")
        print(f"generated {len(full_lst)} questions")

    for json_question in json_questions:
        line = copy.deepcopy(line_)
        line["question"] = json_question["question"]
        line["gold_answer"] = json_question["answer"]
        line["difficulty"] = json_question["difficulty"]
        line["wiki_entity"] = "None"
        full_lst.append(line)
        print(json.dumps(line), file=f)
    f.close()
    return full_lst


def process_paragraph_chunk(start_idx, end_idx, paragraph, agent_info, line_, generate_qa_func, wiki_entity):
    try:
        json_questions = generate_qa_func(
            paragraph[start_idx:end_idx],
            agent_info,
            line_["additional_requirement"],
        )
    except Exception as e:
        print(f"Error generating questions for chunk {start_idx}-{end_idx}: {e}")
        return []

    results = []
    for json_question in json_questions:
        line = copy.deepcopy(line_)
        line["question"] = json_question["question"]
        line["gold_answer"] = json_question["answer"]
        line["difficulty"] = json_question["difficulty"]
        line["wiki_entity"] = wiki_entity
        results.append(line)
    return results

def generate_long_questions(
    line_,
    agent_info,
    outfile_prefix,
    generate_qa_func=gen_qa_pairs_augmented,
    total_num_questions=50,
):
    if os.path.exists(f"{outfile_prefix}.KI_questions.json"):
        print("Found ", f"{outfile_prefix}.KI_questions.json")
        full_lst = []
        with open(f"{outfile_prefix}.KI_questions.json", "r") as f:
            for line in f:
                full_lst.append(json.loads(line))
        return full_lst

    paragraph, wiki_entity = search_step(line_["category"], output_more=True)
    print(len(paragraph), "length of paragraph")
    if len(paragraph) == 0:
        print("Empty paragraph, skipping...")
        return {}

    full_lst = []
    tasks = []

    with open(f"{outfile_prefix}.KI_questions.json", "w") as f:
        with ThreadPoolExecutor() as executor:
            for start_idx in range(0, len(paragraph), 20):
                if start_idx > total_num_questions:
                    break
                end_idx = min(start_idx + 20, len(paragraph))
                tasks.append(
                    executor.submit(
                        process_paragraph_chunk,
                        start_idx,
                        end_idx,
                        paragraph,
                        agent_info,
                        line_,
                        generate_qa_func,
                        wiki_entity,
                    )
                )

            for future in tqdm.tqdm(as_completed(tasks), total=len(tasks), desc='Processing paragraph chunk'):
                results = future.result()
                for line in results:
                    full_lst.append(line)
                    print(json.dumps(line), file=f)

    return full_lst


def saliency_rerank(json_lst, num_keep=5):
    for line_ in json_lst:
        page_title = line_["category"].replace(" ", "_")
        pageviews = get_pageviews(page_title)
        line_["salience"] = (
            pageviews if pageviews is not None else 0
        )  # add the pageviews to the line.
    # sort by the saliency
    json_lst = sorted(json_lst, key=lambda x: x["salience"], reverse=True)
    for line in json_lst:
        print(
            f'salience of {line["category"]}: ',
            round(line["salience"] / 1000000, 2),
            "M",
        )
    return json_lst[:num_keep]


def process_line_qa(line_, agent_info, outfile_prefix, historical_psg, generate_qa_func):
    paragraph, wiki_entity = search_step(line_["category"])
    if wiki_entity in historical_psg:
        print("Found repetitive wiki entity, skipping...", wiki_entity)
        return None, None

    if len(paragraph) == 0:
        print("Empty paragraph, skipping...")
        return None, None

    if "additional_requirement" not in line_:
        print("Missing additional requirement, skipping...")
        return None, None

    page_title = line_["category"].replace(" ", "_")
    pageviews = get_pageviews(page_title)
    line_["salience"] = pageviews if pageviews is not None else 0
    print(f"Salience of {page_title}: ", round(line_["salience"] / 1000000, 2), "M")

    try:
        json_questions = generate_qa_func(
            line_, agent_info, outfile_prefix + f"__{page_title}"
        )
    except Exception as e:
        print(f"Error generating questions for {page_title}: {e}")
        return None, None

    line_["paragraph"] = paragraph
    line_["wiki_entity"] = wiki_entity
    return line_, json_questions

def generate_full_qa(
    theme,
    agent_info,
    history,
    iters,
    outfile_prefix="att1",
    historical_psg=None,
    category_gen_func=_refine_categories_targetacc_augmented,
    generate_qa_func=generate_long_questions,
    acc_target=None,
    apply_saliency_rerank=True,
):
    if os.path.exists(f"{outfile_prefix}.KI_questions.json"):
        print("FOUND KI_questions.json")
        return

    if acc_target is not None:
        json_category = category_gen_func(
            theme,
            agent_info,
            history,
            iters,
            outfile_prefix=outfile_prefix,
            acc_target=acc_target,
        )
    else:
        json_category = category_gen_func(
            theme, agent_info, history, iters, outfile_prefix=outfile_prefix
        )

    if apply_saliency_rerank:
        json_category = saliency_rerank(json_category, 5)

    full_lst = []
    historical_psg = historical_psg or []

    with ThreadPoolExecutor() as executor:
        tasks = [
            executor.submit(
                process_line_qa,
                line_,
                agent_info,
                outfile_prefix,
                historical_psg,
                generate_qa_func,
            )
            for line_ in json_category
        ]

        for future in tqdm.tqdm(as_completed(tasks), total=len(tasks), desc='Processing line for QA'):
            result = future.result()
            if result is None:
                continue
            line_, json_questions = result
            if line_:
                historical_psg.append(line_["wiki_entity"])
            if json_questions:
                full_lst.extend(json_questions)

    with open(f"{outfile_prefix}.KI_questions.json", "w") as f:
        json.dump(full_lst, f)

    with open(f"{outfile_prefix}.categories_augmented.json", "w") as f:
        json.dump(json_category, f)

    return historical_psg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument(
        "--test_taker_modelname", default="gpt-3.5-turbo"
    )
    parser.add_argument(
        "--test_taker_modelname2", default=None
    )
    parser.add_argument(
        "--agent_modelname", default="gpt-4-turbo-preview"
    )
    parser.add_argument("--tool_modelname", default=None)
    parser.add_argument(
        "--temperature", type=float, default=0.001
    )
    parser.add_argument(
        "--pairwise", type=str, default="no"
    )
    parser.add_argument(
        "--exp_mode", type=str, default="ki_wiki"
    )
    parser.add_argument(
        "--theme", type=str, default="history"
    )
    parser.add_argument(
        "--use_helm", type=str, default="yes"
    )
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--acc_target", type=str, default="0.1--0.3"
    )
    parser.add_argument("--num_iters", type=int, default=8)  # option that

    parser.add_argument(
        "--outfile_prefix1", type=str, default="att1"
    )

    args2 = parser.parse_args()
    args = copy.deepcopy(args2)

    if args.use_helm == "yes":  # use the helm model server.
        test_taker_info = helm_process_args(args.test_taker_modelname)
        print("loaded helm models")
    else:
        test_taker_lm, test_taker_tokenizer, modelpath_name, test_taker_client = (
            process_args_for_models(args.test_taker_modelname)
        )
        test_taker_info = (test_taker_lm, test_taker_tokenizer, test_taker_client)

    agent_lm, agent_tokenizer, agent_name, agent_client = process_args_for_models(
        args.agent_modelname
    )

    if args.tool_modelname is None:
        tool_lm, tool_tokenizer, tool_name, tool_client = (
            agent_lm,
            agent_tokenizer,
            agent_name,
            agent_client,
        )
    else:
        tool_lm, tool_tokenizer, tool_name, tool_client = process_args_for_models(
            args.tool_modelname
        )

    evaluator_info = (tool_lm, tool_tokenizer, tool_client)
    agent_info = (agent_lm, agent_tokenizer, agent_client)  # agent model

    if args.exp_mode == "autobencher":
        history_dict = []
        historical_psg = []
        for iters in range(args.num_iters):
            args.outfile_prefix = args.outfile_prefix1 + str(iters + 1)
            summarized_content = summarize_over_history(
                history_dict, gold_key="gold_answer", verbose=False
            )
            history = [summarized_content]
            historical_psg = generate_full_qa(
                args.theme,
                agent_info,
                history,
                iters + 1,
                outfile_prefix=args.outfile_prefix,
                historical_psg=historical_psg,
                category_gen_func=_refine_categories_targetacc_augmented,
                generate_qa_func=generate_long_questions,
                acc_target=args.acc_target,
            )
            with open(f"{args.outfile_prefix}.KI_questions.json", "r") as f:
                json_category = json.load(f)
            if len(json_category) == 1:  # remove the outer embedded list.
                json_category = json_category[0]
            gold_answer_json = copy.deepcopy(json_category)
            json_dict = solve_and_compare_questions(
                test_taker_info,
                evaluator_info,
                json_category,
                gold_answer_json,
                args.outfile_prefix,
                "gold_answer",
            )
            history_dict.append(json_dict)

            verbose_description = get_summary_of_results(json_dict, verbose=False)
            print(verbose_description)

    elif args.exp_mode == "naive_baseline":
        """
        This is the most naive version of AutoBencher,
        There is no previleged information (wiki) and no adaptive search.
        """
        history_dict = []
        for iters in range(args.num_iters):
            args.outfile_prefix = args.outfile_prefix1 + str(iters + 1)
            summarized_content = summarize_over_history(
                history_dict, gold_key="gold_answer"
            )
            history = [summarized_content]

            json_category = _ask_question(
                args.theme,
                agent_info,
                history,
                iters + 1,
                outfile_prefix=args.outfile_prefix,
            )
            gold_answer_json = copy.deepcopy(json_category[0])

            json_dict = solve_and_compare_questions(
                test_taker_info,
                agent_info,
                json_category,
                gold_answer_json,
                args.outfile_prefix,
                "answer",
            )
            history_dict.append(json_dict)

    elif args.exp_mode == "baseline_without_privileged_info":
        """
        The topic proposal component is the same as AutoBencher, meaning that this uses adaptive search.
        The dataset generation component does not use privileged information.
        """
        history_dict = []
        historical_psg = []
        for iters in range(args.num_iters):
            args.outfile_prefix = args.outfile_prefix1 + str(iters + 1)
            summarized_content = summarize_over_history(
                history_dict, gold_key="gold_answer", verbose=False
            )
            history = [summarized_content]

            historical_psg = generate_full_qa(
                args.theme,
                agent_info,
                history,
                iters + 1,
                outfile_prefix=args.outfile_prefix,
                historical_psg=historical_psg,
                category_gen_func=_refine_categories_targetacc_augmented,
                generate_qa_func=generate_dataset_without_docs,
                apply_saliency_rerank=True,
            )
            with open(f"{args.outfile_prefix}.KI_questions.json", "r") as f:
                json_category = json.load(f)
            if len(json_category) == 1:
                json_category = json_category[0]
            gold_answer_json = copy.deepcopy(json_category)
            json_dict = solve_and_compare_questions(
                test_taker_info,
                agent_info,
                json_category,
                gold_answer_json,
                args.outfile_prefix,
                "gold_answer",
            )
            history_dict.append(json_dict)

    elif args.exp_mode == "baseline_without_adaptive_search":
        """
        This baseline do not use adaptive search, but it uses privileged information.
        """
        history_dict = []
        historical_psg = []
        for iters in range(args.num_iters):
            args.outfile_prefix = args.outfile_prefix1 + str(iters + 1)
            summarized_content = summarize_over_history(
                history_dict, gold_key="gold_answer", verbose=False
            )
            history = [summarized_content]
            historical_psg = generate_full_qa(
                args.theme,
                agent_info,
                history,
                iters + 1,
                outfile_prefix=args.outfile_prefix,
                historical_psg=historical_psg,
                category_gen_func=_refine_categories_random_augmented,
                generate_qa_func=generate_long_questions,
                apply_saliency_rerank=False,
            )
            with open(f"{args.outfile_prefix}.KI_questions.json", "r") as f:
                json_category = json.load(f)
            if len(json_category) == 1:
                json_category = json_category[0]
            gold_answer_json = copy.deepcopy(json_category)
            json_dict = solve_and_compare_questions(
                test_taker_info,
                agent_info,
                json_category,
                gold_answer_json,
                args.outfile_prefix,
                "gold_answer",
            )
            history_dict.append(json_dict)
