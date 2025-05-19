DEFAULT_JSON_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your reasoning and language skills.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
Reply "TERMINATE" in the end when everything is done.
"""

COMPARE_ANSWERS_STR = """Your goal is to compare the prediction with the gold answer, and judge the correctness of the prediction. 
We'd still consider the prediction to be correct if 
1. the prediction is semantically the same as the gold answer: formating or different way of reference shouldn't affect correctness. For example, if the gold answer is Jan 21, and the test taker output is 01/21, we would still consider the prediction to be correct. For example, United States and USA refer to the same entity.
2. the prediction refers a broader entity that contains the gold answer. For example, if the gold answer is Beijing, and the test taker output is Asia, we will then consider correctness based on the question.
3. If the question is slightly ambiguous, such that there are multiple correct answers: For example, if the question asks for reasons why something happens, and it could be caused by multiple reasons, we will consider the prediction to be correct if the prediction contains one of the correct answers.

You should output a short and succinct reasoning for the your correctness prediction. Then, you should output delimiter "##" and output "true" if the prediction is correct, and "false" if the prediction is incorrect.
Example Format: 
Question: What is 1+1?
pred=2 || gold=2.0
reason: identical numbers ## true
"""

GEN_QA_WITHOUT_DOCS_CONTEXT = """You will generate a few question and answer pairs on the topic: {{TOPIC}}
Make sure not to ask subjective questions, and let the question's correct answer be a concise short phrase. 
Make the answer concise. 

You will also receive additional requirements on the questions. You should follow these additional requirements when generating the questions.
For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.

Try to generate a diverse set of 50 questions, and make sure that the questions are not too similar to each other while satisfying the additional requirements. If you can't generate 50 questions, generate as many as you can.

Formatting: 
Each question should be a dictionary with the following keys: id, question, answer, estimated difficulty. 
The questions should be exactly in the following format (a list of dictionaries): 
```json
{"id": "1", "question": "<question>", "answer": "<answer>", "difficulty": "1"}, 
{"id": "2", "question": "<question>", "answer": "<answer>", "difficulty": "1"}, 
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets. 
If you are generating double quotes as content of <question> or <answer>, make sure to escape them with a backslash. 
    """

QA_PAIRS_AUGMENTED_CONTEXT = """Conditioned on the wikipedia paragraph, you will generate a few question and answer pairs. 
Make sure not to ask subjective questions, and let the question's correct answer be a concise short phrase. 
Make sure that the question you selected is answerable by the given wikipedia paragraph, and make the answer concise. It's recommended to use the exact text from the paragraph as answers.
Make sure that the questions are also answerable by an expert **without the wikipedia paragraph**. For example, dont ask questions that are too specific to the paragraph, like "what are the three locations mentioned in the paragraph?". Or "who's the most famous soldier, according to the paragraph?".

You will also receive additional requirements on the questions. You should follow these additional requirements when generating the questions.
For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.

Try to generate a diverse set of 15 questions, and make sure that the questions are not too similar to each other while satisfying the additional requirements. If you can't generate 15 questions, generate as many as you can.

Formatting: 
Each question should be a dictionary with the following keys: id, question, answer, estimated difficulty. 
The questions should be exactly in the following format (a list of dictionaries): 
```json
{"id": "1", "question": "<question>", "answer": "<answer>", "difficulty": "1"}, 
{"id": "2", "question": "<question>", "answer": "<answer>", "difficulty": "1"}, 
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets. 
If you are generating double quotes as content of <question> or <answer>, make sure to escape them with a backslash. 
"""

REFINE_CATEGORIES_RANDOM_AUGMENTED_CONTEXT = """ Your goal is to select from a list of categories for knowledge intensive questions so that the selected subset are not repetitive from prior selectioins and covers a wide range of topics that are important.
The categories should be selected based on three criteria: (1) aligned with THEME, (2) salient and cover important topics.
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.

Output Formatting: 
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement. 
Make sure the categories are similar to wikipedia categories. 
The categories should be exactly in the following format (a list of dictionaries): 
```json 
[
{"id": "1", "category": "Ancient Philosophers", "parent_category": "History", "additional_requirement": "only ask about famous people and their ideologies"}, 
{"id": "2", "category": "Second World War", "parent_category": "History", "additional_requirement": "major battles"}, 
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration: 
The goal is to find a set of categories that have broad coverage of topics and are not repetitive from prior selections. 

At every iteration, you are given a list of categories that you have already explored and their respective accuracy. Also, you are given a larger set of candidate categories for this iteration, and you should use the information from previous iterations to select the top 10 categories from the list. 
DO NOT REPEAT any of the categories that you have already explored.
"""

REFINE_CATEGORIES_TARGETACC_AUGMENTED = """ Your goal is to select from a list of categories for knowledge intensive questions so that the selected subset are likely to achieve the target accuracy of {ACC_TARGET}.
The categories should be selected based on three criteria: (1) aligned with THEME, (2) likely to obtain the target accuracy of {ACC_TARGET}, you can judge this based on the accuracy statistics from previous iterations. and (3) salient and cover important topics.
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.

Output Formatting: 
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement. 
Make sure the categories are similar to wikipedia categories. 
The categories should be exactly in the following format (a list of dictionaries): 
```json 
[
{"id": "1", "category": "Ancient Philosophers", "parent_category": "History", "additional_requirement": "only ask about famous people and their ideologies"}, 
{"id": "2", "category": "Second World War", "parent_category": "History", "additional_requirement": "major battles"}, 
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration: 
The goal is to find a set of categories that with accuracy close to the target accuracy level of {ACC_TARGET}. 

At every iteration, you are given a list of categories that you have already explored and their respective accuracy. Also, you are given a larger set of candidate categories for this iteration, and you should use the information from previous iterations to select the top 10 categories from the list, that are most likely to achieve the target accuracy level, while still being relevant and salient. 
In later iterations you should receive as input the categories that you have already explored and their respective accuracy. You should
DO NOT REPEAT any of the categories that you have already explored.
"""

GENERATE_CATEGORIES_TARGETACC_AUGMENTED = """ Your goal is to come up with a list of categories for knowledge intensive questions that achieve the target accuracy of {ACC_TARGET}.
The categories should be diverse and cover important topics, under the theme of THEME. 
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.
Constructing the categories is like building a tree structure of history, and (category, parent_category) is like specifying a node and its parent. We should select the most precise parent category, for example if you are trying to expand the category "second world war" to make it more specific by adding the node "famous battles in second world war", you should specify the parent category as "second world war" instead of "history".

Output Formatting: 
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement. 
Make sure the categories are similar to wikipedia categories. 
The categories should be exactly in the following format (a list of dictionaries): 
```json 
[
{"id": "1", "category": "Ancient Philosophers", "parent_category": "History", "additional_requirement": "only ask about famous people and their ideologies"}, 
{"id": "2", "category": "Second World War", "parent_category": "History", "additional_requirement": "major battles"}, 
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration: 
The goal is to find a set of categories that with accuracy close to the target accuracy level of {ACC_TARGET}. 

For iteration 1, you can start with a wide variety of categories for us to build upon later. 
In later iterations you should receive as input the categories that you have already explored and their respective accuracy. You should
1. Think about breadth. Brainstorm questions with different categories to have broader coverage. Coming up with new categories that can are likely to achieve the target accuracy level.
2. For example, If you find the model now lacks categories of 0.3 -- 0.5 accuracy, you should come up with more categories that would yield accuracy in that range, by either reducing the difficulty of questions that achieve lower accuracy (via subcategory or via additional requirement), or increasing the difficulty of questions that achieve higher accuracy.
3. DO NOT REPEAT any of the categories that you have already explored.
"""


GENERATE_CATEGORIES_RANDOM_AUGMENTED = """ Your goal is to come up with a list of categories for knowledge intensive questions that have broad coverage and are salient. 
The categories should be diverse and cover important topics, under the theme of THEME. 
You can also specify some additional requirements for each category. This additional requirement will be passed to the question asker, and this helps with controlling the contents of the question and modulate their difficulties. For example, "only ask about major events in the paragraph, and avoid niched events". That way, you should only ask questions about major events in the paragraph, which is one way to make the questions easier.
Constructing the categories is like building a tree structure of history, and (category, parent_category) is like specifying a node and its parent. We should select the most precise parent category, for example if you are trying to expand the category "second world war" to make it more specific by adding the node "famous battles in second world war", you should specify the parent category as "second world war" instead of "history".

Output Formatting: 
Each category should be a dictionary with the following keys: id, category, parent_category, additional_requirement. 
Make sure the categories are similar to wikipedia categories. 
The categories should be exactly in the following format (a list of dictionaries): 
```json 
[
{"id": "1", "category": "Ancient Philosophers", "parent_category": "History", "additional_requirement": "only ask about famous people and their ideologies"}, 
{"id": "2", "category": "Second World War", "parent_category": "History", "additional_requirement": "major battles"}, 
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


Iteration: 
The goal is to find a set of categories that have broad coverage of topics and are salient. 

For iteration 1, you can start with a wide variety of categories for us to build upon later. 
In later iterations you should receive as input the categories that you have already explored and their respective accuracy. You should
1. Think about breadth. Brainstorm questions with different categories to have broader coverage.
2. DO NOT REPEAT any of the categories that you have already explored.
"""

ASK_QUESTION = """
Your goal is to comprehensively evaluate the knowledge of a language model. 
In each iteration, you should output 30 ** knowledge-intensive ** questions of different categories, and write these questions in a json file. All the question need to pertain to the them of THEME.

To generate a question, you will follow the following steps: 
1. Come up with a category (e.g. physical phenomenon, capitals of countries, etc.)
2. This category will be used to search for a wikipedia page.
3. Conditioned on the wikipedia paragraph, you will generate a question and answer pair. Make sure that the question you selected is answerable by the given wikipedia paragraph.

Output formatting: 
Each question should be a dictionary with the following keys: id, question, answer, category, estimated difficulty.
Note: do not come up with repetitive questions. If you have asked a question, do not ask it again! 
Come up with 30 concrete questions, and write them in the following format. It's helpful to first come up with a plan for this iteration, and then write the questions.
The questions should be exactly in the following format (a list of dictionaries): 
```json
[
{"id": "1", "question": "What's the tallest mountain in the world?", "answer": "Mount Everest", "category": "Geography", "difficulty": "1"}, 
{"id": "2", "question": "Who discovered green tea", "answer":"Shen Nong", "category": "History", "difficulty": "1"},
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.

Iteration: 
The goal is to search for a category of questions that the language model is weak at. 
For iteration 1, you can start with questions of different categories, and start with a difficulty level of 1-2. Make sure the questions that you come up with are concrete questions that has a concrete solution, not just place holders, and come up with 30 questions. Do not leave place holders.  
In later iterations you should 
1. Think about breadth. Brainstorm questions with different categories if there are missing categories to make the evaluation more comprehensive and have broad coverage. 
2. For the categories that the model is strong at, increase the difficulty level of the questions. 
3. For the categories that the model is weak at, try to probe for diverse types of failure modes. Remember the goal is to get a comprehensive evaluation of the model. We want to know all the failure modes of the model, and all its strength.  
"""