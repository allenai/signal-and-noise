DEFAULT_JSON_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your reasoning and language skills.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
Reply "TERMINATE" in the end when everything is done.
"""

GENERATE_PYTHON_ANSWERS_CONTEXT = """Your goal is to generate the python code that solves the provided math questions. 
The provided math problems are in the format of json. You should write the python code for all the problems in python coding blocks. You should NOT need to write a large solve function that can solve all the problems and handle all cases, just provide problem specific solution for each question. 
Do not use string matching to figure out which problem to solve. Just do as follows, write the expression for each question inline, then print the answer. 
```python
ans1 = 1+1
print("1. ANSWER:", ans1)
ans2 = np.sin(1)
print("2. ANSWER:", ans2)
``` 
Also, make sure to answer questions **sequentially**, in the same order as question id. If there is a question that's not answerable, set the answer to be "N/A".
*You should not omit any questions, and DO NOT use ellipsis*
For questions with a numeric answer, make sure to simplify the final result and report the final answer as a decimal number. For example, for a sympy expression, you should simplify it and report the final answer as a decimal number via evalf(). But note that 'float' object has no attribute 'evalf'. 
If you are solving a trigonometric equation using sympy, make sure to use sympy.sin(x) for variable x. Do not use math.sin(x) or np.sin(x), which cause errors. Also, remember that sympy.sin takes in radians, not degrees. So you want to judge the unit and then convert degrees into radians. 
Implement error handling for each function, if there is an error (in input or in execution), output "N/A" for that question.
The python code should print the answer of each problem in the following format: 
1. ANSWER: <answer1>
2. ANSWER: <answer2>
3. ANSWER: <answer3>
...
"""

CATEGORY_GENERATION_CONTEXT = """ Your goal is to come up with comprehensive categories of math questions, and then efficiently find the subcategories within that achieve close to AIM accuracy, for each category of math questions. 
In each iteration, you should come up with a plan towards generating questions, and write the plan in a json file. 

To generate the plan, you should first come up with the set of categories to expand.
Then, for each category (e.g., arithmetic), you should expand it with a few subcategory description that's detailed, precise and concrete (e.g., two-digit addition of two terms). 
Note that you should design carefully to efficiently find the subcategories within that achieve close to AIM accuracy. For example, don't get stuck in questions that are overall too simple, and you should first broadly explore various difficulties levels, and gradually pin down the subcategories that are close to AIM accuracy.
Make sure that the subcategory you generated are concrete such that it allows some LM to generate around 50 math questions that instantiate this subcategory. 

# Output format:
Each question should be a dictionary with the following keys: id, question, category, difficulty levels (1-10).
The questions should be exactly in the following format (a list of dictionaries): 
```json
[
{"id": "1", "category": "Arithmetic", "subcategory_description": "single-digit addition of two terms", "difficulty": "1"}, 
{"id": "2", "category": "Arithmetic", "subcategory_description": "single-digit subtraction of two terms", "difficulty": "1"},
...
]
``` 
Do not use python code block. Write concrete categories and subcategories. Do not leave place holders or ellipsis.
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.


The goal is to search for (category, subcategory_description) pair that achieves AIM accuracy. You do not need to answer the questions. 
For iteration 1, you can start with a variety of different categories. 
Make sure the (category, subcategory_description) that you come up with are concrete, not just place holders, and come up with 5 (category, subcategory_description) pairs.

In later iterations you should 
1. Think about breadth. Brainstorm more categories, if there are missing categories to make the evaluation more comprehensive and have broader coverage. 
2. Adjust the difficulty level of the subcategories to match the AIM accuracy. If all the subcategories tend to achieve accuracy greater than AIM, generate subcategory description with increased difficulties.
3. If all the subcategories tend to achieve lower accuracy than AIM, make the subcategory description easier. 


Note: do not come up with repetitive subcategories. 
It's helpful to first come up with a plan for this iteration, and then write the questions.

"""

DESCRIPTION_GENERATION_CONTEXT = """Your goal is to come up with math questions that match the description. 
In each iteration, you receive as input a subcategory_description that describes the types of question to ask.  
You should come up with around 50 math questions that matches the subcategory description, and write these questions in a json file.
Each question should be a dictionary with the following keys: id, question, category, answer, difficulty levels (1-10).
You do not need to answer the questions. 

Note: do not come up with repetitive questions. If you have asked a question, do not ask it again! 
Come up with 100 concrete questions, and write them in the following format. 
Do not leave place holders or ellipsis!!!
It's helpful to first come up with a plan for this iteration, and then write the questions.
The questions should be exactly in the following format (a list of dictionaries): 
```json
[
{"id": "1", "question": "What is 5 + 3?", "category": "Arithmetic", "answer": "8", "difficulty": "1"}, 
{"id": "2", "question": "What is 5 - 3?", "category": "Arithmetic", "answer": "2", "difficulty": "1"},
...
]
``` 
Do not use python code block. 
Make sure that you generate a valid json block (surrounded by ```json [...] ```). Surrounded by the [] brackets.
"""