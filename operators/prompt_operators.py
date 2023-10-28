import json
import random
import requests
import time
from pyabsa import TextClassification as TC
from termcolor import colored


def extract_dict(text):
    start = text.find("{")  # 找到第一个 '{' 的位置
    end = text.rfind("}")  # 找到最后一个 '}' 的位置
    if start != -1 and end != -1 and start < end:
        dict_text = text[start : end + 1]  # 提取字典部分的文本
        try:
            dictionary = eval(dict_text)  # 使用 eval 将文本转换为字典对象
            return dictionary
        except SyntaxError as e:
            print(e)
    return None


class PromptOperator(object):
    def __init__(self, checkpoint=None):
        self.params = {"prompt": None, "message_history": None, "password": "Password!"}
        # self.chatter_url = "https://chatter.pagekite.me/chat"
        self.chatter_url = "http://localhost:6789/chat"

        self.message_history = []

        if checkpoint:
            self.pareto_classifier = TC.TextClassifier(checkpoint=checkpoint)
        else:
            self.pareto_classifier = None

        self.operations = {
            "Definition Mutation": {
                "steps": "1. Take a definition as input.\n"
                "2. Mutate the definition by replacing a random word with a synonym.\n"
                "3. Output the mutated definition.",
                "description": "Mutate the definition by replacing a random word with a synonym.",
                "output": "Output the mutated definition.",
            },
            "Sentence Rearrangement": {
                "steps": "1. Take a sentence as input.\n"
                "2. Rearrange the words in the sentence to create a new sentence.\n"
                "3. Output the new sentence.",
                "description": "Rearrange the words in the sentence to create a new sentence.",
                "output": "Output the new sentence.",
            },
            "Instruction Clarification": {
                "steps": "1. Take a set of instructions as input.\n"
                "2. Identify any ambiguous or unclear parts of the instructions.\n"
                "3. Provide a clarification or ask a question to resolve the ambiguity.\n"
                "4. Output the clarification or question.",
                "description": "Identify any ambiguous or unclear parts of the instructions.",
                "output": "Provide a clarification or ask a question to resolve the ambiguity.",
            },
            "Error Correction": {
                "steps": "1. Take a sentence with grammatical errors as input.\n"
                "2. Identify the errors in the sentence.\n"
                "3. Correct the errors and output the corrected sentence.",
                "description": "Identify the errors in the sentence and correct them.",
                "output": "Output the corrected sentence.",
            },
            "Language Simplification": {
                "steps": "1. Take a complex sentence as input.\n"
                "2. Simplify the language in the sentence while preserving the meaning.\n"
                "3. Output the simplified sentence.",
                "description": "Simplify the language in the sentence while preserving the meaning.",
                "output": "Output the simplified sentence.",
            },
            "New Definition Generation": {
                "steps": "1. Take a concept or term related to the target task as input.\n"
                "2. Generate a new definition for the concept or term.\n"
                "3. Ensure that the new definition accurately captures the essence of the concept or term.\n"
                "4. Output the newly generated definition.",
                "description": "Generate a new definition for the concept or term.",
                "output": "Output the newly generated definition.",
            },
        }

    def chat(self, prompt, **kwargs):
        self.params.update(
            {
                "prompt": prompt,
                "system_prompt": "The diversity and quality of instruction fine-tuning are very important, which has motivated me to propose an evolutionary multi-objective instruction optimization framework. ",
                "message_history": kwargs.get("message_history", '[]'),
                "tag": "EvoPrompt-" + kwargs.get("dataset", "default"),
            }
        )
        try:
            response = requests.post(self.chatter_url, params=self.params, timeout=600)

            if "error" in response.json():
                print(response.json())
                time.sleep(5)
                return self.chat(prompt)
            if response.status_code != 200:
                print(response.status_code)
                print(response.text)
                time.sleep(5)
                return self.chat(prompt)
            return response.json()["response"], response.json()["message_history"]
        except Exception as e:
            print(e)
            time.sleep(5)
            return self.chat(prompt)

    def evolve(self, operators, objectives, **kwargs):
        for k, v in operators.items():
            operators[k]["steps"] = "tl;dr"
        print("Operators:", operators)
        _unary_operate_prompt = """Your are a langauge expert. In the next generation,
and you are evolving the instruction operators based on the population-aggregated minimisation objective values,
Please try to create new operators, and recombine the set of the original and new operators .
<Instruction Operators>: {} 
{}
Please only output all (not incremental) the new operations (with detailed steps instead of tl;dr) as a python DICT type ==>
""".format(
            operators,
            objectives,
        )
        try:
            print("Original Operators:", operators)
            _response, message_history = self.chat(
                _unary_operate_prompt,
                message_history=str(self.message_history[-3:]),
                **kwargs
            )
            # _response, message_history = self.chat(_unary_operate_prompt, message_history=None, **kwargs)
            print("Response:", _response)
            new_operators = extract_dict(_response)
            print("New Operators:", new_operators)
            if new_operators:
                self.message_history = message_history
                self.message_history.pop()
                return new_operators
            else:
                return self.evolve(operators, objectives, **kwargs)
        except Exception as e:
            print(e)
            return self.evolve(operators, objectives, **kwargs)

    def random_operate(self, prompt, prompt2=None, **kwargs):
        random.seed(random.randint(0, 100000))
        op_name, operation = random.choice(list(self.operations.items()))
        print("Operation:", op_name)
        prompt = self.evolve(prompt, operation, **kwargs)
        if self.pareto_classifier:
            cls_res = self.pareto_classifier.predict(
                json.dumps({"prompt": str(prompt), "label": "pareto"})
                .replace('{"prompt": "', "")
                .replace('", "label": "', "$LABEL$")
                .replace('"}', ""),
                print_result=False,
                auto_device=True,
            )

            if cls_res["label"] != "pareto":
                print(colored("Un-pareto Prompt: \n", "red"), prompt)
                return self.random_operate(prompt, prompt2=prompt2, **kwargs)
            else:
                print(colored("Pareto Prompt: \n", "green"), prompt)
                return prompt

        print("Current Prompt:\n", prompt)
        return prompt
