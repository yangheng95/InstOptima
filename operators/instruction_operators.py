import json
import random
import requests
import time
from pyabsa import TextClassification as TC
from termcolor import colored

from entity.instruction import Instruction


def extract_inst(text):
    if "<inst>" not in text or "</inst>" not in text:
        return text
    start = text.find("<inst>") + 6  # 找到第一个 '{' 的位置
    end = text.rfind("</inst>")  # 找到最后一个 '}' 的位置
    return text[start:end]  # 提取字典部分的文本


class InstructOperator(object):
    def __init__(self, checkpoint=None):
        self.params = {"prompt": None, "message_history": None, "password": "Password!"}
        self.chatter_url = "https://chatter.pagekite.me/chat"

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
        self.message_history = [
            {
                "role": "user",
                "content": "The diversity and quality of instruction fine-tuning are very important, which has motivated me to propose an evolutionary multi-objective instruction optimization framework. Consequently, I need you to serve as instruction operators, such as definition mutation and example substitutions in an instruction. ",
            },
            {
                "role": "assistant",
                "content": "As an AI language model, I can assist you in various ways, including serving as an instruction operator within your proposed framework. I can help with tasks such as definition mutation and example substitutions in an instruction. Please provide me with more specific details about the instructions you would like to optimize, and I'll do my best to assist you.",
            },
        ]

    def chat(self, prompt, **kwargs):
        self.params.update(
            {
                "prompt": prompt,
                "system_prompt": "The diversity and quality of instruction fine-tuning are very important, which has motivated me to propose an evolutionary multi-objective instruction optimization framework. Consequently, I need you to serve as instruction operators, such as definition mutation and example substitutions in an instruction. ",
                "message_history": str(self.message_history),
                "tag": "EvoInst-" + kwargs.get("dataset", "default"),
            }
        )
        try:
            response = requests.post(self.chatter_url, params=self.params)
            # self.message_history = response.json()["message_history"]
            if "error" in response.json():
                print(response.json())
                time.sleep(5)
                return self.chat(prompt)
            if response.status_code != 200:
                print(response.status_code)
                print(response.text)
                time.sleep(5)
                return self.chat(prompt)
            return response.json()["response"]
        except Exception as e:
            print(e)
            time.sleep(5)
            return self.chat(prompt)

    def unary_operate(self, inst_individual, operation, **kwargs):
        _unary_operate_prompt = """I will give you an instruction and the operation to be performed on it.
Here is an instruction to be evolved: "{}", and its minimisation objective values are "{}". 
The steps to operate the instruction: "{}" (random seed: "{}"). The steps are for your reference only, do not include them in your response.
please warp the new instruction with <inst> and </inst> tags, your cannot say anything else except for the new instruction: 
""".format(
            inst_individual.definition,
            inst_individual.objectives,
            operation["steps"],
            random.randint(0, 100000),
        )
        response = self.chat(_unary_operate_prompt, **kwargs)
        if response.count('<inst>') > 1:
            inst = response.partition("\n\n")[2]
        else:
            inst = response
        new_definition = extract_inst(inst).strip()

        if not new_definition:
            new_definition = response

        return Instruction(
            definition=new_definition, example=inst_individual.example, **kwargs
        )

    def evolve(self, instruction_individual, instruction_individual2=None, **kwargs):
        random.seed(random.randint(0, 100000))
        op_name, operation = random.choice(list(self.operations.items()))
        print("Operation:", op_name)
        instruction_individual = self.unary_operate(
            instruction_individual, operation, **kwargs
        )
        if self.pareto_classifier:
            cls_res = self.pareto_classifier.predict(
                json.dumps(
                    {"instruction": str(instruction_individual), "label": "pareto"}
                )
                .replace('{"instruction": "', "")
                .replace('", "label": "', "$LABEL$")
                .replace('"}', ""),
                print_result=False,
                auto_device=True,
            )

            if cls_res["label"] != "pareto":
                print(
                    colored("Un-pareto Instruction: \n", "red"), instruction_individual
                )
                return self.evolve(
                    instruction_individual,
                    instruction_individual2=instruction_individual2,
                    **kwargs
                )
            else:
                print(
                    colored("Pareto Instruction: \n", "green"), instruction_individual
                )
                return instruction_individual

        print("Current Instruction:\n", instruction_individual)
        return instruction_individual
