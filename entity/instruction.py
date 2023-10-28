from models.aspect_based_sentiment_analysis.instruction import APCInstruction
from models.generative_summarization.instruction import SumInstruction
from models.natural_language_inference.instruction import NLIInstruction
from models.text_classification.instruction import TCInstruction


class Instruction(object):
    def __init__(self, definition=None, example=None, **kwargs):
        self.objectives = "Uncalculated"
        if definition is not None and example is not None:
            definition = str(definition)
            example = str(example)
        if kwargs.get("dataset") in ["SST2", "Amazon", "AgNews"]:
            if definition is not None and example is not None:
                self.definition, self.example = definition, example
                self.prompt = TCInstruction(definition, example)
            else:
                instruction = TCInstruction()
                self.definition, self.example = (
                    instruction.bos_instruction,
                    instruction.example,
                )
                self.prompt = instruction

        elif kwargs.get("dataset") in [
            "toy",
            "Laptop14",
            "Restaurant14",
            "Restaurant15",
            "Restaurant16",
        ]:
            if definition is not None and example is not None:
                self.definition, self.example = definition, example
                self.prompt = APCInstruction(definition, example)
            else:
                instruction = APCInstruction()
                self.definition, self.example = (
                    instruction.bos_instruction,
                    instruction.example,
                )
                self.prompt = instruction
        elif kwargs.get("dataset") in ["SNLI", "MNLI"]:
            if definition is not None and example is not None:
                self.definition, self.example = definition, example
                self.prompt = NLIInstruction(definition, example)
            else:
                instruction = NLIInstruction()
                self.definition, self.example = (
                    instruction.bos_instruction,
                    instruction.example,
                )
                self.prompt = instruction
        elif kwargs.get("dataset") in ["Gigaword"]:
            if definition is not None and example is not None:
                self.definition, self.example = definition, example
                self.prompt = NLIInstruction(definition, example)
            else:
                instruction = SumInstruction()
                self.definition, self.example = (
                    instruction.bos_instruction,
                    instruction.example,
                )
                self.prompt = instruction
        else:
            raise ValueError("dataset not assigned or not supported")

    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, item):
        return self.prompt[item]

    def __str__(self):
        return self.prompt.__str__()

    def __repr__(self):
        return self.prompt.__repr__()


if __name__ == "__main__":
    p = Instruction(None)
    print(p)
