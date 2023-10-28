class Instruction:
    def __init__(self, bos_instruction=None, eos_instruction=None):
        self.bos_instruction = bos_instruction
        self.eos_instruction = eos_instruction

    def set_instruction(self, bos_instruction, eos_instruction):
        self.bos_instruction = bos_instruction
        self.eos_instruction = eos_instruction

    def get_instruction(self):
        return self.bos_instruction, self.eos_instruction


class NLIInstruction(Instruction):
    def __init__(
        self, bos_instruction=None, example=None, eos_instruction="Your judgment is:"
    ):
        super().__init__(bos_instruction, eos_instruction)
        self.eos_instruction = eos_instruction

        if bos_instruction is None:
            bos_instruction = """Definition: The inputs are pairs of sentences. The task is to identify the relationship between the second sentence and the first sentence, choosing from "entailment", "contradiction", or "neutral". Here are some examples:"""

        if example is None:
            example = f"""
input: "John is a teacher. He teaches math in a high school." "He works in education."
{self.eos_instruction} entailment

input: "The cat is sleeping on the couch." "There is a dog running in the yard."
{self.eos_instruction} neutral
"""

        self.bos_instruction = bos_instruction
        self.example = example

    def __str__(self):
        return (
            self.bos_instruction
            + "\n"
            + self.example
            + "\n[The input example here]\n"
            + self.eos_instruction
        )

    def __len__(self):
        return len(self.bos_instruction) + len(self.example) + len(self.eos_instruction)

    def __repr__(self):
        return (
            self.bos_instruction
            + "\n"
            + self.example
            + "\n[The input example here]\n"
            + self.eos_instruction
        )

    def prepare_input(self, input_text):
        return (
            self.bos_instruction
            + "\n"
            + self.example
            + "\ninput: "
            + input_text
            + "\n"
            + self.eos_instruction
        )
