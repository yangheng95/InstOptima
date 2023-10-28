class Instruction:
    def __init__(self, bos_instruction=None, eos_instruction=None):
        self.bos_instruction = bos_instruction
        self.eos_instruction = eos_instruction

    def set_instruction(self, bos_instruction, eos_instruction):
        self.bos_instruction = bos_instruction
        self.eos_instruction = eos_instruction

    def get_instruction(self):
        return self.bos_instruction, self.eos_instruction


class SumInstruction(Instruction):
    def __init__(self, bos_instruction=None, example=None, eos_instruction=None):
        super().__init__(bos_instruction, eos_instruction)
        self.example = example
        if self.eos_instruction is None:
            self.eos_instruction = "The summary is:"

        if self.bos_instruction is None:
            self.bos_instruction = """Generative Summarization for Gigaword Dataset: Given a set of input sentences, your goal is to generate a concise summary. Here are some examples in the following"""
        if self.example is None:
            self.example = f"""
Input: australia 's current account deficit shrunk by a record #.## billion dollars -lrb- #.## billion us -rrb- in the june quarter due to soaring commodity prices , figures released monday showed .
{self.eos_instruction}: australian current account deficit narrows sharply

Input: Gat least two people were killed in a suspected bomb attack on a passenger bus in the strife-torn southern philippines on monday , the military said .
{self.eos_instruction}: at least two dead in southern philippines blast

"""

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
            + "\nInput: "
            + input_text
            + "\n"
            + self.eos_instruction
        )
