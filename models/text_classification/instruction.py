class Instruction:
    def __init__(self, bos_instruction=None, eos_instruction=None):
        self.bos_instruction = bos_instruction
        self.eos_instruction = eos_instruction

    def set_instruction(self, bos_instruction, eos_instruction):
        self.bos_instruction = bos_instruction
        self.eos_instruction = eos_instruction

    def get_instruction(self):
        return self.bos_instruction, self.eos_instruction


class TCInstruction(Instruction):
    def __init__(self, bos_instruction=None, example=None, eos_instruction=None):
        super().__init__(bos_instruction, eos_instruction)
        self.example = example
        if self.eos_instruction is None:
            self.eos_instruction = "The statement is:"

        if self.bos_instruction is None:
            self.bos_instruction = """Definition: The input are sentences about a product or service. The task is to classify texts. Here are some examples in the following"""
        if self.example is None:
            self.example = f"""
input: there is a freedom to watching stunts that are this crude , this fast-paced and this insane .
{self.eos_instruction} positive

input: Great food, good size menu, great service and an unpretensious setting.
{self.eos_instruction} positive
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
            + "\ninput: "
            + input_text
            + "\n"
            + self.eos_instruction
        )
