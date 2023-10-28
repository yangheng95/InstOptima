from objectives.objective import Objective
from entity.instruction import Instruction


class Individual(object):
    __pid__ = 0
    """
    Individual is the basic unit of a population.
    """

    def __init__(self, instruction=None, generation_id=0, **kwargs):
        self.generation_id = generation_id
        op = kwargs.get("op", None)

        if instruction is None:
            self.instruction = Instruction(**kwargs)
            self.instruction = op.evolve(self.instruction, self.instruction, **kwargs)
        else:
            self.instruction = instruction

        self.instruction.objectives = Objective(
            inst_individual=self.instruction, **kwargs
        )
        self.objectives = self.instruction.objectives
        self.rank = 0
        self.crowding_distance = 0.0
        self.pid = Individual.__pid__
        Individual.__pid__ += 1
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update_objectives(self):
        self.objectives.update()

    def __getitem__(self, item):
        return self.instruction[item]

    def __len__(self):
        return len(self.instruction)

    def __repr__(self):
        return "Individual(genotype={}, objectives={})".format(
            self.instruction, self.objectives
        )
