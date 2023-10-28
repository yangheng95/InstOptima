import random

import numpy

from entity.individual import Individual


class Population(list):
    def __init__(self, size=0, **kwargs):
        super(Population, self).__init__()

        for _ in range(size):
            no_eval = kwargs.pop("no_eval", True)
            self.append(Individual(no_eval=no_eval, **kwargs))

    def objectives(self):
        objs = numpy.vstack([individual.objectives for individual in self])
        avg_objs = numpy.nanmean(objs, axis=0)
        return "Minimize Objective(instruction_length={}, perplexity={}, reciprocal_metric={})".format(
            round(avg_objs[0], 2), round(avg_objs[1], 2), round(avg_objs[2], 2)
        )
        # return "Minimize Objective(instruction_length={}, reciprocal_metric={})".format(
        #     round(avg_objs[0], 2), round(avg_objs[1], 2)
        # )

    def init_from_list(self, list_of_individuals):
        for individual in list_of_individuals:
            self.append(individual)
        return self

    def __str__(self):
        return "\n".join([str(individual) for individual in self])

    def __repr__(self):
        return "\n".join([str(individual) for individual in self])


if __name__ == "__main__":
    p = Population()
    print(p[0].fitness + p[1].fitness)
    print(p)
    print(len(p))
