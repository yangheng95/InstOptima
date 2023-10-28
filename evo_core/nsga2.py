import os.path
import pickle
import random

import tqdm

from entity.individual import Individual
from entity.population import Population
from operators.instruction_operators import InstructOperator
from operators.prompt_operators import PromptOperator


def nsga2(population_size, num_generations, **kwargs):
    inst_op = InstructOperator(kwargs.get("checkpoint"))
    prompt_op = PromptOperator(kwargs.get("checkpoint"))
    dir_path = (
        f"results/"
        + kwargs.get("plm")
        + "/"
        + kwargs.get("dataset")
        + "/"
    )
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    fmt_str = dir_path + "Generation-{}-Pid-{}-{}.pkl"
    if not os.path.exists(fmt_str.format(0, 0, kwargs.get("dataset"))):
        population = Population(size=population_size, op=inst_op, **kwargs)
    else:
        population = Population(0)

    evo_data = {
        "instance_operator": inst_op,
        "prompt_operator": prompt_op,
        "population": population,
        "fronts": [],
        "__offspring__": Population(0),
        "__history_fronts__": [],
        "__next_population__": Population(0),
        "__history_population__": Population(0),
        "operations": {},
    }

    # Perform NSGA-II for specified number of generations
    for generation in tqdm.tqdm(range(num_generations), desc="Evolving using NSGA-II"):
        print("Generation {} of {}".format(generation, num_generations - 1))
        for p_id in range(population_size):
            if os.path.exists(fmt_str.format(generation, p_id, kwargs.get("dataset"))):
                evo_data = pickle.load(
                    open(fmt_str.format(generation, p_id, kwargs.get("dataset")), "rb")
                )
                print(
                    "Resume evolution from "
                    + fmt_str.format(generation, p_id, kwargs.get("dataset"))
                )

                continue

            # Select parents using binary tournament selection
            parent1 = evo_data["population"][p_id]
            parent2 = binary_tournament_selection(
                evo_data["population"],
            )
            print("Parent 1: {}".format(parent1.pid))
            # with contextlib.redirect_stdout(None):
            # Apply crossover and mutation operators
            instruction = evo_data["instance_operator"].evolve(
                parent1.instruction, parent2.instruction, **kwargs
            )
            child = Individual(instruction, generation + 1, **kwargs)
            # torch.cuda.empty_cache()

            evo_data["__offspring__"].append(child)

            if not os.path.exists(
                fmt_str.format(generation, p_id, kwargs.get("dataset"))
            ):
                print(
                    "Save to " + fmt_str.format(generation, p_id, kwargs.get("dataset"))
                )
                with open(
                    fmt_str.format(generation, p_id, kwargs.get("dataset")), "wb"
                ) as fpkl:
                    pickle.dump(
                        evo_data,
                        fpkl,
                    )

        # Combine parent population and offspring
        combined_population = evo_data["population"] + evo_data["__offspring__"]
        combined_population = list(set(combined_population))

        # for p in evo_data["population"]:
        #     if len(p.objectives) > 2:
        #         p.objectives = p.objectives[:2]
        # Perform non-dominated sorting
        evo_data["fronts"] = non_dominated_sort(combined_population)

        # Assign ranks and crowding distances
        assign_ranks(evo_data["fronts"])
        calculate_crowding_distances(evo_data["fronts"])

        evo_data["__history_population__"].extend(combined_population)
        evo_data["__history_fronts__"].append(evo_data["fronts"])

        # Create the next generation population
        for front in evo_data["fronts"]:
            evo_data["__next_population__"].extend(front)

            if len(evo_data["__next_population__"]) >= len(evo_data["population"]):
                break

        # Sort the next population based on rank and crowding distance
        evo_data["__next_population__"].sort(
            key=lambda x: (x.rank, -x.crowding_distance)
        )

        # Truncate the next population to the original population size
        evo_data["population"] = Population().init_from_list(
            evo_data["__next_population__"][: len(evo_data["population"])]
        )

        # operator_evolution
        if (
            generation != 0
            and generation % 1 == 0
            and generation not in evo_data["operations"]
        ):
            evo_data["instance_operator"].operations = evo_data[
                "prompt_operator"
            ].evolve(
                evo_data["instance_operator"].operations,
                evo_data["__offspring__"].objectives(),
                **kwargs,
            )

            evo_data["operations"][generation] = evo_data[
                "instance_operator"
            ].operations
            print("Save to " + fmt_str.format(generation, p_id, kwargs.get("dataset")))
            with open(
                fmt_str.format(generation, p_id, kwargs.get("dataset")), "wb"
            ) as fpkl:
                pickle.dump(
                    evo_data,
                    fpkl,
                )

        if generation == num_generations - 1:
            print(
                "Save to " + fmt_str.format(generation + 1, p_id, kwargs.get("dataset"))
            )
            with open(
                fmt_str.format(generation + 1, p_id, kwargs.get("dataset")), "wb"
            ) as fpkl:
                pickle.dump(
                    evo_data,
                    fpkl,
                )

        evo_data["__offspring__"] = Population(0)
        evo_data["__next_population__"] = Population(0)

    return evo_data


def binary_tournament_selection(population, tournament_size=2):
    # Select an individual using binary tournament selection
    random.seed(random.randint(0, 100000))
    selected_individual = random.choice(population)

    for _ in range(tournament_size - 1):
        individual = random.choice(population)

        if individual.rank < selected_individual.rank:
            selected_individual = individual
        elif (
            individual.rank == selected_individual.rank
            and individual.crowding_distance > selected_individual.crowding_distance
        ):
            selected_individual = individual

    return selected_individual


# import random

# def binary_tournament_selection(population, tournament_size=2):
#     selected_individuals = []
#     population_size = len(population)
#
#     for _ in range(population_size):
#         tournament = random.sample(population, tournament_size)
#         winner = None
#
#         for individual in tournament:
#             if winner is None:
#                 winner = individual
#             else:
#                 if individual.rank < winner.rank:  # Lower rank is better
#                     winner = individual
#                 elif individual.rank == winner.rank and individual.crowding_distance > winner.crowding_distance:  # Higher crowding distance is better
#                     winner = individual
#
#         selected_individuals.append(winner)
#
#     return selected_individuals


def non_dominated_sort(population):
    # Non-dominated sorting algorithm

    dominance = {}
    rank = {}
    fronts = [[]]

    for p in population:
        dominance[p] = set()
        rank[p] = 0

    for p in population:
        for q in population:
            if p == q:
                continue

            if dominates(p, q):
                dominance[p].add(q)
            elif dominates(q, p):
                rank[p] += 1

        if rank[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []

        for p in fronts[i]:
            for q in dominance[p]:
                rank[q] -= 1

                if rank[q] == 0:
                    next_front.append(q)

        i += 1
        fronts.append(next_front)

    return fronts


def dominates(p, q):
    # Check if p dominates q

    dominates_p = False
    dominates_q = False

    for i in range(len(p.objectives)):
        if p.objectives[i] < q.objectives[i]:
            dominates_p = True
        elif p.objectives[i] > q.objectives[i]:
            dominates_q = True

    return dominates_p and not dominates_q


def assign_ranks(fronts):
    # Assign ranks to individuals based on fronts

    rank = 1

    for front in fronts:
        for individual in front:
            individual.rank = rank

        rank += 1


def calculate_crowding_distances(fronts):
    # Calculate crowding distance for individuals in each front

    for front in fronts:
        if not front:
            continue
        num_objectives = len(front[0].objectives)

        for individual in front:
            individual.crowding_distance = 0.0

        for obj_index in range(num_objectives):
            front.sort(key=lambda x: x.objectives[obj_index])
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (
                    front[i + 1].objectives[obj_index]
                    - front[i - 1].objectives[obj_index]
                )
