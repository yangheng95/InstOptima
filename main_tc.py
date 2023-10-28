from evo_core.nsga2 import nsga2

# Configurations
generation_num = 20
population_size = 2
tournament_size = 2

dataset = "SST2"
# dataset = "AgNews"
# dataset = 'Amazon'

plm = "google/flan-t5-base"
# plm = "google/flan-t5-small"
# plm = "t5-base"
# plm = "t5-small"
# plm = "roberta-base"
# plm = "bert-base-uncased"

evo_data = nsga2(
    population_size,
    num_generations=generation_num,
    tournament_size=tournament_size,
    dataset=dataset,
    plm=plm,
)
print(evo_data)
