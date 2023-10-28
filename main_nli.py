import pickle

from evo_core.nsga2 import nsga2

# Configurations
generation_num = 20
population_size = 2

# dataset = 'toy'

dataset = "SNLI"
# dataset = "MNLI"

plm = "google/flan-t5-base"
# plm = "google/flan-t5-small"
# plm = "t5-base"
# plm = "t5-small"
# plm = "bert-base-uncased"

evo_data = nsga2(
    population_size,
    num_generations=generation_num,
    dataset=dataset,
    plm=plm,
)

pickle.dump(evo_data, open("nsga2_result_v4.pkl", "wb"))
pareto_front = evo_data["fronts"][0]


for individual in pareto_front:
    print("-" * 100)
    print(f"Prompt: {individual.instruction}")
    print(f"Objectives: {individual.objectives}")
evo_data = pickle.load(open("nsga2_result_v4.pkl", "rb"))

# PLOT THE PARETO FRONTS
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("Metrics")
ax.set_ylabel("Perplexity")
ax.set_zlabel("Prompt Length")
colors = ["r", "g", "b", "y", "c", "m", "k", "w"]
for front in evo_data["fronts"]:
    prompt_length = [individual.objectives[0] for individual in front]
    perplexity = [individual.objectives[1] for individual in front]
    metrics = [individual.objectives[2] for individual in front]
    ax.scatter(metrics, perplexity, prompt_length, c=colors.pop(0))
plt.savefig("pareto_front.png", dpi=1000)
plt.show()


print("Done!")
