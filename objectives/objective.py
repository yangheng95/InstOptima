import random
import warnings

import numpy as np
from transformers import set_seed

from entity.instruction import Instruction
# from models.aspect_based_sentiment_analysis.chagpt import train_absa
from models.aspect_based_sentiment_analysis.flant5 import train_absa
# from models.generative_summarization.chagpt import train_sum
from models.generative_summarization.flant5 import train_sum
from models.natural_language_inference.flant5 import train_nli
from models.text_classification.flant5 import train_tc
from objectives.perplexity import Perplexity

perplexity = Perplexity()


class Objective(object):
    def __init__(self, inst_individual, **kwargs):
        assert isinstance(inst_individual, Instruction)
        self.prompt = inst_individual

        self.prompt_length = np.inf
        self.perplexity = np.inf
        self.metric = np.inf
        self.objectives = [
            self.prompt_length,
            self.perplexity,
            self.metric,
        ]
        self.dataset = kwargs.get("dataset", "default")
        self.plm = kwargs.get("plm", None)
        # self.update(**kwargs)
        if not kwargs.get("no_eval", False):
            self.update(**kwargs)
        print(self)

    def update(self, **kwargs):
        import time

        time5 = time.time()
        self.metric = self._metric(**kwargs)
        time6 = time.time()
        print(f"performance time: {time6 - time5}")
        time1 = time.time()
        self.prompt_length = self._prompt_length()
        time2 = time.time()
        print(f"length time: {time2 - time1}")
        time3 = time.time()
        self.perplexity = self._perplexity()
        time4 = time.time()
        print(f"perplexity time: {time4 - time3}")
        self.objectives = [
            self.prompt_length,
            self.perplexity,
            1 / sum(self.metric.values()) if self.metric else np.inf,
        ]

    def _prompt_length(self):
        return len(self.prompt)

    def _perplexity(self):
        return perplexity.calculate_perplexity(self.prompt)

    def _metric(self, **kwargs):
        k = kwargs.get("k", 1)
        if self.dataset in ["Laptop14", "Restaurant14", "Restaurant15", "Restaurant16"]:
            metrics = {}
            for i in range(k):
                set_seed(random.randint(0, 10000))
                print(f"{i}-th metrics...")
                metrics[i] = train_absa(
                    epoch=3,
                    instruction=self.prompt.definition,
                    example=self.prompt.example,
                    dataset=self.dataset,
                    plm=self.plm,
                )
            merged_metrics = {}
            for metric in metrics.values():
                for k, v in metric.items():
                    if k not in merged_metrics:
                        merged_metrics[k] = []
                    merged_metrics[k].append(v)
            return {k: np.mean(v) for k, v in merged_metrics.items()}

        elif self.dataset in ["SST2", "Amazon", "AgNews"]:
            metrics = {}
            for i in range(k):
                set_seed(random.randint(0, 10000))
                print(f"{i}-th metrics...")
                metrics[i] = train_tc(
                    epoch=3,
                    instruction=self.prompt.definition,
                    example=self.prompt.example,
                    dataset=self.dataset,
                    plm=self.plm,
                )
            merged_metrics = {}
            for metric in metrics.values():
                for k, v in metric.items():
                    if k not in merged_metrics:
                        merged_metrics[k] = []
                    merged_metrics[k].append(v)
            return {k: np.mean(v) for k, v in merged_metrics.items()}
        elif self.dataset in ["SNLI", "MNLI"]:
            metrics = {}
            for i in range(k):
                set_seed(random.randint(0, 10000))
                print(f"{i}-th metrics...")
                metrics[i] = train_nli(
                    epoch=3,
                    instruction=self.prompt.definition,
                    example=self.prompt.example,
                    dataset=self.dataset,
                    plm=self.plm,
                )
            merged_metrics = {}
            for metric in metrics.values():
                for k, v in metric.items():
                    if k not in merged_metrics:
                        merged_metrics[k] = []
                    merged_metrics[k].append(v)
            return {k: np.mean(v) for k, v in merged_metrics.items()}
        elif self.dataset in ["Gigaword"]:
            metrics = {}
            for i in range(k):
                set_seed(random.randint(0, 10000))
                print(f"{i}-th metrics...")
                metrics[i] = train_sum(
                    epoch=1,
                    instruction=self.prompt.definition,
                    example=self.prompt.example,
                    dataset=self.dataset,
                    plm=self.plm,
                )
            merged_metrics = {}
            for metric in metrics.values():
                for k, v in metric.items():
                    if k not in merged_metrics:
                        merged_metrics[k] = []
                    merged_metrics[k].append(v)
            return {k: np.mean(v) for k, v in merged_metrics.items()}

        else:
            warnings.warn(f"Dataset {self.dataset} not supported, return None.")
            return None

    def __repr__(self):
        return "Minimize Objective(instruction_length={}, perplexity={}, reciprocal_metric={})".format(
            self.prompt_length, self.perplexity, self.metric
        )

    def __str__(self):
        return "Minimize Objective(instruction_length={}, perplexity={}, reciprocal_metric={})".format(
            round(self.objectives[0], 2),
            round(self.objectives[1], 2),
            round(self.objectives[2], 2),
        )

    def __getitem__(self, item):
        return self.objectives[item]

    def __len__(self):
        return len(self.objectives)


# import random
# import warnings
#
# import numpy as np
#
# from models.natural_language_inference.train import train_nli
# from entity.instruction import Instruction
#
# from objectives.perplexity import Perplexity
#
# from models.aspect_based_sentiment_analysis.train import train_absa
# from models.text_classification.train import train_tc
# from transformers import set_seed
#
# # perplexity = Perplexity()
#
#
# class Objective(object):
#     def __init__(self, inst_individual, **kwargs):
#         assert isinstance(inst_individual, Instruction)
#         self.prompt = inst_individual
#
#         self.prompt_length = np.inf
#         self.perplexity = np.inf
#         self.metric = np.inf
#         self.objectives = [
#             self.prompt_length,
#             # self.perplexity,
#             self.metric,
#         ]
#         self.dataset = kwargs.get("dataset", "default")
#         self.plm = kwargs.get("plm", None)
#         # self.update(**kwargs)
#         if not kwargs.get("no_eval", False):
#             self.update(**kwargs)
#         print(self)
#
#     def update(self, **kwargs):
#         self.prompt_length = self._prompt_length()
#         # self.perplexity = self._perplexity()
#         self.metric = self._metric(**kwargs)
#
#         self.objectives = [
#             self.prompt_length,
#             # self.perplexity,
#             1 / sum(self.metric.values()) if self.metric else np.inf,
#         ]
#
#     def _prompt_length(self):
#         return len(self.prompt)
#
#     # def _perplexity(self):
#     #     return perplexity.calculate_perplexity(self.prompt)
#
#     def _metric(self, **kwargs):
#         k = kwargs.get("k", 1)
#         if self.dataset in ["Laptop14", "Restaurant14", "Restaurant15", "Restaurant16"]:
#             metrics = {}
#             for i in range(k):
#                 set_seed(random.randint(0, 10000))
#                 print(f"{i}-th metrics...")
#                 metrics[i] = train_absa(
#                     epoch=3,
#                     instruction=self.prompt.definition,
#                     example=self.prompt.example,
#                     dataset=self.dataset,
#                     plm=self.plm,
#                 )
#             merged_metrics = {}
#             for metric in metrics.values():
#                 for k, v in metric.items():
#                     if k not in merged_metrics:
#                         merged_metrics[k] = []
#                     merged_metrics[k].append(v)
#             return {k: np.mean(v) for k, v in merged_metrics.items()}
#
#         elif self.dataset in ["SST2", "Amazon", "AgNews"]:
#             metrics = {}
#             for i in range(k):
#                 set_seed(random.randint(0, 10000))
#                 print(f"{i}-th metrics...")
#                 metrics[i] = train_tc(
#                     epoch=3,
#                     instruction=self.prompt.definition,
#                     example=self.prompt.example,
#                     dataset=self.dataset,
#                     plm=self.plm,
#                 )
#             merged_metrics = {}
#             for metric in metrics.values():
#                 for k, v in metric.items():
#                     if k not in merged_metrics:
#                         merged_metrics[k] = []
#                     merged_metrics[k].append(v)
#             return {k: np.mean(v) for k, v in merged_metrics.items()}
#         elif self.dataset in ["SNLI", "MNLI"]:
#             metrics = {}
#             for i in range(k):
#                 set_seed(random.randint(0, 10000))
#                 print(f"{i}-th metrics...")
#                 metrics[i] = train_nli(
#                     epoch=3,
#                     instruction=self.prompt.definition,
#                     example=self.prompt.example,
#                     dataset=self.dataset,
#                     plm=self.plm,
#                 )
#             merged_metrics = {}
#             for metric in metrics.values():
#                 for k, v in metric.items():
#                     if k not in merged_metrics:
#                         merged_metrics[k] = []
#                     merged_metrics[k].append(v)
#             return {k: np.mean(v) for k, v in merged_metrics.items()}
#         else:
#             warnings.warn(f"Dataset {self.dataset} not supported, return None.")
#             return None
#
#     # def __repr__(self):
#     #     return "Minimize Objective(instruction_length={}, perplexity={}, reciprocal_metric={})".format(
#     #         self.prompt_length, self.perplexity, self.metric
#     #     )
#     #
#     # def __str__(self):
#     #     return "Minimize Objective(instruction_length={}, perplexity={}, reciprocal_metric={})".format(
#     #         round(self.objectives[0], 2),
#     #         round(self.objectives[1], 2),
#     #         round(self.objectives[2], 2),
#     #     )
#
#     def __repr__(self):
#         return "Minimize Objective(instruction_length={}, reciprocal_metric={})".format(
#             self.prompt_length, self.metric
#         )
#
#     def __str__(self):
#         return "Minimize Objective(instruction_length={}, reciprocal_metric={})".format(
#             round(self.objectives[0], 2),
#             round(self.objectives[1], 2),
#         )
#
#     def __getitem__(self, item):
#         return self.objectives[item]
#
#     def __len__(self):
#         return len(self.objectives)
