# -*- coding: utf-8 -*-
# file: test.py
# time: 10:43 01/07/2023
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.
import pickle

import findfile
import numpy as np

from pymoo.indicators.hv import HV


for model in [
    # 'flan-t5-base',
    "flan-t5-small",
]:
    for dataset in [
        "Laptop14",
        "Restaurant14",
        "SNLI",
        "MNLI",
        "SST-2",
        "AgNews",
    ]:
        for generation in range(1, 21):
            for f in findfile.find_cwd_files(
                [dataset, model, "Generation-" + str(generation)+'-', "29", "pkl"],
                exclude_key=[".ignore", "06"],
            ):
                with open(f, "rb") as fin:
                    evo_data = pickle.load(fin)
                solutions = []

                for individual in evo_data["__history_fronts__"][-1][0]:
                    solutions.append(
                        [
                            individual.objectives.objectives[0] / 1000.0,
                            # individual.objectives.objectives[0],
                            individual.objectives.objectives[1],
                            individual.objectives.objectives[2],
                        ]
                    )

                solutions = np.array(solutions)
                print(solutions.shape)
                hv = HV(ref_point=np.array([1.5, 1.5, 1.5]))

                print("Generation", generation, f, hv(solutions))
