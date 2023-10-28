import json

import findfile
import pandas as pd

from .instruction import (
    TCInstruction,
)


class InstructDatasetLoader:
    def __init__(
        self,
        train_df_id,
        test_df_id,
        train_df_ood=None,
        test_df_ood=None,
        sample_size=1,
    ):
        self.train_df_id = train_df_id.sample(frac=sample_size, random_state=1999)
        self.test_df_id = test_df_id
        if train_df_ood is not None:
            self.train_df_ood = train_df_ood.sample(frac=sample_size, random_state=1999)
        else:
            self.train_df_ood = train_df_ood
        self.test_df_ood = test_df_ood

    def prepare_instruction_dataloader(self, df, instruction, example):
        """
        Prepare the data in the input format required.
        """
        tc_instructor = TCInstruction(instruction, example)
        alldata = []
        for i, data in df.iterrows():
            # TC task
            alldata.append(
                {
                    "text": tc_instructor.prepare_input(data["text"]),
                    "labels": data["label"],
                }
            )

        alldata = pd.DataFrame(alldata)
        return alldata

    def create_datasets(self, tokenize_function):
        from datasets import DatasetDict, Dataset

        """
        Create the training and test dataset as huggingface datasets format.
        """
        # Define train and test sets
        if self.test_df_id is None:
            indomain_dataset = DatasetDict(
                {"train": Dataset.from_pandas(self.train_df_id)}
            )
        else:
            indomain_dataset = DatasetDict(
                {
                    "train": Dataset.from_pandas(self.train_df_id),
                    "test": Dataset.from_pandas(self.test_df_id),
                }
            )
        indomain_tokenized_datasets = indomain_dataset.map(
            tokenize_function, batched=True
        )

        if (self.train_df_ood is not None) and (self.test_df_ood is None):
            other_domain_dataset = DatasetDict(
                {"train": Dataset.from_pandas(self.train_df_id)}
            )
            other_domain_tokenized_dataset = other_domain_dataset.map(
                tokenize_function, batched=True
            )
        elif (self.train_df_ood is None) and (self.test_df_ood is not None):
            other_domain_dataset = DatasetDict(
                {"test": Dataset.from_pandas(self.train_df_id)}
            )
            other_domain_tokenized_dataset = other_domain_dataset.map(
                tokenize_function, batched=True
            )
        elif (self.train_df_ood is not None) and (self.test_df_ood is not None):
            other_domain_dataset = DatasetDict(
                {
                    "train": Dataset.from_pandas(self.train_df_ood),
                    "test": Dataset.from_pandas(self.test_df_ood),
                }
            )
            other_domain_tokenized_dataset = other_domain_dataset.map(
                tokenize_function, batched=True
            )
        else:
            other_domain_dataset = None
            other_domain_tokenized_dataset = None

        return (
            indomain_dataset,
            indomain_tokenized_datasets,
            other_domain_dataset,
            other_domain_tokenized_dataset,
        )


def read_text(data_path, data_type="train"):
    data = []

    files = findfile.find_cwd_files(
        [data_path, "tc", data_type, ".dat"], exclude_key=[".txt"]
    )
    for f in files:
        print(f)
        with open(f, "r", encoding="utf8") as fin:
            for line in fin.readlines():
                text, _label = line.strip().split("$LABEL$")
                _label = _label.strip()
                label = "negative" if _label == "0" else "positive"
                data.append({"text": text, "label": label})

    return data[:1000]
    # if 'train' not in data_type:
    #     return data[:100]
    # else:
    #     return data
