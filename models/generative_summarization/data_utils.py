import os.path
import pickle

import pandas as pd

from models.generative_summarization.instruction import SumInstruction


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
        tc_instructor = SumInstruction(instruction, example)
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
    from datasets import load_dataset

    # write to file

    train_data = []
    if os.path.exists("datasets/sum_datasets/Gigaword/train.txt"):
        train_data = pickle.load(open("datasets/sum_datasets/Gigaword/train.txt", "rb"))
    else:
        os.makedirs("datasets/sum_datasets/Gigaword", exist_ok=True)
        # 选择Gigaword数据集
        dataset = load_dataset("gigaword")

        # 获取训练集、验证集和测试集
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        test_dataset = dataset["test"]

        train_tile = "datasets/sum_datasets/Gigaword/train.txt"
        for i in range(0, 5000):
        # for i in range(len(train_dataset)):
            text = train_dataset[i]["document"]
            label = train_dataset[i]["summary"]
            train_data.append({"text": text, "label": label})

        pickle.dump(train_data, open(train_tile, "wb"))

    test_data = []
    if os.path.exists("datasets/sum_datasets/Gigaword/test.txt"):
        test_data = pickle.load(open("datasets/sum_datasets/Gigaword/test.txt", "rb"))
    else:
        test_tile = "datasets/sum_datasets/Gigaword/test.txt"
        for i in range(0, 1000):
            text = test_dataset[i]["document"]
            label = test_dataset[i]["summary"]
            test_data.append({"text": text, "label": label})

        pickle.dump(test_data, open(test_tile, "wb"))

    if data_type == "train":
        return train_data

    else:
        return test_data
