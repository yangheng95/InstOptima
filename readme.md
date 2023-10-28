# Evolutionary Multi-objective Instruction Optimization via LLM

This repo is for our
paper: [Evolutionary Multi-objective Instruction Optimization via Large Language Model-based Instruction Operators](https://arxiv.org/abs/2310.17630).

## Abstract

Instruction-based language modeling has received significant attention in pretrained language models. However, the
efficiency of instruction engineering remains low and hinders the development of instruction studies. Recent studies
have focused on automating instruction generation, but they primarily aim to improve performance without considering
other crucial objectives that impact instruction quality, such as instruction length and perplexity. Therefore, we
propose a novel approach (i.e., InstOptima) that treats instruction generation as an evolutionary multi-objective
optimization problem. In contrast to text edition-based methods, our approach utilizes a large language model (LLM) to
simulate instruction operators, including mutation and crossover. Furthermore, we introduce an objective-guided
mechanism for these operators, allowing the LLM to comprehend the objectives and enhance the quality of the generated
instructions. Experimental results demonstrate improved fine-tuning performance and the generation of a diverse set of
high-quality instructions.

## Requirements Installation

### Install Conda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ./Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

### Create Conda Environment

```bash
conda create -n instoptima python=3.9
conda activate instoptima
```

### Install PyTorch

```bash
conda install pytorch pytorch-cuda -c pytorch -c nvidia
```

### Install Other Requirements

```bash
pip install -r requirements.txt
```

## Quick Examples

For all tasks, we can start the chatbot by running the following command:

```bash
python chatbot.py
```

This will start the chatbot service on port 6789. You can then send a POST request to the chatbot service to get the
response. By using this chatbot, all the instructions will be generated by the LLM-based instruction operators and
archived
in the current directory.

### ABSA Example

For ABSA task, please revise the config in the `main.py` file and run the following command:

```bash
python main.py
```

## Notice

This is the initial version of our code. We will update the code and add more examples in the future. If you have any
questions, please feel free to contact us.

## Citation

```bibtex
@inproceedings{
anonymous2023instoptima,
title={InstOptima: Evolutionary Multi-objective Instruction Optimization via Large Language Model-based Instruction Operators},
author={Anonymous},
booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
year={2023},
url={https://openreview.net/forum?id=8oy8hUeem9}
}
```