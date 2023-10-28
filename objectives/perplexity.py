import torch
from transformers import RobertaForMaskedLM, AutoTokenizer, AutoConfig
from autocuda import auto_cuda

from entity.instruction import Instruction


class Perplexity(object):
    def __init__(self, model="roberta-base", device="cuda"):
        assert model, "The model must not be None"
        self.device = device if device else auto_cuda()
        pretrained_config = AutoConfig.from_pretrained(model)

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.mlm = RobertaForMaskedLM(pretrained_config).to(self.device)

    def calculate_perplexity(self, prompt):
        inputs = self.tokenizer(
            str(prompt), return_tensors="pt", truncation=True, padding=True
        )
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.mlm(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            # perplexity = torch.exp(loss)
            perplexity = torch.exp(loss / inputs["input_ids"].size(1))

        return perplexity.item()


if __name__ == "__main__":
    p = Perplexity(device="cpu")
    print(p.calculate_perplexity(Instruction("hello world")))
