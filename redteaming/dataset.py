import json
import linecache
import random
import subprocess

from torch.utils.data import DataLoader, Dataset
import copy
import torch
from transformers import DataCollatorForSeq2Seq
import random

class SFTDataset(Dataset):
    def __init__(self, tokenizer, prompt_file, instruction_file, split="train") -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = []
        
        with open(prompt_file, "r") as f:
            for line in f:
                prompt = json.loads(line)["attacker_prompt"]
                self.prompts.append(prompt)
        with open(instruction_file, "r") as f:
            instructions = json.load(f)
        self.instructions = [x["instruction"].strip() for x in instructions]
        random.seed(42)
        random.shuffle(self.instructions)
        num_vals = int(len(self.instructions) * 0.1)
        
        if split == "train":
            self.instructions = self.instructions[num_vals:]
        elif split == "val":
            self.instructions = self.instructions[:num_vals]
        
        print(len(self.instructions))

    
    def __len__(self):
        return len(self.instructions)
    
    def __getitem__(self, index):
        prompt = random.choice(self.prompts)
        instruction = self.instructions[index]
        item = self.encode(prompt, instruction)

        return item
    
    def get_labels(self):
        return self.labels

    def encode(self, prompt, instruction):
        example = prompt + " " + instruction

        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)

        example[~example_mask] = 0
        labels[~label_mask] = -100

        return {"input_ids": example.tolist(),
                "labels": labels.tolist(),
                "attention_mask": example_mask.tolist()}



class SafetyDataset(Dataset):
    def __init__(self, tokenizer, instruction_file) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        
        with open(instruction_file, "r") as f:
            self.data = json.load(f)
        print(len(self.data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        instruction = self.data[index]["instruction"]
        response = self.data[index]["response"]
        item = self.encode(instruction, response)

        return item
    

    def encode(self, prompt, response):
        
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role":"user", "content": prompt}], tokenize=False)
        example = chat_prompt + " " + response

        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)

        example[~example_mask] = 0
        labels[~label_mask] = -100

        return {"input_ids": example.tolist(),
                "labels": labels.tolist(),
                "attention_mask": example_mask.tolist()}


class RedTeamDataset(Dataset):
    def __init__(self, jsonl_file) -> None:
        super().__init__()
        self.file_name = jsonl_file
        self.total_size = int(subprocess.check_output(
            "wc -l " + jsonl_file, shell=True).split()[0])


    def __getitem__(self, index) :
        line = linecache.getline(self.file_name, index+1)
        prompt = json.loads(line)["attacker_prompt"]

        return prompt

    def __len__(self):
        return self.total_size

def get_dataloader(name, tokenizer, prompt_file, sft_file=None,  batch_size=16, shuffle=True):
    if name == "redteam":
        dataset = RedTeamDataset(prompt_file)
        def collate_fn(data):
            return tokenizer(data, padding=True, truncation=True, return_tensors="pt")

        dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return dataloader

    elif name == "sft":
        tr_dataset = SFTDataset(tokenizer, prompt_file, sft_file, split="train")
        val_dataset = SFTDataset(tokenizer, prompt_file, sft_file, split="val")
        
        tr_dataloader = DataLoader(tr_dataset, batch_size, shuffle=shuffle, collate_fn=DataCollatorForSeq2Seq(tokenizer))
        val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=DataCollatorForSeq2Seq(tokenizer))
        
        return tr_dataloader, val_dataloader
    
    elif name == "distillation":
        print("distillation")
        tr_dataset = SFTDataset(tokenizer, prompt_file, sft_file, split="full")
        
        tr_dataloader = DataLoader(tr_dataset, batch_size, shuffle=shuffle, collate_fn=DataCollatorForSeq2Seq(tokenizer))
        
        return tr_dataloader

    elif name == "safety-tuning":
        tr_dataset = SafetyDataset(tokenizer, prompt_file)
        tr_dataloader = DataLoader(tr_dataset, batch_size, shuffle=shuffle, collate_fn=DataCollatorForSeq2Seq(tokenizer))
        
        return tr_dataloader


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataloader = get_dataloader(tokenizer, "./prompts/alpaca.jsonl", 16, True)
    for batch in dataloader:
        print(batch)
