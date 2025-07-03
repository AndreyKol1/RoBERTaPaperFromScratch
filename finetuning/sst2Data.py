from datasets import load_dataset
from transformers import RobertaTokenizerFast, DataCollatorWithPadding
from torch.utils.data import DataLoader

class SST2:
    def __init__(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.ds = load_dataset("stanfordnlp/sst2")

    def tokenize_inputs(self, example):
        tokenized_inputs = self.tokenizer(example["sentence"], truncation=True, max_length=128)
        return tokenized_inputs
    
    def preprocess_and_load(self):
        tokenized_train_ds = self.ds["train"].map(self.tokenize_inputs, batched=True, remove_columns=["idx", "sentence"])
        tokenized_val_ds = self.ds["validation"].map(self.tokenize_inputs, batched=True, remove_columns=["idx", "sentence"])

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)

        train_dl = DataLoader(tokenized_train_ds, batch_size=32, shuffle=True, collate_fn=data_collator)
        valid_dl = DataLoader(tokenized_val_ds, batch_size=32, shuffle=False, collate_fn=data_collator)

        return train_dl, valid_dl