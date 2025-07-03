from trainingPart.datasetPreprocessing import PreprocessDataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast
from datasets import load_dataset

class WikiData:
    def __init__(self):
        self.ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def preprocess_and_load(self):
        train_ds = PreprocessDataset(self.ds["train"], self.tokenizer)
        valid_ds = PreprocessDataset(self.ds["validation"], self.tokenizer)

        train_dl = DataLoader(train_ds, batch_size=32, drop_last=True, shuffle=True, num_workers=3)
        valid_dl = DataLoader(valid_ds, batch_size=32, drop_last=True, shuffle=False, num_workers=3)

        return train_dl, valid_dl