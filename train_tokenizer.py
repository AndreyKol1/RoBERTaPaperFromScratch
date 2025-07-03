from datasets import load_dataset

from tokenizer import BPE


class TrainTokenizer:
    def __init__(self):
        self.ds = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:1%]"
        )

    def write_text(self):
        with open("wiki_text.txt", "w", encoding="utf-8") as f:
            for article in self.ds:
                f.write(article["text"] + "\n")

    def extract_text(file_path):
        with open(f"{file_path}", "r") as f:
            data = f.read()
        return data


if __name__ == "__main__":
    tokenizer = BPE()
    trainer = TrainTokenizer()
    trainer.write_text()

    train_data = trainer.extract_text("wiki_text.txt")
    tokenizer.train(train_data, vocab_size=15996, verbose=True)
    tokenizer.save("tokenizer.pickle")
