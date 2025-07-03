from RoBERTaModule import RoBERTaModule
from wikidata import WikiData

roberta_module = RoBERTaModule()
wikidata = WikiData()

if __name__ == "__main__":

    train_dl, valid_dl = wikidata.preprocess_and_load()
    roberta_module.train_model(train_dl, valid_dl, num_epochs=10, lr=6e-4)