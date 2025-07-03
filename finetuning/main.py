from fineTuning import RoBERTaModule
from sst2Data import SST2

roberta_module = RoBERTaModule()
sst2Data = SST2()

if __name__ == "__main__":

    train_dl, valid_dl = sst2Data.preprocess_and_load()
    roberta_module.load_checkpoint(path="finishedBest10.pt", location="cpu") # path to your base model(should be installed localy)
    roberta_module.train_model(train_dl, valid_dl, num_epochs=10, lr=1e-5, device="cpu") # hyperparameters as in paper