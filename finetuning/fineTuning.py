import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from datasets import load_dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, RobertaTokenizerFast
from modelFineTuning import RoBERTa


class RoBERTaModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.model = RoBERTa(vocab_size=self.tokenizer.vocab_size, padding_idx=self.tokenizer.pad_token_id, num_labels = 2) # !!

    def forward(self, x, attn_mask):
        return self.model(x, attn_mask)

    def train_model(self, train_loader, validation_loader, num_epochs, lr=2e-5, optimizer=None, scheduler=None, scaler=None, device="cuda"):
        #device = torch.device("cuda")
        self.model.to(device)

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(0.06 * total_steps) # 6% of the total number of steps are warmups steps as in paper

        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.99, 0.999), eps=1e-6, weight_decay=0.01)

        if scheduler is None:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps) # linear insted of cosine as in paper

        if scaler is None:
            scaler = GradScaler()

        writer = SummaryWriter()

        # early stopping
        patience_counter = 0
        patience_limit = 5
        epsilon = 1e-3
        best_valid_loss = float("inf")
        best_model_state = None

        for epoch in range(num_epochs):
            # train part
            self.model.train()
            total_loss_train = 0

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
                input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)

                with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    output = self.model(input_ids, attention_mask)
                    loss = F.cross_entropy(output.view(-1, output.shape[-1]), labels.view(-1), ignore_index=-100)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) # unscale before clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # gradient clipping
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                optimizer.zero_grad()

                total_loss_train += loss.item()

            train_loss = total_loss_train / len(train_loader)

            # validation part

            self.model.eval()

            total_loss_valid = 0
            total_correct = 0
            total_tokens = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(validation_loader):
                    input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)


                    with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        output = self.model(input_ids, attention_mask)
                        loss = F.cross_entropy(output.view(-1, output.shape[-1]), labels.view(-1), ignore_index=-100, label_smoothing=0.05)

                    preds = torch.argmax(output, dim=-1)
                    correct = (preds == labels).float().sum()

                    total_loss_valid += loss.item()
                    total_correct += correct
                    total_tokens += labels.size(0)

            validation_loss = total_loss_valid / len(validation_loader)
            validation_accuracy = total_correct / total_tokens

            if validation_loss < best_valid_loss - epsilon:
                best_valid_loss = validation_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    self.model.load_state_dict(best_model_state)
                    self.save_checkpoint(best_model_state, optimizer, scheduler, scaler, path="cpEarly.pt")
                    break


            print(f'Epoch {epoch + 1}, train loss: {train_loss:.4f}, valid loss: {validation_loss:.4f}, '
                 f'Validation accuracy: {validation_accuracy:.4f}')

        writer.close()

        self.model.load_state_dict(best_model_state)
        self.save_checkpoint(self.model.state_dict(), optimizer, scheduler, path="finishedBest.pt")


    def save_checkpoint(self, model, optimizer, scheduler, path="checkpoint.pt"):
        torch.save({
            "model_state_dict": model,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, model=None, path="finished.pt", location="cuda"):
        checkpoint = torch.load(path, map_location=location, weights_only=True)

        if not model:
          self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
          model.load_state_dict(checkpoint["model_state_dict"], strict=False)