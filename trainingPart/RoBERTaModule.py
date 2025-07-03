import copy
import torch
import torch.nn.functional as F
from architecture.model import RoBERTa
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from transformers import RobertaTokenizerFast


class RoBERTaModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.model = RoBERTa(vocab_size=self.tokenizer.vocab_size, padding_idx=self.tokenizer.pad_token_id)

    def forward(self, x, attn_mask):
        return self.model(x, attn_mask)

    def train_model(self, train_loader, validation_loader, num_epochs, lr=6e-4, optimizer=None, scheduler=None, scaler=None):
        device = torch.device("cuda")
        self.model.to(device)

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(0.1 * total_steps) # 10% of the total number of steps are warmups steps

        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.98, 0.999), eps=1e-6, weight_decay=0.01)

        if scheduler is None:
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        if scaler is None:
            scaler = GradScaler()

        writer = SummaryWriter()

        # early stopping
        patience_counter = 0
        patience_limit = 5
        epsilon = 1e-3
        best_valid_loss = torch.tensor(float('inf'))
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
                        loss = F.cross_entropy(output.view(-1, output.shape[-1]), labels.view(-1), ignore_index=-100)

                    preds = output.argmax(dim=-1)
                    mask = labels != -100
                    correct = ((preds == labels) & mask).float().sum()

                    total_loss_valid += loss.item()
                    total_correct += correct.item()
                    total_tokens += mask.float().sum().item()

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
                    self.save_checkpoint(best_model_state, optimizer, scheduler, scaler, epoch, best_valid_loss, path="cpEarly.pt")
                    break

            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", validation_loss, epoch)
            writer.add_scalar("Accuracy/Validation", validation_accuracy, epoch)

            if epoch % 3 == 0:
              test_sentences = [
                "The <mask> barked at the girl",
                "She wore a <mask> dress to the party",
                "The <mask> is shining brightly",
                "The cat <mask> on the mat.",
                "The president gave a <mask> speech.",
                "She took a <mask> before dinner."
            ]
              for sent in test_sentences:
                  prediction = self.inference(self.model, self.tokenizer, sent, self.tokenizer.pad_token_id)
                  print(f"Inference [{sent}] → {prediction}")

              self.save_checkpoint(self.model.state_dict(), optimizer, scheduler, scaler, epoch, best_valid_loss, path="checkpoint.pt")

            print(f'Epoch {epoch + 1}, train loss: {train_loss:.4f}, valid loss: {validation_loss:.4f}')

        writer.close()

        last_model_copy = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_model_state)
        self.save_checkpoint(self.model.state_dict(), optimizer, scheduler, scaler, epoch, best_valid_loss, path="finishedBest.pt")

        self.save_checkpoint(last_model_copy, optimizer, scheduler, scaler, epoch, best_valid_loss, path="finishedLast.pt")

    def inference(self, model, tokenizer, sentence, pad_token_id, device="cuda"):
        model.eval()

        input_ids = tokenizer.encode(sentence)
        input_ids_tensor = torch.tensor([input_ids]).to(device)

        attention_mask = (input_ids_tensor != pad_token_id).long()

        mask_token_id = tokenizer.mask_token_id

        mask_indices = [i for i, token in enumerate(input_ids) if token == mask_token_id]
        if not mask_indices:
            raise ValueError("<mask> token not found in input. Make sure tokenizer uses correct <mask> ID.")

        with torch.no_grad():
            logits = model(input_ids_tensor, attention_mask)

        predicted_tokens = []
        for idx in mask_indices:
          pred_token_id = logits[0, idx].argmax().item()
          predicted_tokens.append(tokenizer.decode([pred_token_id]))

        return predicted_tokens if len(mask_indices) > 1 else predicted_tokens[0]


    def save_checkpoint(self, model, optimizer, scheduler, scaler, epoch, best_valid_loss, path="checkpoint.pt"):
        torch.save({
            "model_state_dict": model,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_valid_loss": best_valid_loss,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, optimizer, scheduler, scaler, model=None, path="finished.pt"):
        checkpoint = torch.load(path)

        if not model:
          self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
          model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        epoch = checkpoint["epoch"]
        best_valid_loss = checkpoint["best_valid_loss"]

        print(f"Checkpoint loaded from {path}")
        return epoch, best_valid_loss