# MiniRoBERTa (17.7M Parameters)

This project implements a Transformer-based language model from scratch, inspired by RoBERTa. The goal was to develop a deeper understanding of Transformer internals and apply that knowledge by building and training a small-scale RoBERTa-like model on the WikiText dataset, followed by fine-tuning on the SST-2 sentiment classification task.

---

## 📌 Project Highlights

- ✅ **Built from scratch**: Architecture, tokenizer, training loop
- 📊 **17.7M parameters**
- 📚 **Pretrained** on WikiText for 10 epochs
- 🔄 **Fine-tuned** on SST-2 → **80% accuracy**
- 🧪 **Inference** via Gradio app
- 🛠️ **Single GPU training**, resource-efficient

---

## 🧱 Model Architecture

| Component              | Value       |
|------------------------|-------------|
| `hidden_size`          | 256         |
| `num_attention_heads`  | 8           |
| `attention_head_dim`   | 32          |
| `num_layers`           | 6           |
| `max_seq_length`       | 128         |
| `vocab_size`           | 50265       |

---

## 🔧 Pretraining Configuration

- **Dataset**: WikiText (via `Salesforce/wikitext`)
- **Optimizer**: Adam (`betas=(0.98, 0.999)`, `eps=1e-6`)
- **Learning Rate**: 1e-6
- **Scheduler**: Cosine decay with 10% warmup
- **Epochs**: 10
- **Batch Size**: 32
- **Loss Function**: Masked Language Modeling (MLM)

---

## 🚀 Inference

Base Model: https://huggingface.co/spaces/DornierDo17/RoBERTaModelPreTrained
Fine-Tuned Model: https://huggingface.co/spaces/DornierDo17/MiniRoBERTaFineTuned

## 🤗 Model on Hugging Face

Base Model: https://huggingface.co/DornierDo17/RoBERTa_17.7M
Fine-Tuned Model: https://huggingface.co/DornierDo17/MiniRoBERTa_SST2

## Changes made to transition from BERT model to RoBERTa

1. Dynamic masking instead of static masking (as used in BERT).
2. Removed Next Sentence Prediction (NSP) — sequences are packed with full sentences, without the NSP loss.

## Experiments

1. 🔤 Custom Tokenizer Experiment 
Initially I questioned myself whether the model will be able to give some meanigfull results with custom built tokenizer. 
- After 12 hours of pretraining, I observed that the results were not promising enough, likely due to noisy tokenization and limited compute.
- Since I haven't had resources to make it clean after 12 hours of pre-training and observing results using custom tokenizer, I decided to not stick with this idea.
- Based on these observations, I decided to drop the custom tokenizer and switch to a more stable solution.

2. 🚀 Transition to **RobertaTokenizerFast**
- I modified my data preprocessing logic and adopted RobertaTokenizerFast from the Hugging Face ecosystem.
- The model performance improved significantly:
  - Loss dropped from 9.5 to 2.6 (training and validation) over 10 epochs.
  - Custom inference tests began yielding meaningful predictions.
- These results confirmed that my custom MiniRoBERTa model was learning successfully.

3. 🎯 Fine-Tuning on SST-2
- Post pretraining, I fine-tuned the model on the SST-2 sentiment classification dataset, following the original RoBERTa paper.
- Despite having only 17.7 million parameters, the model reached 80% accuracy — a strong result.
- For comparison:
  - RoBERTa-Base has ~125M parameters
  - My model is 10x smaller, and yet only 15% behind in accuracy

## ✅ Summary

A small, from-scratch RoBERTa-style model trained on limited resources can still learn meaningful representations and adapt to downstream tasks.

