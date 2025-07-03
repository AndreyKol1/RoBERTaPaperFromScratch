import torch
from torch.utils.data import Dataset
from nltk import sent_tokenize

class PreprocessDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.texts = dataset["text"]
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.MASK_TOKEN_ID = tokenizer.mask_token_id
        self.PAD_TOKEN_ID = tokenizer.pad_token_id
        self.CLS_TOKEN_ID = tokenizer.cls_token_id
        self.SEP_TOKEN_ID = tokenizer.sep_token_id

        self.all_chunks = []
        for article in self.texts:
           self.all_chunks.extend(self._chunk(article))

    def __len__(self):
        return len(self.all_chunks)
    
    def __getitem__(self, idx):
        sequence_chunked = self.all_chunks[idx]
        attention_mask = [1 if token != self.PAD_TOKEN_ID else 0 for token in sequence_chunked]
        masked_input_ids, labels = self.mask_tokens(torch.tensor(sequence_chunked, dtype=torch.long))
          
        return {
            "input_ids" : masked_input_ids,
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": labels
        }

    def _chunk(self, article):
        sentences = sent_tokenize(article)

        chunks = []

        current_chunk = [self.CLS_TOKEN_ID]

        for sentence in sentences:
            tokenized = self.tokenizer.encode(sentence)

            if len(tokenized) > self.max_length - 2: # first check if tokenized sentences is more than self.maxlength - 2 in length
                truncated = tokenized[:self.max_length-2]
                chunk = [self.CLS_TOKEN_ID] + truncated + [self.SEP_TOKEN_ID]
                chunks.append(chunk)
                continue

            if len(current_chunk) + len(tokenized) + 1 > self.max_length: # second check when stack senteces exceed limit how many PAD when need 
                current_chunk.append(self.SEP_TOKEN_ID)
                current_chunk += [self.PAD_TOKEN_ID] * (self.max_length - len(current_chunk))
                chunks.append(current_chunk)

                current_chunk = [self.CLS_TOKEN_ID]
            
            current_chunk.extend(tokenized)

        if len(current_chunk) > 1: # check after last iteration if there something left in current_chunk
            current_chunk.append(self.SEP_TOKEN_ID)
            current_chunk += [self.PAD_TOKEN_ID] * (self.max_length - len(current_chunk))
            chunks.append(current_chunk)

        if not chunks: # if chunks are empty all sequence is padded
          return [[self.CLS_TOKEN_ID, self.SEP_TOKEN_ID] + [self.PAD_TOKEN_ID] * (self.max_length - 2)]
        else:
          return chunks
    
    def mask_tokens(self, input_ids): # function for dynamic masking 
        labels = input_ids.clone()
        orig_input_ids = input_ids.clone()
        
        idx = [i for i, index in enumerate(input_ids) if index not in [self.CLS_TOKEN_ID, self.SEP_TOKEN_ID, self.PAD_TOKEN_ID]]
        if len(idx) == 0: 
          labels[:] = -100
          return input_ids, labels

        if int((len(idx)*0.15)) == 0: # case, where senteces is too short
          labels[:] = -100
          return input_ids, labels

        idx = torch.tensor(idx, dtype=torch.long)
        idx_to_mask = torch.multinomial(torch.ones(len(idx)), int(len(idx) * 0.15), replacement=False) # select 15% of indexes to mask
        probs = torch.rand(len(idx_to_mask)) # range of numbers to further pick from
        mask_with_word = probs < 0.8
        mask_with_random = (probs < 0.9) & (probs>= 0.8)
        to_keep = probs >= 0.9

        indices_to_mask_word = idx_to_mask[mask_with_word] # 80% for MASK token
        indices_to_mask_random = idx_to_mask[mask_with_random] # 10% for random word from vocab
        indices_to_keep = idx_to_mask[to_keep] # 10% stays the same

        input_ids[indices_to_mask_word] = self.MASK_TOKEN_ID
        input_ids[indices_to_mask_random] = torch.randint(low=0, high=len(self.tokenizer.vocab), size=(len(indices_to_mask_random),), dtype=torch.long)
        
        labels[:] = -100
        position_to_mask = torch.cat([indices_to_mask_word, indices_to_mask_random, indices_to_keep]) # select positions for MLM
        labels[position_to_mask] = orig_input_ids[position_to_mask]

        return input_ids, labels