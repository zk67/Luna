import torch
from torch.utils.data import Dataset

class ChatDataset(Dataset):
    def __init__(self, records, tokenizer, context_size):
        self.records = records
        self.tokenizer = tokenizer
        self.context_size = context_size

        
    def __len__(self):
        return len(self.records)


    #function to encode a conversation into token ids, adding special tokens for the beginning of the sequence, user, assistant, and end of the sequence.
    def _encode_conversation(self, conversation):
        messages = conversation["messages"]
        conversation_tokens_ids = []
        user_content  = messages[0]["content"]
        assistant_content = messages[1]["content"]
        conversation_tokens_ids.append(self.tokenizer.bos_id)
        conversation_tokens_ids.append(self.tokenizer.user_id)
        conversation_tokens_ids.extend(self.tokenizer.encode(user_content))
        conversation_tokens_ids.append(self.tokenizer.assistant_id)
        conversation_tokens_ids.extend(self.tokenizer.encode(assistant_content))
        conversation_tokens_ids.append(self.tokenizer.eos_id)
        return conversation_tokens_ids

        
    #function to get a training example from the dataset, which includes input_ids, labels, and attention_mask for a given index.
    def __getitem__(self, idx):
        record = self.records[idx]
        conversation_tokens_ids = self._encode_conversation(record)
        conversation_tokens_ids = conversation_tokens_ids[:self.context_size + 1]
        input_ids = conversation_tokens_ids[:-1]  #remove the last token (EOS) for input
        labels = conversation_tokens_ids[1:]  #remove the first token (BOS) for labels

        input_ids.extend([self.tokenizer.pad_id] * (self.context_size - len(input_ids)))
        labels.extend([-100] * (self.context_size - len(labels)))

        attention_mask = [1 if token_id != self.tokenizer.pad_id else 0 for token_id in input_ids]

        # Mask the user and assistant tokens in the labels with -100 to ignore them during loss calculation.
        for i, token_id in enumerate(labels):
            if token_id == self.tokenizer.user_id:
                labels[i] = -100

            if token_id == self.tokenizer.assistant_id:
                break 

        torch_input_ids = torch.tensor(input_ids, dtype=torch.long)
        torch_labels = torch.tensor(labels, dtype=torch.long)
        torch_attention_mask = torch.tensor(attention_mask, dtype=torch.bool)

        return {
            "input_ids": torch_input_ids,
            "labels": torch_labels,
            "attention_mask": torch_attention_mask
        }