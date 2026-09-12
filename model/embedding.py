
from torch import nn
import torch

class TokenPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, context_size, dropout_rate):
        super(TokenPositionEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(context_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    #forward method takes input_ids,
    # generates position_ids,
    # retrieves token and position embeddings,
    # sums them, applies dropout
    # and returns the final embeddings.
    def forward(self, input_ids):
        num_tokens = input_ids.size(1)
        position_ids = torch.arange(num_tokens, dtype=torch.long, device=input_ids.device)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        return x

    