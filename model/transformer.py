import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.w0 = nn.Linear(hidden_size, hidden_size)

        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)

        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(hidden_size, intermediate_size)
        self.w3 = nn.Linear(intermediate_size, hidden_size)

    # FIX 2: Removed unused input_ids parameter to keep clean signatures
    def forward(self, attention_mask, causal_mask, x):
        batch_size, sequence_length, hidden_size = x.shape    

        # Save identity for true Pre-LN residual connection
        residual = x
        normed_x = self.norm1(x)    

        # compute the query, key and value matrices for the attention mechanism
        Q = self.query(normed_x)
        K = self.key(normed_x)
        V = self.value(normed_x)

        # split into multi-heads
        Q = Q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)  

        # compute the attention scores
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # apply the causal mask and the attention mask
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # convert scores to probabilities and compute attention output
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output  = attention_weights @ V

        # reshape attention output back to sequence shape
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            self.hidden_size
        )

        # FIX 3: Corrected residual flow (add to the original un-normalized tensor)
        w0 = self.w0(attention_output)
        x = residual + w0

        # Feed-Forward network with another Pre-LN residual block
        residual_ffn = x
        normed_ffn_x = self.norm2(x)
        
        gate = torch.nn.functional.silu(self.w1(normed_ffn_x))
        value = self.w2(normed_ffn_x)
        ffn_output = gate * value
        ffn_output = self.w3(ffn_output)
        
        x = residual_ffn + ffn_output

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        context_size,
        num_heads=4,
        intermediate_size=512,
        num_layers=6
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(context_size, hidden_size)

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size,
                num_heads,
                intermediate_size
            )
            for _ in range(num_layers)
        ])  

        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask, causal_mask):
        sequence_length = input_ids.shape[1]
        positions = torch.arange(sequence_length, device=input_ids.device)

        token_embedding = self.token_embedding(input_ids)
        position_embedding = self.position_embedding(positions)

        x = token_embedding + position_embedding


        for layer in self.layers:
            # Matches the streamlined block signature now
            x = layer(attention_mask, causal_mask, x)

        logits = self.lm_head(x)

        return logits
