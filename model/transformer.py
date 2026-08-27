import torch
import torch.nn as nn
import math


def apply_rope(x):
    batch_size, num_heads, sequence_length, head_dim = x.shape

    positions = torch.arange(
        sequence_length,
        device=x.device
    )

    inv_freq = 1.0 / (
        10000 ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=x.device
            ).float() / head_dim
        )
    )

    angles = torch.outer(positions, inv_freq)

    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    x_rotated_even = x_even * cos - x_odd * sin
    x_rotated_odd = x_even * sin + x_odd * cos

    x_rotated = torch.stack(
        [x_rotated_even, x_rotated_odd],
        dim=-1
    )

    return x_rotated.flatten(-2)

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

        Q = apply_rope(Q)
        K = apply_rope(K) 

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
        x = self.token_embedding(input_ids)

        for layer in self.layers:
            # Matches the streamlined block signature now
            x = layer(attention_mask, causal_mask, x)

        logits = self.lm_head(x)

        return logits
