import torch
from torch import nn

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate ):
        super().__init__()
        if (hidden_size % num_heads) != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        self.head_dim = hidden_size // num_heads
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # Linear layers for query, key, value, and output projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Dropout layers for attention and residual connections
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.residual_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask):
        Q = self.q_proj(x)
        K = self.k_proj(x) 
        V = self.v_proj(x)

        # Reshape Q, K, V for multi-head attention
        Q = Q.reshape(Q.size(0), Q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(K.size(0), K.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(V.size(0), V.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))

        # Create a causal mask
        # create a matrice in the size size_of_sequence x size_of_sequence with 1's in the 
        # lower triangle and 0's in the upper triangle 
        # then unsqueeze it to have the shape (1, 1, size_of_sequence, size_of_sequence) to
        # be compatible with the attention scores
        causal_mask = torch.tril(torch.ones(attention_scores.size(-2), attention_scores.size(-1), device=attention_scores.device)).unsqueeze(0).unsqueeze(0)

        #scaling the attention scores by the square root of the head dimension
        attention_scores = attention_scores / (self.head_dim ** 0.5)

        # Apply the padding attention mask to ignore padding tokens in the input sequences, setting 
        # their attention scores to negative infinity
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_length)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        # Apply the causal mask to the attention scores to prevent attending to future tokens
        # by setting their attention scores to negative infinity
        attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))

        # Compute attention weights using softmax to get a probability distribution over the tokens
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # Apply dropout to the attention weights for regularization
        attention_probs = self.attention_dropout(attention_probs)

        attention_output = torch.matmul(attention_probs, V)

        attention_output = attention_output.transpose(1, 2).reshape(
            attention_output.size(0),
            attention_output.size(2),
            self.hidden_size
        )

        # Project the attention output back to the original hidden size
        attention_output = self.out_proj(attention_output)

        # Apply dropout to the output of the attention layer for regularization
        attention_output = self.residual_dropout(attention_output)

        return attention_output
