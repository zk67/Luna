import torch.nn as nn

class SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate

        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size) 
        self.down_proj = nn.Linear(intermediate_size, hidden_size) 

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Compute the up projection and gate projection
        up  = self.up_proj(x)
        gate = self.gate_proj(x)

        # Apply the SwiGLU activation function
        activated_gate = nn.functional.silu(gate)

        #combine the up projection and the gate projection using multiplication element by element
        hidden = activated_gate * up

        #apply the down projection to reduce the dimensionality back to the hidden size
        output = self.down_proj(hidden)

        # Apply dropout for regularization
        output = self.dropout(output) 
        return output
