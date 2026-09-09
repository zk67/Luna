from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 2000
    hidden_size: int = 224
    context_size: int = 128
    num_heads: int = 4
    intermediate_size: int = 896
    num_layers: int = 4
    dropout: float = 0.1

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 0.0001
    weight_decay: float = 0.01  #to regulate the model's complexity in the optimizer and prevent overfitting
    epochs: int = 10
    grad_clip: float = 1.0 #Gradient clipping value to prevent exploding gradients
    train_split: float = 0.9   #Proportion of data to use for training so 0.1 is used for validation
    seed: int = 42 #Random seed for reproducibility

@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.8 #factor to change the scores of the logits before applying softmax
    top_k: int | None = 40 #limit the consideration to the 40 most likely next tokens (40 logits, 40 scores)


#to check the validation of the formula: head_dim = hidden_size // num_heads
# which is a requirement for the multi-head attention mechanism in the transformer architecture.
def validate(self):
    if self.hidden_size % self.num_heads != 0:
        raise ValueError(...)