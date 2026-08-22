import numpy as np
import config

#function to create training examples from token ids (number of inputs = CONTEXT_SIZE) --> (target 1)
def create_training_examples(token_ids, context_size = config.CONTEXT_SIZE):
    inputs = []
    targets = []

    for i in range(len(token_ids) - context_size):
        inputs.append(token_ids[i:i + context_size])
        targets.append(token_ids[i + context_size])
    return np.array(inputs), np.array(targets)