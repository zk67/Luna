import numpy as np

#function to create training examples from token ids (inputs 3) --> (target 1)
def create_training_examples(token_ids, context_size =3):
    inputs = []
    targets = []

    for i in range(len(token_ids) - context_size):
        inputs.append(token_ids[i:i + context_size])
        targets.append(token_ids[i + context_size])
    return np.array(inputs), np.array(targets)