from training.dataset import create_training_examples
from model.model import initialize_matrix, get_embeddings, get_scores, softmax
import numpy as np

def main():

    token_ids = [36, 34, 32, 44, 47, 25]

    inputs, targets = create_training_examples(token_ids)
    print("Inputs:", inputs)
    print("Targets:", targets)

    embeddings = get_embeddings(inputs)
    scores = get_scores(embeddings) 
    # scores = get_scores(embeddings)
    # print("Scores:", scores)
    # softmax_scores = softmax(scores)
    # print("Softmax Scores:", softmax_scores)

    # print("Max Softmax Score:", np.max(softmax_scores))
    # print("Index of Max Softmax Score:", np.argmax(softmax_scores))


if __name__ == "__main__":
    main()