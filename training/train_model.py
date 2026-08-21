import numpy as np
import json
from training.dataset import create_training_examples
from model.model import initialize_matrix, get_embeddings, get_scores, softmax, loss_function, get_dscores, gradient_descent
from tokenizer.encoder import encode_token_ids

#function that trains the model using the training examples created from the token ids and updates the embedding 
#and output matrices using gradient descent
def train_model(loop=10, learning_rate=0.01):

    #embedding_matrix = initialize_matrix(lines= 1844, columns= 8)
    #output_matrix = initialize_matrix(lines= 24, columns= 1844)

    embedding_matrix = np.load("model/matrices/embedding_matrix.npy")
    output_matrix = np.load("model/matrices/output_matrix.npy")

    token_ids = encode_token_ids()
    inputs, targets = create_training_examples(token_ids)

    print("tokenids:", len(token_ids))
    best_loss = float("inf")

    for i in range(loop):
        # forward
        embeddings = get_embeddings(inputs, embedding_matrix)
        scores = get_scores(embeddings, output_matrix)
        softmax_scores = softmax(scores)

        # loss
        average_loss = loss_function(softmax_scores, targets)
        print("Loss:", average_loss)

        # backward
        dscores = get_dscores(softmax_scores, targets)

        # update matrices using gradient descent
        embedding_matrix, output_matrix =gradient_descent(embedding_matrix=embedding_matrix, 
                        output_matrix=output_matrix, 
                        dscores= dscores, 
                        embeddings= embeddings, 
                        learning_rate= learning_rate,
                        inputs= inputs)

    if average_loss < best_loss:
        best_loss = average_loss

        #save
        np.save("model/matrices/embedding_matrix.npy", embedding_matrix)
        np.save("model/matrices/output_matrix.npy", output_matrix)