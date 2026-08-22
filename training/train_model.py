import json

import numpy as np
from training.dataset import create_training_examples
from training.model import get_embeddings, get_scores, softmax, loss_function, get_dscores, gradient_descent
from tokenizer.encoder import encode_token_ids
import config
import paths

#function that trains the model using the training examples created from the token ids and updates the embedding 
#and output matrices using gradient descent
def train_model(loop=config.TRAINING_LOOPS, learning_rate=config.LEARNING_RATE, batch_size=config.BATCH_SIZE, best_loss=config.BEST_LOSS):

    embedding_matrix = np.load(paths.EMBEDDING_MATRIX_PATH)
    output_matrix = np.load(paths.OUTPUT_MATRIX_PATH)

    with open(paths.CORPUS_PATH, "r", encoding="utf-8") as file:
        corpus = file.read()

    token_ids = encode_token_ids(corpus)
    inputs, targets = create_training_examples(token_ids)

    if batch_size is None:
            batch_size = len(inputs)

    for i in range(loop):
        total_loss = 0
        number_of_batches = 0
        average_batch_loss = 0
        for start in range(0, len(inputs), batch_size):
            batch_inputs = inputs[start:start + batch_size]
            batch_targets = targets[start:start + batch_size]

            # forward
            embeddings = get_embeddings(batch_inputs, embedding_matrix)
            scores = get_scores(embeddings, output_matrix)
            softmax_scores = softmax(scores)

            # loss
            average_loss = loss_function(softmax_scores, batch_targets)
            total_loss += average_loss
            number_of_batches += 1

            # backward
            dscores = get_dscores(softmax_scores, batch_targets)

            # update matrices using gradient descent
            embedding_matrix, output_matrix =gradient_descent(embedding_matrix=embedding_matrix, 
                            output_matrix=output_matrix, 
                            dscores= dscores, 
                            embeddings= embeddings, 
                            learning_rate= learning_rate,
                            inputs= batch_inputs)
            
        average_batch_loss = total_loss / number_of_batches
        print("Average Loss per batch: ", average_batch_loss)

        if average_batch_loss < best_loss:
            best_loss = average_batch_loss

            #save
            np.save(paths.EMBEDDING_MATRIX_PATH, embedding_matrix)
            np.save(paths.OUTPUT_MATRIX_PATH, output_matrix)
