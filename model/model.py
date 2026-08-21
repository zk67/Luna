import numpy as np

#create matrix of random values
def initialize_matrix(lines, columns):
    matrix = np.random.rand(lines, columns) * 0.01
    return matrix

#get embeddings for the given inputs from the embedding matrix
def get_embeddings(inputs, embedding_matrix):
    return embedding_matrix[inputs].reshape(inputs.shape[0], -1)

#does a matric multiplication of the embeddings and the output matrix to get the scores for each token in the vocabulary
#(embedding vectors which is a concatenation of the input vectors) * (output matrix) = (scores for each token in the vocabulary)
def get_scores(embeddings, output_matrix):
    return embeddings @ output_matrix

#use the function exp(scores) / sum(exp(score)) to convert the scores into probabilities
def softmax(scores):
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

#calculate the loss between the predictions and the targets
#use the function -sum(targets * log(predictions)) / number of examples to calculate the loss
def loss_function(softmax_scores, targets):
    target_probs = softmax_scores[np.arange(len(targets)), targets]
    loss_vector = -np.log(target_probs + 1e-9)
    return np.mean(loss_vector)

#calculate the dscores of each example in the batch, which is the difference between the
def get_dscores(softmax_scores, targets):
    dscores = softmax_scores.copy()
    dscores[np.arange(len(targets)), targets] -= 1
    dscores /= len(targets)
    return dscores

def gradient_descent(embedding_matrix, output_matrix, dscores, embeddings, learning_rate, inputs):
    dW2 = embeddings.T @ dscores
    dW1 = dscores @ output_matrix.T

    dW1 = dW1.reshape(inputs.shape[0], inputs.shape[1], -1)

    embedding_matrix[inputs] -= learning_rate * dW1
    output_matrix -= learning_rate * dW2

    # print("dW1 mean:", np.mean(np.abs(dW1)))
    # print("dW2 mean:", np.mean(np.abs(dW2)))

    return embedding_matrix, output_matrix

    
