import numpy as np

#create matrix of random values
def initialize_matrix(lines, columns):
    embedding_matrix = np.random.rand(lines, columns) * 0.01
    return embedding_matrix

#get embeddings for the given inputs from the embedding matrix
def get_embeddings(inputs):
    embedding_matrix = np.load("model/embedding_matrix.npy")
    return embedding_matrix[inputs].reshape(inputs.shape[0], -1)

#does a matric multiplication of the embeddings and the output matrix to get the scores for each token in the vocabulary
#(embedding vectors which is a concatenation of the input vectors) * (output matrix) = (scores for each token in the vocabulary)
def get_scores(embeddings):
    output_matrix = np.load("model/output_matrix.npy")
    return embeddings @ output_matrix

#use the function exp(scores) / sum(exp(score)) to convert the scores into probabilities
def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)

#calculate the loss between the predictions and the targets
#use the function -sum(targets * log(predictions)) / number of examples to calculate the loss
def loss_function(predictions, targets):
    pass