from tokenizer.encoder import encode_token_ids, decode_token_ids
import config
import numpy as np
import paths
from training.model import get_embeddings, get_scores, softmax

#function that generates text based on the given prompt and the trained model.
#It uses the embedding and output matrices to predict the next token in the sequence, and continues generating
# tokens until it reaches the specified maximum number of tokens.
def generate(prompt, max_token=2):

    embedding_matrix = np.load(paths.EMBEDDING_MATRIX_PATH)
    output_matrix = np.load(paths.OUTPUT_MATRIX_PATH)

    encoded_prompt = encode_token_ids(prompt)
    if len(encoded_prompt) < config.CONTEXT_SIZE:
        encoded_prompt = [0] * (config.CONTEXT_SIZE - len(encoded_prompt)) + encoded_prompt
    if len(encoded_prompt) > config.CONTEXT_SIZE:
        encoded_prompt = encoded_prompt[-config.CONTEXT_SIZE:]
    
    context = np.array(encoded_prompt[-config.CONTEXT_SIZE:])
    context = context.reshape(1, -1)

    generated_tokens = []
    i = 0

    while i < max_token:
        embeddings = get_embeddings(context, embedding_matrix)
        scores = get_scores(embeddings, output_matrix)
        softmax_scores = softmax(scores)

        #print the top 3 token ids and their probabilities
        top_indices = np.argsort(softmax_scores[0])[-10:][::-1]
        for index in top_indices:
            print(
                "Token ID:", index,
                "Probability:", softmax_scores[0][index]
            )

        token = np.argmax(softmax_scores)
        generated_tokens.append(token)
        if(i +1 == max_token):
            return decode_token_ids(generated_tokens)

        context = np.append(context, token)
        context = context[-config.CONTEXT_SIZE:]
        context = context.reshape(1, -1)

        i += 1




