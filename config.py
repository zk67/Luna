import json

with open("tokenizer/vocab.json", "r", encoding="utf-8") as file:
    vocab = json.load(file)


VOCAB_SIZE = len(vocab)
EOS_TOKEN_ID = vocab["<EOS>"]
PAD_TOKEN_ID = vocab["<PAD>"]
HIDDEN_SIZE = 128
CONTEXT_SIZE = 64
HEADS = 4
INTERMEDIATE_SIZE = 512
LAYERS = 6
LEARNING_RATE = 0.0003
BATCH_SIZE = 8
TRAINING_LOOPS = 10
TEMPERATURE = 0.8
