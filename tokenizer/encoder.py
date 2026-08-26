from importlib.resources import path
import paths
from tokenizer.tokenizer import split_text, bpe_function, add_new_tokens
import json

#function that encodes the input text into token ids using the vocabulary and merged rules stored
#in "tokenizer/vocab.json" and "tokenizer/merge.json" respectively.
def encode_prompt(input_text):

    #read files
    with open(paths.VOCAB_PATH, 'r', encoding='utf-8') as file:
        try:
            vocab = json.load(file)
        except json.JSONDecodeError:
            vocab = {}

    with open(paths.MERGE_PATH, 'r', encoding='utf-8') as file:
        try:
            merged_rules = json.load(file)
        except json.JSONDecodeError:
            merged_rules = []

    corpus = split_text(input_text) 

    for rule in merged_rules:
        bpe_function(corpus, rule)

    print("Corpus after applying BPE:", corpus)

    key_corpus = []

    for word in corpus:
        for token in word:
            if token in vocab:
                key = vocab[token]
                key_corpus.append(key)
            else:
                key_corpus.append(vocab["<UNK>"])

    #not necessary to add the special tokens during training they are already in the prompt
    # key_corpus.insert(0, vocab["<USER>"])
    # key_corpus.append(vocab["<ASSISTANT>"])
    
    return key_corpus

#function that decodes the token ids back into text using the vocabulary stored in "tokenizer/vocab.json".
def decode_token_ids(token_ids):
    with open(paths.VOCAB_PATH, 'r', encoding='utf-8') as file:
        try:
            vocab = json.load(file)
        except json.JSONDecodeError:
            vocab = {}
    
    reverse_vocab = {v: k for k, v in vocab.items()}
    sentence = ""

    for token_id in token_ids:
        if token_id in reverse_vocab:
            sentence += reverse_vocab[token_id]
        else:
            sentence += "<UNK> "
    

    return sentence.replace("▁", " ").strip()