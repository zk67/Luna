from tokenizer import sentence_to_corpus, bpe_function, add_new_tokens
import json

#function that encodes the input text into token ids using the vocabulary and merged rules stored
#in "tokenizer/vocab.json" and "tokenizer/merge.json" respectively. (unknown token_id is number 85)
def encode_token_ids():

    #read files
    with open('tokenizer/vocab.json', 'r', encoding='utf-8') as file:
        try:
            vocab = json.load(file)
        except json.JSONDecodeError:
            vocab = {}

    with open('tokenizer/merge.json', 'r', encoding='utf-8') as file:
        try:
            merged_rules = json.load(file)
        except json.JSONDecodeError:
            merged_rules = []

    with open('tokenizer/corpus_decode.txt', 'r', encoding='utf-8') as file:
        input_text = file.read()

    corpus = sentence_to_corpus(input_text) 

    print("Corpus:", corpus)

    for rule in merged_rules:
        bpe_function(corpus, rule)

    key_corpus = []

    for word in corpus:
        for token in word:
            if token in vocab:
                key = vocab[token]
                key_corpus.append(key)
            else:
                key_corpus.append(vocab["<UNK>"])

    print("Corpus:", corpus)
    print("Key Corpus:", key_corpus)
