import json
import re
import config
import paths

#Take a sentence and a vocab dictionary and return a corpus of the sentence.
def split_text(text):
    tokens = re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿ]+|\d+|[,.!?:;()\[\]{}\"']",
        text
    )

    corpus = []

    for word in tokens:
      if word.isalnum():
        word = "▁" + word
      corpus.append(list(word))

    return corpus

#add new tokens to the vocab dictionary if they are not already present in the vocab.
def add_new_tokens(corpus, vocab):
    
    SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<USER>",
    "<ASSISTANT>"
    ]

    for token in SPECIAL_TOKENS:
      if token not in vocab:
          vocab[token] = len(vocab)

    for word in corpus:
        for char in word:
            if char not in vocab:
                max_key = max(
                    vocab.values(),
                    default=-1
                )
                vocab[char] = max_key + 1


#Take a corpus and return a tuple of the most frequent pair of letters in the corpus with their frequency count.
#If there are no pairs, return None.
def most_frequent_pair(corpus):
  map = {}
  for i in range(len(corpus)):
      word = corpus[i]
      for j in range(len(word)-1):
          letter_pair = (word[j], word[j+1])
          if letter_pair in map:
              map[letter_pair] += 1
          else:
              map[letter_pair] = 1
  if not map:
      return None
  return list(max(map.items(), key=lambda x: x[1])[0])

  #merge the most recurrent pair of letters in the corpus and update the corpus by replacing the pair with 
  # a single token of the merged pair
def bpe_function(corpus, recurrence):
  word = recurrence[0] + recurrence[1]

  for i in range(len(corpus)):
    word_list = corpus[i]
    new_corpus = []
    j = 0
    while j < len(word_list):
        if j+1 < len(word_list) and word_list[j] == recurrence[0] and word_list[j+1] == recurrence[1]:
          word_list[j] = word
          new_corpus.append(word_list[j])
          #jump 2 index 
          j += 2
        else:
          new_corpus.append(word_list[j])
          j += 1
    corpus[i] = new_corpus

#Use BPE function to tokenize a text and save the vocab and merge files.
#The function reads the input text from "data/input_text.txt", updates the vocabulary and the merge rules, and saves them in "tokenizer/vocab.json" and 
#"tokenizer/merge.json" respectively.
def tokenize_function(input_text):
 #read files
  with open( paths.VOCAB_PATH, "r", encoding="utf-8") as file:
    try:
        vocab = json.load(file)
    except json.JSONDecodeError:
        vocab = {}

  with open(paths.MERGE_PATH, "r", encoding="utf-8") as file:
      try:
          merged_rules = json.load(file)
      except json.JSONDecodeError:
          merged_rules = []
  
  #text into list of words each word is a list of characters lowercased, and the words are stored in a list called corpus.
  #special characters are added as a list of 1 special character.
  corpus = split_text(input_text)

  #add spécial tokens to vocab, then add each character in the corpus to the vocab if it is not already present.
  add_new_tokens(corpus, vocab)

  while len(vocab) < config.VOCAB_MAX_SIZE:
    #most recurrent pair of letters in the corpus with their frequency count.
    recurrence = most_frequent_pair(corpus)

    if recurrence is None:
      print("No more pairs to merge.")
      break

    #merge the most recurrent pair of letters in the corpus and update the corpus by replacing the pair with 
    # a single token of the merged pair
    bpe_function(corpus, recurrence)

    #take the pair
    pair = recurrence[0] + recurrence[1]
    merged_pair = [recurrence[0], recurrence[1]]

    if pair not in vocab:
      max_id = max(vocab.values(), default=-1) 
      vocab[pair] = max_id + 1   
      if merged_pair not in merged_rules:
        merged_rules.append(merged_pair)

          

  with open(paths.VOCAB_PATH, "w", encoding="utf-8") as file:
    json.dump(vocab, file, indent=4, ensure_ascii=False)
  
  with open(paths.MERGE_PATH, "w", encoding="utf-8") as file:
    json.dump(merged_rules, file, indent=4, ensure_ascii=False)

  return vocab
