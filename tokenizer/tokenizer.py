import json
import re;
import paths

#Take a sentence and a vocab dictionary and return a corpus of the sentence.
def tokenize_text(text):
    tokens = re.findall(
        r"[A-Za-z]+|\d+|[,.!?:;()\[\]{}\"']",
        text.lower()
    )

    corpus = []

    for word in tokens:
        corpus.append(list(word))

    return corpus

#add new tokens to the vocab dictionary if they are not already present in the vocab.
def add_new_tokens(corpus, vocab):
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

#function that takes a corpus and a recurrence (a rule from the merge.json file) and merges the two letter or 
#group of letters in the corpus into a single group.
def bpe_function(corpus, recurrence):
  word = recurrence[0] + recurrence[1]

  for i in range(len(corpus)):
    corp = corpus[i]
    new_corpus = []
    j = 0
    while j < len(corp):
        if j+1 < len(corp) and corp[j] == recurrence[0] and corp[j+1] == recurrence[1]:
          corp[j] = word
          new_corpus.append(corp[j])
          #jump 2 index 
          j += 2
        else:
          new_corpus.append(corp[j])
          j += 1
    corpus[i] = new_corpus

#Use BPE function to tokenize a text and save the vocab and merge files.
#The function reads the input text from "data/corpus.txt", updates the vocabulary and the merge rules, and saves them in "tokenizer/vocab.json" and 
#"tokenizer/merge.json" respectively.
def tokenize_function():

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

  with open(paths.CORPUS_PATH, "r", encoding="utf-8") as file:
    input_text = file.read()
  
  #pre-tokenisation
  corpus = tokenize_text(input_text.lower())
  add_new_tokens(corpus, vocab)

  tokens = []
  print(corpus)

  while True:

    #most recurrent pair of letters in the corpus with their frequency count.
    recurrence = most_frequent_pair(corpus)

    print(recurrence)

    if recurrence is None:
      print("No more pairs to merge.")
      break

    #merge the most recurrent pair of letters in the corpus.
    bpe_function(corpus, recurrence)

    print(corpus)

    #if new word, add to vocab
    word = recurrence[0] + recurrence[1]
    if word not in vocab:
      tokens.append(word)

    merge_list = [recurrence[0], recurrence[1]]
    
    if merge_list not in merged_rules:
      merged_rules.append(merge_list)

 #end loop, save data in files

  for token in tokens:
    if vocab.get(token) is None:
        max_key = max(vocab.keys(), key=lambda x: vocab[x],default=None)
        if max_key is None:
            vocab[token] = 0
        else:
          vocab[token] = vocab[max_key] + 1
          

  with open(paths.VOCAB_PATH, "w", encoding="utf-8") as file:
    json.dump(vocab, file, indent=4, ensure_ascii=False)
  
  with open(paths.MERGE_PATH, "w", encoding="utf-8") as file:
    json.dump(merged_rules, file, indent=4, ensure_ascii=False)

  return vocab