import json
import re
import paths

WORD_START = "▁"
TOKEN_PATTERN = r"<PAD>|<UNK>|<BOS>|<EOS>|<USER>|<ASSISTANT>|[A-Za-zÀ-ÖØ-öø-ÿ]+|\d+|[,.!?:;()\[\]{}\"']"

SPECIAL_TOKENS = [
"<PAD>",
"<UNK>",
"<BOS>",
"<EOS>",
"<USER>",
"<ASSISTANT>"
]

class BPETokenizer:
  def __init__(self):
    self.vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    self.merged_rules = []
    self.vocab_size = len(self.vocab) 

  @property
  def pad_id(self):
    return self.vocab["<PAD>"]
  @property
  def unk_id(self):
    return self.vocab["<UNK>"]
  @property
  def bos_id(self):
    return self.vocab["<BOS>"]
  @property
  def eos_id(self):
    return self.vocab["<EOS>"]
  @property
  def user_id(self):
    return self.vocab["<USER>"]
  @property
  def assistant_id(self):
    return self.vocab["<ASSISTANT>"]

  #------------INTERNAL METHODS------------

  #Take a sentence and a vocab dictionary and return a corpus of the sentence.
  def _split_text(self, text):
    tokens = re.findall(TOKEN_PATTERN, text)
    corpus = []

    for word in tokens:
      if word.isalnum():
        word = WORD_START + word
      corpus.append(list(word))
    return corpus


  #add new tokens to the vocab dictionary if they are not already present in the vocab.
  def _add_new_tokens(self, corpus):
    for word in corpus:
        for char in word:
            if char not in self.vocab:
                max_key = max(
                    self.vocab.values(),
                    default=-1
                )
                self.vocab[char] = max_key + 1

  #Take a corpus and return a tuple of the most frequent pair of letters in the corpus with their frequency count.
  #If there are no pairs, return None.
  def _most_frequent_pair(self, corpus):
    pair_counts = {}
    for i in range(len(corpus)):
        word = corpus[i]
        for j in range(len(word)-1):
            letter_pair = (word[j], word[j+1])
            if letter_pair in pair_counts:
                pair_counts[letter_pair] += 1
            else:
                pair_counts[letter_pair] = 1
    if not pair_counts:
        return None
    return list(max(pair_counts.items(), key=lambda x: x[1])[0])

  #Merge the most recurrent pair of letters in the corpus and update the corpus by replacing the pair with 
  # a single token of the merged pair
  def _apply_merge(self,corpus, pair):
    merged_token = pair[0] + pair[1]

    for i in range(len(corpus)):
      word_tokens = corpus[i]
      new_word_tokens = []
      j = 0
      while j < len(word_tokens):
          if j+1 < len(word_tokens) and word_tokens[j] == pair[0] and word_tokens[j+1] == pair[1]:
            word_tokens[j] = merged_token
            new_word_tokens.append(word_tokens[j])
            #jump 2 index 
            j += 2
          else:
            new_word_tokens.append(word_tokens[j])
            j += 1
      corpus[i] = new_word_tokens
      

  #------------PUBLIC METHODS------------
  
  #Trains the BPE tokenizer by splitting the input text into tokens, adding initial characters to the vocabulary,
  #repeatedly merging the most frequent token pairs, and storing the learned merge rules until the vocabulary size
  #limit is reached.
  def train(self, text, vocab_size_limit=2000):
    corpus = self._split_text(text)
    self._add_new_tokens(corpus)

    while len(self.vocab) < vocab_size_limit:
      pair = self._most_frequent_pair(corpus)

      if pair is None:
        print("No more pairs to merge.")
        break

      #merge the most recurrent pair of letters in the corpus and update the corpus by replacing the pair with 
      # a single token of the merged pair
      self._apply_merge(corpus, pair)

      #take the pair
      merged_token = pair[0] + pair[1]
      #take the merged rule as a list of the two letters in the pair
      merged_rule = [pair[0], pair[1]]

      #add them if not already present in the vocab and merged rules
      if merged_token not in self.vocab:
        max_id = max(self.vocab.values(), default=-1) 
        self.vocab[merged_token] = max_id + 1   
        if merged_rule not in self.merged_rules:
          self.merged_rules.append(merged_rule)

  # Encodes input text into token IDs using the trained vocabulary and BPE merge rules.
  def encode(self, input_text):    
    corpus = self._split_text(input_text) 
    for rule in self.merged_rules:
        self._apply_merge(corpus, rule)

    token_ids = []

    for word in corpus:
        for token in word:
            if token in self.vocab:
                token_id = self.vocab[token]
                token_ids.append(token_id)
            else:
                token_ids.append(self.vocab["<UNK>"])
    #not necessary to add the special tokens during training they are already in the prompt
    return token_ids

  
  # Decodes a sequence of token IDs back into text using the tokenizer vocabulary.
  def decode(self, token_ids):
    reverse_vocab = {v: k for k, v in self.vocab.items()}
    sentence = ""

    for token_id in token_ids:
        if token_id not in reverse_vocab:
            continue

        token = reverse_vocab[token_id]
        sentence += token
    return sentence.replace("▁", " ").strip()
  

  #saves the vocab and merged rules to json files in the tokenizer directory.
  def save(self):
    with open(paths.VOCAB_PATH, "w", encoding="utf-8") as vocab_file:
      json.dump(self.vocab, vocab_file, ensure_ascii=False, indent=4)

    with open(paths.MERGES_PATH, "w", encoding="utf-8") as merge_file:
      json.dump(self.merged_rules, merge_file, ensure_ascii=False, indent=4)

  #loads the vocab and merged rules from json files in the tokenizer directory.
  def load(self):
    with open(paths.VOCAB_PATH, "r", encoding="utf-8") as vocab_file:
      self.vocab = json.load(vocab_file)

    with open(paths.MERGES_PATH, "r", encoding="utf-8") as merge_file:
      self.merged_rules = json.load(merge_file)

