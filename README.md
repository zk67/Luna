# Luna AI

A tiny language model built **entirely from scratch in Python and NumPy**, no PyTorch, no TensorFlow, no pre-trained weights. Luna implements the core building blocks of a language model (a custom BPE tokenizer, an embedding layer, a forward/backward pass, and gradient descent) and exposes them through an interactive CLI for training and chatting.

This project was built as a learning exercise to understand, at the implementation level, how a language model actually works under the hood.

## What Luna can do

- Learn a vocabulary from a text corpus using a **Byte Pair Encoding (BPE) tokenizer** implemented from scratch
- Encode text into token IDs and decode token IDs back into text
- Train a small neural network (embedding matrix + output matrix) using **manual forward pass, cross-entropy loss, and gradient descent** (no autodiff library)
- Generate answers token-by-token from a trained model
- Reconfigure, retrain, or reset the model entirely through an interactive terminal menu

In its current state, Luna is trained on a small Q&A corpus about world capitals and continents (e.g. `what is the capital of france ? paris.`) and answers short, single-token-style questions in that fixed format.

## Why it's built this way

Luna intentionally avoids deep learning frameworks. Every core mechanism — tokenization, embeddings, matrix multiplication for scoring, softmax, cross-entropy loss, and gradient updates — is implemented by hand with NumPy. The goal is to understand each step of the pipeline, not to build a production-ready chatbot.

**Known limitations (by design, at this stage):**
- No attention mechanism — the model treats its fixed-size context window as a single flat vector, without the ability to dynamically weigh which token matters most
- Single linear layer (embeddings → scores), no hidden layers or non-linearity yet
- Fixed context window (`CONTEXT_SIZE`), with left-padding for short prompts and truncation for long ones
- Best suited for short, structured questions — free-form conversation is out of scope for now

## Project structure

```
Luna/
├── main.py                  # CLI entry point: menus, chat loop, training controls
├── config.py                # Model & training hyperparameters (auto-updated by the CLI)
├── paths.py                 # Centralized file paths
├── data/
│   └── corpus.txt           # Training corpus (question ? answer. per line)
├── tokenizer/
│   ├── tokenizer.py         # BPE training: builds vocab.json and merge.json from the corpus
│   ├── encoder.py           # encode_token_ids() / decode_token_ids()
│   ├── vocab.json           # Learned vocabulary (token -> id)
│   └── merge.json           # Learned BPE merge rules
├── training/
│   ├── model.py             # Embeddings, scores, softmax, loss, gradient descent
│   ├── dataset.py           # Builds (context, target) training pairs from token ids
│   └── train_model.py       # Training loop (batching, loss tracking, checkpointing)
├── generation/
│   └── generation.py        # Autoregressive token-by-token generation
└── model/matrices/
    ├── embedding_matrix.npy # Learned embedding matrix
    └── output_matrix.npy    # Learned output (scoring) matrix
```

## How it works

**1. Tokenization (BPE, from scratch)**
`tokenizer/tokenizer.py` pre-splits the corpus into words and characters, then iteratively merges the most frequent adjacent pair of tokens — the classic Byte Pair Encoding algorithm — until no more useful merges remain. The resulting vocabulary and merge rules are saved to `vocab.json` and `merge.json`, and reused at encoding time to tokenize any new input consistently.

**2. Model architecture**
For a context of `CONTEXT_SIZE` token IDs:
1. Each token ID is looked up in the **embedding matrix** (`VOCAB_SIZE x EMBEDDING_SIZE`) and the resulting vectors are concatenated into a single flat vector.
2. That vector is multiplied by the **output matrix** (`EMBEDDING_SIZE * CONTEXT_SIZE x VOCAB_SIZE`) to produce a raw score for every token in the vocabulary.
3. **Softmax** turns those scores into a probability distribution over the vocabulary.

**3. Training**
For every `(context, target)` pair generated from the corpus, `train_model.py` runs a forward pass, computes the **cross-entropy loss** against the true next token, and backpropagates the gradients manually to update both the embedding and output matrices via gradient descent. The matrices are only saved to disk when a new best average loss is reached.

**4. Generation**
`generation.py` starts from a user prompt, encodes it into token IDs, pads or truncates it to `CONTEXT_SIZE`, and then repeatedly: runs a forward pass, picks the most likely next token (`argmax` over the softmax output), appends it to the context (sliding window), and repeats until `max_token` tokens have been generated.

## Getting started

**Requirements:** Python 3.12+, NumPy

```bash
pip install numpy
python main.py
```

From the main menu you can:
- **Chat with Luna** — ask a question in the trained format (e.g. `what is the capital of france ?`) (e.g. `capital of france ?`) (e.g. `what is the continent of france ?`)(e.g. `continent of france ?`)
- **Train the AI or change parameters** — train on the current corpus, load a new corpus (which resets the vocabulary and matrices), tweak training hyperparameters, or reset the model from scratch

## Example

```
Question on a country or a capital ?: what is the capital of france ?
Luna: paris.
```

## Roadmap / possible next steps

- Add a hidden layer with a non-linear activation to increase model capacity
- Implement a basic self-attention mechanism
- Move from single-token answers to full generated sentences (requires fully autoregressive training over multi-token targets)
- Expand and diversify the training corpus

## Author

Zakaria Soudaki
