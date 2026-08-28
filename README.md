# Luna

A small language model built from the ground up: a custom Byte Pair Encoding tokenizer with no external tokenizer library, and a decoder-only Transformer trained with PyTorch. Luna is trained on a Q&A dataset covering programming and computer science fundamentals, and answers prompts through autoregressive, token-by-token generation.

This project was built to understand — at the implementation level — how a modern language model actually works, from raw text to a trained, chat-style assistant.

## Overview

Luna started as a fully from-scratch implementation (tokenizer, embeddings, forward/backward pass, and gradient descent, all in NumPy, with no deep learning framework) to learn the mechanics behind LLMs firsthand. It has since evolved into a PyTorch-based Transformer, so that the project moves from "understanding every operation by hand" to "using the tools the industry actually uses" (`autograd`, `nn.Module`, `AdamW`) while keeping the tokenizer fully custom.

## What Luna can do

- Learn a vocabulary from a raw text corpus using a **from-scratch Byte Pair Encoding (BPE) tokenizer** (no `tiktoken`, no `sentencepiece`)
- Encode/decode text to and from token IDs, including special tokens (`<USER>`, `<ASSISTANT>`, `<EOS>`, `<PAD>`, `<UNK>`)
- Build a supervised Q&A training set with **loss masking** (the question tokens are excluded from the loss with `ignore_index=-100`, so the model is only penalized on the tokens it actually needs to generate)
- Train a **multi-head self-attention Transformer** (PyTorch `nn.Module`) with causal + padding masking, Pre-LN residual connections, RMSNorm, and a SwiGLU-style feed-forward block
- Generate answers autoregressively (greedy or temperature-based sampling) from a natural-language prompt

Example:
```
Prompt: What is time complexity?
Generated: Time complexity describes how an algorithm's execution time
grows as the size of its input increases.
```

## Architecture

**Tokenizer (from scratch, no external library)**
- Regex-based pre-tokenization (words, digits, punctuation)
- Iterative BPE merging: at each step, finds the most frequent adjacent token pair across the corpus and merges it, until the target vocabulary size is reached
- Vocabulary and merge rules persisted to `tokenizer/vocab.json` and `tokenizer/merge.json`, reused for consistent encoding at both training and generation time

**Model (PyTorch)**
- Token embedding + learned positional embedding
- N stacked Transformer blocks, each with:
  - RMSNorm → multi-head self-attention (scaled dot-product, causal mask + padding mask) → residual connection
  - RMSNorm → SwiGLU feed-forward (`silu(W1x) * W2x` projected back down by `W3`) → residual connection
- Final linear `LM head` projecting the last token's hidden state to vocabulary logits

**Training**
- Custom `Dataset`/`DataLoader` pipeline built on top of the hand-written tokenizer
- Question/answer pairs formatted as `<USER> question <ASSISTANT> answer <EOS>`, with the question span masked out (`-100`) in the targets so loss is computed only on the assistant's response
- `CrossEntropyLoss` with `ignore_index=-100`, `AdamW` optimizer, checkpointing to resume training across sessions
- Trained on **2,052 Q&A pairs** covering programming and CS fundamentals (variables, data types, functions, complexity, OOP concepts, etc.), tokenized into a **5,000-token BPE vocabulary**

**Generation**
- Prompt is encoded, wrapped with `<USER>`/`<ASSISTANT>` tokens, and padded to the model's context size
- At each step, the model predicts the next token from the last real (non-padding) position, appends it, and repeats until `<EOS>` or the context limit is reached
- Supports both **greedy decoding** (`argmax`, deterministic) and **temperature-based sampling** (`multinomial`, more varied output)

## Project structure

```
Luna/
├── config.py                  # Model & training hyperparameters
├── paths.py                   # Centralized file paths
├── data/
│   ├── input_text.txt          # Raw Q&A training corpus
│   ├── dataset.jsonl           # Structured {question, answer} dataset
│   └── model.pth                # Trained model + optimizer checkpoint
├── tokenizer/
│   ├── tokenizer.py            # From-scratch BPE training (vocab + merge rules)
│   ├── encoder.py              # encode_prompt() / decode_token_ids()
│   ├── vocab.json               # Learned vocabulary (token -> id)
│   └── merge.json               # Learned BPE merge rules
├── model/
│   └── transformer.py          # Transformer block + full model (PyTorch nn.Module)
├── training/
│   ├── dataset.py               # Sequence building, masking, attention/causal masks
│   └── train.py                 # Training loop (forward, loss, backward, checkpointing)
└── generation/
    └── generation.py            # Autoregressive generation from a prompt
```

## Getting started

**Requirements:** Python 3.12+, PyTorch

```bash
pip install torch
```

**Train:**
```bash
python -m training.train
```

**Chat with Luna:**
```bash
python -m generation.generation
```

## Current limitations

- Small model (`hidden_size=128`, 6 layers, `context_size=64`) — designed to be trainable quickly on a CPU, not to compete with production LLMs
- Narrow domain: reliable mostly on questions close to the training distribution (programming/CS fundamentals); can blend or misattribute answers on questions that overlap in vocabulary across topics
- Single-turn Q&A rather than multi-turn conversation memory

## Roadmap

- Scale up (`hidden_size`, layers, context) now that the model runs on PyTorch
- Expand and diversify the training dataset to reduce topic confusion
- Move to multi-turn conversation formatting (retaining prior turns in context)
- Add TensorFlow as an alternate backend to compare implementations

## Author

Zakaria Soudaki
