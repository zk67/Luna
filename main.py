import json

import paths
from tokenizer.tokenizer import split_text, tokenize_function
from training.dataset import build_dataset_jsonl, create_attention_masks, create_casual_attention_mask , create_training_examples
from tokenizer.encoder import encode_prompt

def main():
    with open(paths.VOCAB_PATH, "r", encoding="utf-8") as file:
         vocab = json.load(file)

    # tokenize_function(input_text)
    # build_dataset_jsonl(input_text)
    val = "PyTorch utilise principalement Python, Pytorch."
    inputs, targets =create_training_examples(vocab)
    # print("Inputs:", inputs)
    # print("Targets:", targets)

    attention_masks = create_attention_masks(inputs, vocab)
    print("Attention Masks:", attention_masks)

    causal_attention_mask = create_casual_attention_mask()
    print("Causal Attention Mask:", causal_attention_mask)


    

if __name__ == "__main__":
    main()