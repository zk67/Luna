import torch
import torch.nn as nn
import json
from model.transformer import Transformer
from torch.utils.data import TensorDataset, DataLoader
from training.dataset import build_dataset_jsonl, create_attention_masks, create_training_examples, create_casual_mask 
from tokenizer.tokenizer import tokenize_function


def main():
    # with open("data/input_text.txt", "r", encoding="utf-8") as file:
    #     input_text = file.read()

    # print("Building vocab and and merge rules from input text...")
    # tokenize_function(input_text)  # Tokenize the input text and update vocab and merge rules
    # print("done building vocab and merge rules")
    # print("Building dataset from input text...")
    # build_dataset_jsonl(input_text)  # Build the dataset from the input text
    # print("done building dataset")
    # print("starting training...")
    

    with open("tokenizer/vocab.json", "r", encoding="utf-8") as file:
            vocab = json.load(file)

    print("creating training examples...")
    inputs, targets = create_training_examples(vocab)

    print("creating attention masks...")
    attention_masks = create_attention_masks(inputs, vocab)
    casual_attention_mask = create_casual_mask()

    dataset = TensorDataset( inputs, targets, attention_masks )

    # Create batches of 8 examples
    dataloader = DataLoader( dataset, batch_size=8, shuffle=True )


    model = Transformer(
        vocab_size=len(vocab),
        hidden_size=128,
        context_size=64,
        num_heads=4,
        intermediate_size=512,
        num_layers=6
    )

    #loss function that ignores the padding tokens in the target_ids during loss calculation.
    criterion = nn.CrossEntropyLoss( ignore_index=-100 )

    #optimizer that uses the AdamW optimization algorithm with a learning rate of 0.001 to update the model's parameters during training.
    optimizer = torch.optim.AdamW( model.parameters(), lr=0.0003 )

    #load the model and optimizer state from a checkpoint file "data/checkpoint.pth" to resume training from a previous state.
    checkpoint = torch.load("data/model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    num_loops = 10

    print("Starting training for", num_loops, "epochs...")
    model.train() 
    for loop in range(num_loops):
        total_loss = 0

        for input_ids, target_ids, attention_mask in dataloader:   

            #reset the gradients of the model's parameters to zero before each training step to prevent accumulation of gradients from previous steps.
            optimizer.zero_grad()

            #forward pass through the model to get the predicted logits for the input sequences, using the attention masks and casual attention mask to control which tokens the model attends to during training.
            logits = model( input_ids, attention_mask, casual_attention_mask )

            # Reshape logits and targets for CrossEntropyLoss 
            logits = logits.reshape(-1, len(vocab))
            target_ids = target_ids.reshape(-1)

            # Calculate loss
            loss = criterion( logits, target_ids )

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print( f"Epoch {loop + 1}/{num_loops} - " f"Loss: {average_loss:.4f}" )

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, "data/model.pth")
    

if __name__ == "__main__":
    main()