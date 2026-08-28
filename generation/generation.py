import torch

from model.transformer import Transformer
from tokenizer import tokenizer
from training.dataset import build_generation_sequence, create_attention_masks, create_casual_mask
from tokenizer.encoder import encode_prompt, decode_token_ids
import json


# =========================
# Configuration
# =========================

VOCAB_SIZE = 5000
HIDDEN_SIZE = 128
CONTEXT_SIZE = 64
NUM_HEADS = 4
INTERMEDIATE_SIZE = 512
NUM_LAYERS = 6

MODEL_PATH = "data/model.pth"

# À adapter à ton tokenizer
EOS_TOKEN_ID = 3


# =========================
# Generation
# =========================

def generate(model, input_ids, vocab, max_new_tokens=64, temperature=0.8):
    model.eval()
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_ids = build_generation_sequence(input_ids, vocab)
    input_ids = input_ids[ -CONTEXT_SIZE:]
    


    for i in range(max_new_tokens):
        input_ids_copy = input_ids.clone()
        current_length = input_ids_copy.shape[0]
        padding_length = CONTEXT_SIZE - current_length

        if padding_length > 0:
            padding = torch.full((padding_length,), vocab["<PAD>"], dtype=torch.long)
            input_ids_copy = torch.cat([input_ids_copy, padding], dim=0)

        attention_mask = create_attention_masks(input_ids_copy.unsqueeze(0), vocab).squeeze(0)
        causal_mask = create_casual_mask()

        logits = model(input_ids_copy.unsqueeze(0), attention_mask.unsqueeze(0), causal_mask)

        # FIX : prendre la position du dernier token réel, pas -1
        next_token_logits = logits[:, current_length - 1, :]

        next_token_logits = next_token_logits / temperature
        probabilities = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        if next_token.item() == vocab["<EOS>"]:
            break
        if current_length >= CONTEXT_SIZE:
            break

        input_ids = torch.cat([input_ids, next_token.squeeze(0)], dim=0)

    return input_ids


# =========================
# Main
# =========================

def main():

    with open("tokenizer/vocab.json", "r", encoding="utf-8") as file:
        vocab = json.load(file)

    # Créer le modèle
    model = Transformer(
        vocab_size=len(vocab),
        hidden_size=HIDDEN_SIZE,
        context_size=CONTEXT_SIZE,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        num_layers=NUM_LAYERS
    )

    # Charger les poids entraînés
    checkpoint = torch.load(MODEL_PATH)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    # Prompt
    prompt = input("Prompt: ")

    # Tokenizer
    input_ids = encode_prompt(prompt)

    # Génération
    generated = generate(
        model,
        input_ids,
        vocab,
        max_new_tokens=64,
        temperature=0.8
    )

    # Décoder
    output = decode_token_ids(generated.tolist())

    print()
    print("Generated:")
    print(output)


if __name__ == "__main__":
    main()