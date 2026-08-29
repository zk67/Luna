import torch
import config
from model.transformer import Transformer
from tokenizer import tokenizer
from training.dataset import build_generation_sequence, create_attention_masks, create_casual_mask
from tokenizer.encoder import encode_prompt, decode_token_ids
import json
import paths

#function that generates tokens using the trained model based on the input token ids, vocabulary,
#and specified parameters such as maximum number of new tokens to generate and temperature for controlling randomness in token selection.
def generate_tokens(model, input_ids, vocab, max_new_tokens=64, temperature=0.8):
    model.eval()
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_ids = build_generation_sequence(input_ids, vocab)
    input_ids = input_ids[ -config.CONTEXT_SIZE:]
    
    for i in range(max_new_tokens):
        input_ids_copy = input_ids.clone()
        current_length = input_ids_copy.shape[0]
        padding_length = config.CONTEXT_SIZE - current_length

        if padding_length > 0:
            padding = torch.full((padding_length,), vocab['<PAD>'], dtype=torch.long)
            input_ids_copy = torch.cat([input_ids_copy, padding], dim=0)

        attention_mask = create_attention_masks(input_ids_copy.unsqueeze(0), vocab).squeeze(0)
        causal_mask = create_casual_mask()

        logits = model(input_ids_copy.unsqueeze(0), attention_mask.unsqueeze(0), causal_mask)

        # FIX : prendre la position du dernier token réel, pas -1
        next_token_logits = logits[:, current_length - 1, :]

        next_token_logits = next_token_logits / temperature
        probabilities = torch.softmax(next_token_logits, dim=-1)

        #multinomial
        #next_token = torch.multinomial(probabilities, num_samples=1)
        #greedy
        next_token = torch.argmax(probabilities, dim=-1, keepdim=True)

        if next_token.item() == vocab['<EOS>']:
            break
        if current_length >= config.CONTEXT_SIZE:
            break

        input_ids = torch.cat([input_ids, next_token.squeeze(0)], dim=0)

    return input_ids


#function that runs the generation process by prompting the user for input, encoding the prompt,
#decodes the generated token ids back into text for display.
def run_generation(prompt):

    with open("tokenizer/vocab.json", "r", encoding="utf-8") as file:
        vocab = json.load(file)

    # Créer le modèle
    model = Transformer(
        vocab_size=len(vocab),
        hidden_size=config.HIDDEN_SIZE,
        context_size=config.CONTEXT_SIZE,
        num_heads=config.HEADS,
        intermediate_size=config.INTERMEDIATE_SIZE,
        num_layers=config.LAYERS,
        dropout=config.DROPOUT
    )

    # Charger les poids entraînés
    checkpoint = torch.load(paths.MODEL_PATH)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    # Tokenizer
    input_ids = encode_prompt(prompt)

    # Génération
    generated = generate_tokens(
        model,
        input_ids,
        vocab,
        max_new_tokens=config.CONTEXT_SIZE,
        temperature=config.TEMPERATURE
    )

    # Décoder
    output = decode_token_ids(generated.tolist())

    return output
