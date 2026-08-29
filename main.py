import os
import numpy as np
import json
from training.train import train, load_new_dataset, reset_model
from tokenizer.tokenizer import tokenize_function
import config
import paths
from generation.generation import run_generation

# Function to save the current configuration to a file.
def save_config():

    with open("config.py", "w", encoding="utf-8") as f:
        f.write(f"VOCAB_SIZE = {config.VOCAB_SIZE}\n")
        f.write(f"HIDDEN_SIZE = {config.HIDDEN_SIZE}\n")
        f.write(f"CONTEXT_SIZE = {config.CONTEXT_SIZE}\n")
        f.write(f"HEADS = {config.HEADS}\n")
        f.write(f"INTERMEDIATE_SIZE = {config.INTERMEDIATE_SIZE}\n")
        f.write(f"LAYERS = {config.LAYERS}\n")
        f.write(f"BATCH_SIZE = {config.BATCH_SIZE}\n")
        f.write(f"TEMPERATURE = {config.TEMPERATURE}\n")
    
#function that clears the console screen.
def clear():
    os.system("cls" if os.name == "nt" else "clear")

# ---------------------- SCREENS ----------------------

def show_main_menu():
    clear()
    print("╔═════════════════════════════════════════════════════╗")
    print("║ LUNA                                                ║") 
    print("║ Welcome to your local AI Model                      ║") 
    print("╠═════════════════════════════════════════════════════╣")
    print("║ Developer : Zakaria Soudaki                         ║")
    print("║ Version : 0.1.0                                     ║")
    print("╠═════════════════════════════════════════════════════╣")
    print("║ MODEL INFORMATION                                   ║")
    print(f"║ Hidden size : {config.HIDDEN_SIZE:<15}                       ║") 
    print(f"║ Context size : {config.CONTEXT_SIZE:<15}                      ║") 
    print(f"║ Attention heads : {config.HEADS:<15}                   ║") 
    print(f"║ Intermediate size : {config.INTERMEDIATE_SIZE:<15}                 ║") 
    print(f"║ Transformer layers : {config.LAYERS:<15}                ║") 
    print("╠═════════════════════════════════════════════════════╣") 
    print("║ TRAINING INFORMATION                                ║") 
    print(f"║ Batch size : {config.BATCH_SIZE:<15}                        ║") 
    print("╠═════════════════════════════════════════════════════╣") 
    print("║ 1. Chat with Luna                                   ║") 
    print("║ 2. Train the model                                  ║") 
    print("║ 3. Change model parameters                          ║") 
    print("║ 4. Exit                                             ║") 
    print("╚═════════════════════════════════════════════════════╝")


def show_train_menu():
    clear()
    print("╔══════════════════════════════════════════╗")
    print("║              TRAIN THE AI                ║")
    print("╠══════════════════════════════════════════╣")
    print("║                                          ║")
    print("║  Training Data                           ║")
    print("║  ──────────────────────────────────────  ║")
    print(f"║  Current file : input_text.txt           ║")
    print(f"║                                          ║")
    print("║                                          ║")
    print("║  1. Start training                       ║")
    print("║  2. Load a new text file                 ║")
    print("║  3. Reset the training                   ║")
    print("║  4. Back                                 ║")
    print("║                                          ║") 
    print("╚══════════════════════════════════════════╝")

def train_the_AI():
    while True:
        show_train_menu()
        choix = input("\nChoice : ").strip()
        if choix == "1":
            start_training()
        elif choix == "2":
            print("This will load a new text file and reinitialize the vocabulary and merge rules.")
            print("Make sure you have changed the input_text.txt file with the new text.")
            input_file = input("Are you sure you want to load a new text file? (y/n): ").strip()
            if input_file.lower() == "y":
                load_new_dataset()
                print("Vocabulary and merge rules and dataset initialized.")
            else:
                print("Operation cancelled.")
        elif choix == "3":
            print("This will reset the training data and reinitialize the embedding and output matrices.")
            input_reset = input("Are you sure you want to reset the training? (y/n): ").strip()
            if input_reset.lower() == "y":
                reset_model()
                print("\nTraining data reset and matrices initialized.")
            else:
                print("Operation cancelled.")
        elif choix == "4":
            break
        else:
            print("Invalid choice.")


#this function trains the model using the training examples created from the token ids and updates the embedding
#and output matrices using gradient descent 
#the function also prints the average loss per batch and saves the model if the average loss is less than the best loss
def start_training():
    while True:
        reponse = input("Are you sure you want to start training? (y/n): ").strip().lower()

        if reponse == "n":
            break

        while True:
            response = input("For how many loops do you want to train the model? : ").strip()
            if response.isdigit():
                break
            print("Please enter a valid number.")

        train(int(response))

        input_again = input("\nDo you want to train again? (y/n): ").strip().lower()
        if input_again != "y":
            break

def show_model_params_menu(): 
    clear() 
    print("╔═════════════════════════════════════════════════════╗") 
    print("║ MODEL PARAMETERS                                    ║") 
    print("╠═════════════════════════════════════════════════════╣") 
    print(f"║ Batch size : {config.BATCH_SIZE:<13}                          ║") 
    print(f"║ Hidden size : {config.HIDDEN_SIZE:<13}                         ║") 
    print(f"║ Context size : {config.CONTEXT_SIZE:<13}                        ║") 
    print(f"║ Attention heads : {config.HEADS:<13}                     ║") 
    print(f"║ Intermediate size : {config.INTERMEDIATE_SIZE:<13}                   ║") 
    print(f"║ Transformer layers: {config.LAYERS:<13}                   ║") 
    print("║                                                     ║") 
    print("║ 1. Change batch size                                ║") 
    print("║ 2. Change hidden size                               ║") 
    print("║ 3. Change context size                              ║") 
    print("║ 4. Change number of attention heads                 ║") 
    print("║ 5. Change the intermediate size                     ║") 
    print("║ 6. Change the transformer layers                    ║") 
    print("║ 7. Back                                             ║") 
    print("╚═════════════════════════════════════════════════════╝")

def training_parameters():
    while True:
        show_model_params_menu()
        choix = input("\nChoice : ").strip()

        # Batch size
        if choix == "1":
            new_value = input(
                f"Enter new batch size "
                f"(current: {config.BATCH_SIZE}, leave blank to keep current): "
            ).strip()

            if new_value.lower() == "exit" or new_value == "":
                break

            config.BATCH_SIZE = int(new_value)
            save_config()
            print(f"Batch size updated to {config.BATCH_SIZE}.")

        # Hidden size
        elif choix == "2":
            new_value = input(
                f"Enter hidden size "
                f"(current: {config.HIDDEN_SIZE}, leave blank to keep current): "
            ).strip()

            if new_value.lower() == "exit" or new_value == "":
                break

            config.HIDDEN_SIZE = int(new_value)
            save_config()
            print(f"Hidden size updated to {config.HIDDEN_SIZE}.")

        # Context size
        elif choix == "3":
            new_value = input(
                f"Enter context size "
                f"(current: {config.CONTEXT_SIZE}, leave blank to keep current): "
            ).strip()

            if new_value.lower() == "exit" or new_value == "":
                break

            config.CONTEXT_SIZE = int(new_value)
            save_config()
            print(f"Context size updated to {config.CONTEXT_SIZE}.")

        # Number of heads
        elif choix == "4":
            new_value = input(
                f"Enter number of attention heads "
                f"(current: {config.HEADS}, leave blank to keep current): "
            ).strip()

            if new_value.lower() == "exit" or new_value == "":
                break

            config.HEADS = int(new_value)
            save_config()
            print(f"Number of heads updated to {config.HEADS}.")

        # Intermediate size
        elif choix == "5":
            new_value = input(
                f"Enter intermediate size "
                f"(current: {config.INTERMEDIATE_SIZE}, leave blank to keep current): "
            ).strip()

            if new_value.lower() == "exit" or new_value == "":
                break

            config.INTERMEDIATE_SIZE = int(new_value)
            save_config()
            print(f"Intermediate size updated to {config.INTERMEDIATE_SIZE}.")

        # Number of layers
        elif choix == "6":
            new_value = input(
                f"Enter number of layers "
                f"(current: {config.LAYERS}, leave blank to keep current): "
            ).strip()

            if new_value.lower() == "exit" or new_value == "":
                break

            config.LAYERS = int(new_value)
            save_config()
            print(f"Number of layers updated to {config.LAYERS}.")

        # Back
        elif choix == "7":
            break

        else:
            print("Invalid choice.")

def show_chat():
    clear()
    print("╔══════════════════════════════════════════════════╗")
    print("║                  CHAT WITH LUNA                  ║")
    print("╠══════════════════════════════════════════════════╣")
    print('║  Type "exit" to return to the main menu.         ║')
    print("╚══════════════════════════════════════════════════╝\n")

def chat_with_luna():
    show_chat()
    sentence = input("So, I heared you had a question... : ")
    while True:
        if sentence.strip().lower() == "exit":
            break
        response = run_generation(sentence)
        print()
        print(f"Luna: {response}\n")
        sentence = input("Whats next? : ")


# ---------------------- NAVIGATION ----------------------
def main():
    while True:
        show_main_menu()
        choix = input("\nChoice : ").strip()
        if choix == "1":
            chat_with_luna()
        elif choix == "2":
           train_the_AI()
        elif choix == "3":
            training_parameters()
        elif choix == "4":
            print("\nSee you soon !")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()