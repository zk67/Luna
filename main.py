import os
import numpy as np
import json
from training.train import train, load_new_dataset, reset_model
from tokenizer.tokenizer import tokenize_function
import config
import paths
from generation.generation import run_generation

# ---------------------- COULEURS (ANSI) ----------------------
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    DIM = "\033[2m"


def ok(msg):
    print(f"{C.GREEN}✔ {msg}{C.RESET}")


def warn(msg):
    print(f"{C.YELLOW}⚠ {msg}{C.RESET}")


def err(msg):
    print(f"{C.RED}✘ {msg}{C.RESET}")


def info(msg):
    print(f"{C.CYAN}{msg}{C.RESET}")


# ---------------------- ROBUST INPUT HELPERS ----------------------

def ask_choice(prompt, valid_choices):
    """
    Ask for a choice among a list of valid choices (e.g. ["1", "2", "3"]).
    Case- and whitespace-insensitive. Keeps asking until the input is valid.
    """
    valid_normalized = {c.strip().lower() for c in valid_choices}
    while True:
        response = input(prompt).strip().lower()
        if response in valid_normalized:
            return response
        err(f"Invalid choice. Valid options: {', '.join(valid_choices)}")


def ask_yes_no(prompt, default=None):
    """
    Ask for a yes/no confirmation. Accepts y/n/yes/no/1/0/true/false,
    case- and whitespace-insensitive.
    If default is provided ('y' or 'n'), an empty input returns that value.
    """
    yes_values = {"y", "yes", "1", "true"}
    no_values = {"n", "no", "0", "false"}

    while True:
        response = input(prompt).strip().lower()

        if response == "" and default is not None:
            return default == "y"

        if response in yes_values:
            return True
        if response in no_values:
            return False

        err("Answer not recognized. Type 'y' (yes) or 'n' (no).")


def ask_int(prompt, current=None, allow_blank=True, min_value=None):
    """
    Ask for an integer safely (no crash if the user types garbage).
    If allow_blank=True, an empty input keeps the current value
    and returns None to signal "no change".
    """
    while True:
        raw = input(prompt).strip()

        if allow_blank and raw == "":
            return None

        if raw.lower() == "exit":
            return None

        try:
            value = int(raw)
        except ValueError:
            err("That's not a valid number. Try again (or leave blank to cancel).")
            continue

        if min_value is not None and value < min_value:
            err(f"Value must be greater than or equal to {min_value}.")
            continue

        return value


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


# function that clears the console screen.
def clear():
    os.system("cls" if os.name == "nt" else "clear")


# ---------------------- SCREENS ----------------------

def show_main_menu():
    clear()
    print(f"{C.MAGENTA}{C.BOLD}╔═════════════════════════════════════════════════════╗")
    print("║ LUNA                                                ║")
    print("║ Welcome to your local AI Model                      ║")
    print("╠═════════════════════════════════════════════════════╣")
    print(f"{C.RESET}{C.MAGENTA}║{C.RESET} Developer : Zakaria Soudaki                         {C.MAGENTA}║")
    print(f"║ Version : 0.1.0                                     ║{C.RESET}")
    print(f"{C.MAGENTA}╠═════════════════════════════════════════════════════╣{C.RESET}")
    print(f"{C.MAGENTA}║{C.RESET} {C.BOLD}MODEL INFORMATION{C.RESET}                                   {C.MAGENTA}║")
    print(f"║{C.RESET} Hidden size : {config.HIDDEN_SIZE:<15}                       {C.MAGENTA}║")
    print(f"║{C.RESET} Context size : {config.CONTEXT_SIZE:<15}                      {C.MAGENTA}║")
    print(f"║{C.RESET} Attention heads : {config.HEADS:<15}                   {C.MAGENTA}║")
    print(f"║{C.RESET} Intermediate size : {config.INTERMEDIATE_SIZE:<15}                 {C.MAGENTA}║")
    print(f"║{C.RESET} Transformer layers : {config.LAYERS:<15}                {C.MAGENTA}║")
    print(f"╠═════════════════════════════════════════════════════╣")
    print(f"║{C.RESET} {C.BOLD}TRAINING INFORMATION{C.RESET}                                {C.MAGENTA}║")
    print(f"║{C.RESET} Batch size : {config.BATCH_SIZE:<15}                        {C.MAGENTA}║")
    print(f"╠═════════════════════════════════════════════════════╣")
    print(f"║{C.RESET} {C.CYAN}1.{C.RESET} Chat with Luna                                   {C.MAGENTA}║")
    print(f"║{C.RESET} {C.CYAN}2.{C.RESET} Train the model                                  {C.MAGENTA}║")
    print(f"║{C.RESET} {C.CYAN}3.{C.RESET} Change model parameters                          {C.MAGENTA}║")
    print(f"║{C.RESET} {C.CYAN}4.{C.RESET} Exit                                             {C.MAGENTA}║")
    print(f"╚═════════════════════════════════════════════════════╝{C.RESET}")


def show_train_menu():
    clear()
    print(f"{C.CYAN}{C.BOLD}╔══════════════════════════════════════════╗")
    print("║              TRAIN THE AI                ║")
    print(f"╠══════════════════════════════════════════╣{C.RESET}")
    print("║                                          ║")
    print("║  Training Data                           ║")
    print("║  ──────────────────────────────────────  ║")
    print(f"║  Current file : input_text.txt           ║")
    print(f"║                                          ║")
    print("║                                          ║")
    print(f"║  {C.CYAN}1.{C.RESET} Start training                       ║")
    print(f"║  {C.CYAN}2.{C.RESET} Load a new text file                 ║")
    print(f"║  {C.CYAN}3.{C.RESET} Reset the training                   ║")
    print(f"║  {C.CYAN}4.{C.RESET} Back                                 ║")
    print("║                                          ║")
    print(f"{C.CYAN}╚══════════════════════════════════════════╝{C.RESET}")


def train_the_AI():
    while True:
        show_train_menu()
        choice = ask_choice("\nChoice : ", ["1", "2", "3", "4"])

        if choice == "1":
            start_training()

        elif choice == "2":
            warn("This will load a new text file and reinitialize the vocabulary and merge rules.")
            info("Make sure you have changed the input_text.txt file with the new text.")
            if ask_yes_no("Are you sure you want to load a new text file? (y/n): ", default="n"):
                load_new_dataset()
                ok("Vocabulary, merge rules and dataset initialized.")
            else:
                info("Operation cancelled.")

        elif choice == "3":
            warn("This will reset the training data and reinitialize the embedding and output matrices.")
            if ask_yes_no("Are you sure you want to reset the training? (y/n): ", default="n"):
                reset_model()
                ok("Training data reset and matrices initialized.")
            else:
                info("Operation cancelled.")

        elif choice == "4":
            break

        input("\nPress Enter to continue...")


# this function trains the model using the training examples created from the token ids and updates the embedding
# and output matrices using gradient descent
# the function also prints the average loss per batch and saves the model if the average loss is less than the best loss
def start_training():
    while True:
        if not ask_yes_no("Are you sure you want to start training? (y/n): ", default="n"):
            break

        nb_loops = ask_int("For how many loops do you want to train the model? : ", allow_blank=False, min_value=1)
        if nb_loops is None:
            # user typed "exit" or left it blank when it wasn't allowed, ask again
            continue

        train(nb_loops)
        ok("Training finished.")

        if not ask_yes_no("\nDo you want to train again? (y/n): ", default="n"):
            break


def show_model_params_menu():
    clear()
    print(f"{C.CYAN}{C.BOLD}╔═════════════════════════════════════════════════════╗")
    print("║ MODEL PARAMETERS                                    ║")
    print(f"╠═════════════════════════════════════════════════════╣{C.RESET}")
    print(f"║ Batch size : {config.BATCH_SIZE:<13}                          ║")
    print(f"║ Hidden size : {config.HIDDEN_SIZE:<13}                         ║")
    print(f"║ Context size : {config.CONTEXT_SIZE:<13}                        ║")
    print(f"║ Attention heads : {config.HEADS:<13}                     ║")
    print(f"║ Intermediate size : {config.INTERMEDIATE_SIZE:<13}                   ║")
    print(f"║ Transformer layers: {config.LAYERS:<13}                   ║")
    print("║                                                     ║")
    print(f"║ {C.CYAN}1.{C.RESET} Change batch size                                ║")
    print(f"║ {C.CYAN}2.{C.RESET} Change hidden size                               ║")
    print(f"║ {C.CYAN}3.{C.RESET} Change context size                              ║")
    print(f"║ {C.CYAN}4.{C.RESET} Change number of attention heads                 ║")
    print(f"║ {C.CYAN}5.{C.RESET} Change the intermediate size                     ║")
    print(f"║ {C.CYAN}6.{C.RESET} Change the transformer layers                    ║")
    print(f"║ {C.CYAN}7.{C.RESET} Back                                             ║")
    print(f"{C.CYAN}╚═════════════════════════════════════════════════════╝{C.RESET}")


def training_parameters():
    # (displayed label, config attribute, minimum accepted value)
    params = {
        "1": ("batch size", "BATCH_SIZE", 1),
        "2": ("hidden size", "HIDDEN_SIZE", 1),
        "3": ("context size", "CONTEXT_SIZE", 1),
        "4": ("number of attention heads", "HEADS", 1),
        "5": ("intermediate size", "INTERMEDIATE_SIZE", 1),
        "6": ("number of layers", "LAYERS", 1),
    }

    while True:
        show_model_params_menu()
        choice = ask_choice("\nChoice : ", list(params.keys()) + ["7"])

        if choice == "7":
            break

        label, attr, min_value = params[choice]
        current_value = getattr(config, attr)

        new_value = ask_int(
            f"Enter new {label} (current: {current_value}, leave blank to keep current): ",
            current=current_value,
            allow_blank=True,
            min_value=min_value,
        )

        if new_value is None:
            info("No change made.")
        else:
            setattr(config, attr, new_value)
            save_config()
            ok(f"{label.capitalize()} updated to {new_value}.")

        input("\nPress Enter to continue...")


def show_chat():
    clear()
    print(f"{C.GREEN}{C.BOLD}╔══════════════════════════════════════════════════╗")
    print("║                  CHAT WITH LUNA                  ║")
    print(f"╠══════════════════════════════════════════════════╣{C.RESET}")
    print('║  Type "exit" to return to the main menu.         ║')
    print(f"{C.GREEN}╚══════════════════════════════════════════════════╝{C.RESET}\n")


def chat_with_luna():
    show_chat()
    while True:
        sentence = input("So, I heard you had a question... : ").strip()

        if sentence.lower() == "exit":
            break

        if sentence == "":
            warn("Type something before sending!")
            continue

        response = run_generation(sentence)
        print()
        print(f"{C.CYAN}Luna:{C.RESET} {response}\n")


# ---------------------- NAVIGATION ----------------------
def main():
    while True:
        show_main_menu()
        choice = ask_choice("\nChoice : ", ["1", "2", "3", "4"])

        if choice == "1":
            chat_with_luna()
        elif choice == "2":
            train_the_AI()
        elif choice == "3":
            training_parameters()
        elif choice == "4":
            print(f"\n{C.MAGENTA}See you soon !{C.RESET}")
            break


if __name__ == "__main__":
    main()