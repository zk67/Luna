# import os
# import numpy as np
# import json
# from training.model import initialize_matrix
# from tokenizer.tokenizer import tokenize_function
# import config
# import paths
# from training.train_model import train_model
# from generation.generation import generate

# # Function to save the current configuration to a file.
# def save_config():
#     with open("config.py", "w", encoding="utf-8") as f:
#         f.write(f"VOCAB_SIZE = {config.VOCAB_SIZE}\n")
#         f.write(f"EMBEDDING_SIZE = {config.EMBEDDING_SIZE}\n")
#         f.write(f"CONTEXT_SIZE = {config.CONTEXT_SIZE}\n")
#         f.write(f"LEARNING_RATE = {config.LEARNING_RATE}\n")
#         f.write(f"BATCH_SIZE = {config.BATCH_SIZE}\n")
#         f.write(f"TRAINING_LOOPS = {config.TRAINING_LOOPS}\n")
#         f.write(f"BEST_LOSS = {config.BEST_LOSS}\n")

# #this function loads the corpus, the vocabulary and the merge rules, initializes the embedding and output matrices,
# #and saves them in "model/matrices/embedding_matrix.npy" and "model/matrices/output_matrix.npy" respectively.
# def new_corpus():
#     with open(paths.VOCAB_PATH, "w", encoding="utf-8") as file:
#         json.dump({}, file)

#     with open(paths.MERGE_PATH, "w", encoding="utf-8") as file:
#         json.dump([], file)

#     vocab = tokenize_function()
#     config.VOCAB_SIZE = len(vocab)
#     save_config()

#     embedding_matrix = initialize_matrix(
#     config.VOCAB_SIZE,
#     columns=config.EMBEDDING_SIZE
#     )

#     output_matrix = initialize_matrix(
#         lines=config.EMBEDDING_SIZE * config.CONTEXT_SIZE,
#         columns=config.VOCAB_SIZE
#     )

#     np.save(paths.EMBEDDING_MATRIX_PATH, embedding_matrix)
#     np.save(paths.OUTPUT_MATRIX_PATH, output_matrix)
    
# # ==================================================================
# #function that clears the console screen.
# def clear():
#     os.system("cls" if os.name == "nt" else "clear")

# # ---------------------- SCREENS ----------------------

# def show_main_menu():
#     clear()
#     print("╔══════════════════════════════════════════╗")
#     print("║                  LUNA AI                 ║")
#     print("║        Welcome to your local AI          ║")
#     print("╠══════════════════════════════════════════╣")
#     print("║  Developer : Zakaria Soudaki             ║")
#     print("║  Version   : 0.1.0                       ║")
#     print("║                                          ║")
#     print("║  Model Information                       ║")
#     print(f"║  Vocabulary size : {config.VOCAB_SIZE}                   ║")
#     print(f"║  Embedding size  : {config.EMBEDDING_SIZE}                     ║")
#     print(f"║  Context size    : {config.CONTEXT_SIZE}                     ║")
#     print("╠══════════════════════════════════════════╣")
#     print("║                                          ║")
#     print("║  1. Chat with Luna                       ║")
#     print("║  2. Train the AI or change parameters    ║")
#     print("║  3. Exit                                 ║")
#     print("║                                          ║")
#     print("╚══════════════════════════════════════════╝")


# def show_train_menu():
#     clear()
#     print("╔══════════════════════════════════════════╗")
#     print("║              TRAIN THE AI                ║")
#     print("╠══════════════════════════════════════════╣")
#     print("║                                          ║")
#     print("║  Training Data                           ║")
#     print("║  ──────────────────────────────────────  ║")
#     print(f"║  Current file : corpus.txt               ║")
#     print(f"║                                          ║")
#     print("║                                          ║")
#     print("║  1. Start training                       ║")
#     print("║  2. Load a new text file                 ║")
#     print("║  3. Change default Training parameters   ║")
#     print("║  4. Reset the training (matrices)        ║")
#     print("║  5. Back                                 ║")
#     print("║                                          ║") 
#     print("╚══════════════════════════════════════════╝")

# def train_the_AI():
#     while True:
#         show_train_menu()
#         choix = input("\nChoix : ").strip()
#         if choix == "1":
#             start_training()
#         elif choix == "2":
#             print("This will load a new text file and reinitialize the vocabulary and merge rules as well as the embedding and output matrices.")
#             print("Make sure you have changed the corpus.txt file with the new text.")
#             input_file = input("Are you sure you want to load a new text file? (y/n): ").strip()
#             if input_file.lower() == "y":
#                 new_corpus()
#                 print("\nNew corpus loaded and matrices initialized.")
#                 print("Vocabulary and merge rules initialized.")
#             else:
#                 print("Operation cancelled.")
#         elif choix == "3":
#             training_parameters()
#         elif choix == "4":
#             print("This will reset the training data and reinitialize the embedding and output matrices.")
#             input_reset = input("Are you sure you want to reset the training? (y/n): ").strip()
#             if input_reset.lower() == "y":
#                 reset_matrices()
#                 print("\nTraining data reset and matrices initialized.")
#             else:
#                 print("Operation cancelled.")
#         elif choix == "5":
#             break
#         else:
#             print("Choix invalide.")

# def reset_matrices():
#     embedding_matrix = initialize_matrix(
#             config.VOCAB_SIZE,
#             columns=config.EMBEDDING_SIZE
#     )
        
#     output_matrix = initialize_matrix(
#         lines=config.EMBEDDING_SIZE * config.CONTEXT_SIZE,
#         columns=config.VOCAB_SIZE
#     )

#     np.save(paths.EMBEDDING_MATRIX_PATH, embedding_matrix)
#     np.save(paths.OUTPUT_MATRIX_PATH, output_matrix)


# #this function trains the model using the training examples created from the token ids and updates the embedding
# #and output matrices using gradient descent 
# #the function also prints the average loss per batch and saves the model if the average loss is less than the best loss
# def start_training():
#     while True:
#         print("\nEnter the training parameters (leave blank to keep current values):")
#         input_learning_rate = input(f"Learning rate (current: {config.LEARNING_RATE}): ").strip()
#         if input_learning_rate.lower() == "exit":
#             break
#         input_learning_rate = float(input_learning_rate) if input_learning_rate else config.LEARNING_RATE

#         input_batch_size = input(f"Batch size (current: {config.BATCH_SIZE}): ").strip()
#         if input_batch_size.lower() == "exit":
#             break
#         input_batch_size = int(input_batch_size) if input_batch_size else config.BATCH_SIZE

#         input_training_loops = input(f"Training loops (current: {config.TRAINING_LOOPS}): ").strip()
#         if input_training_loops.lower() == "exit":
#             break
#         input_training_loops = int(input_training_loops) if input_training_loops else config.TRAINING_LOOPS

#         input_best_loss = input(f"Best loss (current: {config.BEST_LOSS}): ").strip()
#         if input_best_loss.lower() == "exit":
#             break
#         input_best_loss = float(input_best_loss) if input_best_loss else config.BEST_LOSS

#         train_model(loop=input_training_loops, learning_rate=input_learning_rate, batch_size=input_batch_size, best_loss=input_best_loss)

#         input_again = input("\nDo you want to train again? (y/n): ").strip().lower()
#         if input_again != "y":
#             break


# def show_training_params_menu():
#     clear()
#     print("╔══════════════════════════════════════════╗")
#     print("║          TRAINING PARAMETERS             ║")
#     print("╠══════════════════════════════════════════╣")
#     print("║                                          ║")
#     print(f"║  Learning rate : {config.LEARNING_RATE}                    ║")
#     print(f"║  Batch size    : {config.BATCH_SIZE}                       ║")
#     print(f"║  Training loops: {config.TRAINING_LOOPS}                     ║")
#     print("║                                          ║")
#     print("║  1. Change learning rate                 ║")
#     print("║  2. Change batch size                    ║")
#     print("║  3. Change training loops                ║")
#     print("║  4. Back                                 ║")
#     print("║                                          ║")
#     print("╚══════════════════════════════════════════╝")

# def training_parameters():
#     while True:
#         show_training_params_menu()
#         choix = input("\nChoix : ").strip()

#         if choix == "1":
#             new_learning_rate = input(f"Enter new learning rate: (current: {config.LEARNING_RATE}, leave blank to keep current) ").strip()
#             if new_learning_rate.lower() == "exit" or new_learning_rate == "":
#                 break
#             config.LEARNING_RATE = new_learning_rate
#             save_config()
#             print(f"Learning rate updated to {config.LEARNING_RATE}.")

#         elif choix == "2":
#             new_batch_size = input(f"Enter new batch size: (current: {config.BATCH_SIZE}, leave blank to keep current) ").strip()
#             if new_batch_size.lower() == "exit" or new_batch_size == "":
#                 break
#             config.BATCH_SIZE = int(new_batch_size)
#             save_config()
#             print(f"Batch size updated to {config.BATCH_SIZE}.")

#         elif choix == "3":
#             new_training_loops = input(f"Enter new training loops: (current: {config.TRAINING_LOOPS}, leave blank to keep current) ").strip()
#             if new_training_loops.lower() == "exit" or new_training_loops == "":
#                 break
#             config.TRAINING_LOOPS = int(new_training_loops)
#             save_config()
#             print(f"Training loops updated to {config.TRAINING_LOOPS}.")

#         elif choix == "4":
#             break
#         else:
#             print("Choix invalide.")


# def show_chat():
#     clear()
#     print("╔══════════════════════════════════════════════════╗")
#     print("║                  CHAT WITH LUNA                  ║")
#     print("╠══════════════════════════════════════════════════╣")
#     print("║                                                  ║")
#     print("║  Luna: Hi! Ask me about a country or a capital.  ║")
#     print("║                                                  ║")
#     print("╠══════════════════════════════════════════════════╣")
#     print('║  Type "exit" to return to the main menu.          ║')
#     print("╚══════════════════════════════════════════════════╝\n")

# def generate_response(prompt):
#     response = generate(prompt)
#     response_text = ""
#     for word in response:
#         response_text += word
#     return response_text.strip()

# def chat_with_luna():
#     show_chat()
#     while True:
#         sentence = input("Question on a country or a capital ?: ")
#         if sentence.strip().lower() == "exit":
#             break
#         response = generate_response(sentence)

#         print(f"Luna: {response}\n")


# # ---------------------- NAVIGATION ----------------------
# def main():
#     while True:
#         show_main_menu()
#         choix = input("\nChoix : ").strip()
#         if choix == "1":
#             chat_with_luna()
#         elif choix == "2":
#            train_the_AI()
#         elif choix == "3":
#             print("\nÀ bientôt !")
#             break
#         else:
#             print("Choix invalide.")


# if __name__ == "__main__":
#     main()