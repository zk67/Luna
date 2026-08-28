import numpy as np
from sympy import sequence
import torch
import config
import paths
import json
from tokenizer.encoder import encode_prompt

#function to create training examples from token ids, uses the first window of the context size, 
#and then moves the window by one token to create the next training example.
def create_training_examples(vocab):
    inputs = []
    targets = []

    questions, answers = load_questions_and_answers()
    for i in range(len(questions)):
        q = questions[i]
        a = answers[i]
        q_token_ids = encode_prompt(q)
        a_token_ids = encode_prompt(a)

        # Build the sequence with special tokens
        sequence = build_training_sequence(q_token_ids, a_token_ids, vocab)

        input_ids = sequence[:-1]
        target_ids = sequence[1:]   

        question_length = len(q_token_ids) + 1
        #mask the tokens_ids of the question in the target_ids with -100 to ignore them during loss calculation.
        target_ids[:question_length] = [-100] * question_length     

        if len(input_ids) > config.CONTEXT_SIZE:
            continue

        # Padding
        input_ids.extend([vocab["<PAD>"]] * (config.CONTEXT_SIZE - len(input_ids)))
        target_ids.extend([-100] * (config.CONTEXT_SIZE - len(target_ids)))

        inputs.append(input_ids)
        targets.append(target_ids)

    return (torch.tensor(inputs, dtype=torch.long),torch.tensor(targets, dtype=torch.long))

#function that creates attention masks for the input sequences, where True indicates a token to 
#attend to and False indicates a token to ignore (padding).
def create_attention_masks(inputs, vocab):
    attention_masks = []

    for input in inputs:
        mask = []
        for token_id in input:
            if token_id.item() == vocab["<PAD>"]:
                mask.append(0)
            else:
                mask.append(1)

        attention_masks.append(mask)

    return torch.tensor(attention_masks, dtype=torch.bool)

#casual_mask = matrice with True values in the lower triangle and False values in the upper triangle, 
#to prevent the model from attending to future tokens during training.
def create_casual_mask():
    return torch.tril(torch.ones(config.CONTEXT_SIZE, config.CONTEXT_SIZE, dtype=torch.bool))

#add to the question list the special token <USER> at the beginning and to the answer list the special token <ASSISTANT> at the end.
#Add to the answer list the special token <EOS> at the end.
#return the modified question and answer lists.
def build_training_sequence(question_tokens, answer_tokens, vocab): 
    sequence = ([vocab["<USER>"]] + question_tokens + [vocab["<ASSISTANT>"]] + answer_tokens + [vocab["<EOS>"]])
    return sequence

#the function reads the input text from "data/input_text.txt", splits it into questions and answers,builds the dataset "data/dataset.jsonl".
def build_dataset_jsonl(input_text):
  lines = input_text.splitlines()

  if len(lines) % 2 != 0:
    raise ValueError("Le corpus doit contenir un nombre pair de lignes.")
  
  with open(paths.DATASET_PATH, "w", encoding="utf-8") as file:
    for i in range(0, len(lines), 2):  
      question = lines[i]
      answer = lines[i + 1]
      json.dump({"question": question, "answer": answer}, file, ensure_ascii=False)
      file.write("\n")

# Load questions and answers from the dataset file.
def load_questions_and_answers():
    questions = []
    answers = []
    with open(paths.DATASET_PATH, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            
            questions.append(data["question"])
            answers.append(data["answer"])
    return questions, answers

#function that creates a generation sequence from the question tokens,
#adds the special token <USER> at the beginning and the special token <ASSISTANT> at the end.
#this is to inject the question into the model and generate an answer.
def build_generation_sequence(question_tokens, vocab):
    user_token = torch.tensor(
        [vocab["<USER>"]],
        dtype=torch.long,
        device=question_tokens.device
    )

    assistant_token = torch.tensor(
        [vocab["<ASSISTANT>"]],
        dtype=torch.long,
        device=question_tokens.device
    )

    sequence = torch.cat(
        [user_token, question_tokens, assistant_token]
    )

    return sequence