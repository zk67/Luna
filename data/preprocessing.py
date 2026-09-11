import pathlib
import json

USER_POSSIBLE_KEYS = [
    "question",
    "instruction",
    "prompt",
    "input"
]

ASSISTANT_POSSIBLE_KEYS = [
    "answer",
    "output",
    "response"
]

#Convert the jsonl dataset to a list of dictionaries,
#where each dictionary represents a line in the jsonl file.
def read_jsonl():
    with open(pathlib.RAW_DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = [json.loads(line) for line in file]
    return dataset

#Clean the text by stripping whitespace and checking if it's a valid string.
def clean_text(text):
    if ( not isinstance(text, str)):
        return None
    else:
        text = text.strip()
        if (text == ""):
            return None     
    return text

#Convert a record from the dataset into a list of messages with roles.
#each record is a 2 objects dictionary with a user message and an assistant message.
def instruction_record_to_messages(record):
    message_list = []
    for key, value in record.items():
        cleaned_value = clean_text(value)
        if cleaned_value is None:
            continue  # Skip empty or invalid values

        if key in USER_POSSIBLE_KEYS:
            message_list.append({
                "role": "user",
                "content": cleaned_value
            })

        if key in ASSISTANT_POSSIBLE_KEYS:
            message_list.append({
                "role": "assistant",
                "content": cleaned_value
            })
    return message_list

#Normalize a record by checking if it has exactly one user message and one assistant message.
#If it does, return a dictionary with the messages, otherwise return None.
def normalize_record(record):
    messages = instruction_record_to_messages(record)
    if len(messages) == 2:
        role1 = messages[0]["role"] 
        role2 = messages[1]["role"]
        if role1 == "user" and role2 == "assistant":
            message_dict = {"messages": messages}
            return message_dict
    return None 

#Preprocess the dataset by reading the jsonl file, normalizing each record,
#and returning a list of valid message dictionaries. Each dictionary contains
#a list of rwo messages with roles and content.
def preprocess_dataset():
    dataset = read_jsonl()
    processed_dataset = []
    for line in dataset:
        message_dict = normalize_record(line)
        if message_dict is not None:
            processed_dataset.append(message_dict)
    return processed_dataset

#Iterate over the preprocessed dataset and yield the content of each message.
def iter_training_texts():
    processed_dataset = preprocess_dataset()
    for record in processed_dataset:
        for message in record["messages"]:
            yield message["content"]

