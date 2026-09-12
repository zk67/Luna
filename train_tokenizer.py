import paths
import json
from tokenizer.tokenizer import BPETokenizer

def main():
    with open(paths.PROCESSED_DATASET_PATH, "r", encoding="utf-8") as file:
        processed_dataset = file.read()

    texts = ""

    for record in processed_dataset.splitlines():
        record_dict = json.loads(record)
        messages = record_dict.get("messages")
        content_user = messages[0]["content"]
        content_assistant = messages[1]["content"]
        texts += content_user + " " + content_assistant + " "

    tokenizer = BPETokenizer()
    tokenizer.train(texts)
    tokenizer.save()

if __name__ == "__main__":
    main()