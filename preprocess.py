from data.preprocessing import preprocess_dataset
import json
import paths

def main():
    print("Preprocessing data...")
    # Add your preprocessing code here

    processed_dataset = preprocess_dataset()
    print(f"Valid records: {len(processed_dataset)}")   

    # Save the processed dataset to a JSONL file
    with open(paths.PROCESSED_DATASET_PATH, "w", encoding="utf-8") as file:
        for record in processed_dataset:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")

    print("Preprocessing complete.")
    print(f"Dataset saved to: {paths.PROCESSED_DATASET_PATH}")
