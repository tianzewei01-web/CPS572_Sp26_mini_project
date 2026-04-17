from datasets import load_dataset
import random

def prepare_gsm8k():
    print("Downloading GSM8K dataset...")
    # only load the train split!!!!!!!
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    formatted_data = []
    for row in dataset:
        # match question and answer to Tinker's user and assistant
        convo = [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]}
        ]
        formatted_data.append(convo)
    
    print(f"Successfully formatted {len(formatted_data)} GSM8K samples.")
    return formatted_data



import json

if __name__ == "__main__":
    gsm8k_data = prepare_gsm8k()

    print("\nSample 0:")
    print(gsm8k_data[0])
    
    # save the formatted data to a JSON file
    output_file = "gsm8k_train.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(gsm8k_data, f, ensure_ascii=False, indent=2)
        
    print(f"\nSuccess! Data has been safely saved to {output_file}")