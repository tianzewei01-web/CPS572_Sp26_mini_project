import json
import random

def merge_datasets():
    print("Loading datasets...")
    
    # 1. read the three datasets
    with open("gsm8k_train.json", "r", encoding="utf-8") as f:
        math_data = json.load(f)
    with open("code_train_10k.json", "r", encoding="utf-8") as f:
        code_data = json.load(f)
    with open("tulu_train_50k.json", "r", encoding="utf-8") as f:
        tulu_data = json.load(f)
        
    print(f"Math: {len(math_data)} | Code: {len(code_data)} | Tulu(IF): {len(tulu_data)}")
    
    # 2. concatenate all datasets together  
    all_data = math_data + code_data + tulu_data
    print(f"Total combined size: {len(all_data)} rows")
    
    # 3. shuffle the combined dataset to ensure a good mix of tasks in each batch
    print("Shuffling the data to prevent catastrophic forgetting...")
    random.seed(42) # fix the random seed
    random.shuffle(all_data)
    
    # 4. save the final merged dataset to a new JSON file
    output_file = "final_training_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
        
    print(f"Final merged dataset saved to {output_file}!")

if __name__ == "__main__":
    merge_datasets()