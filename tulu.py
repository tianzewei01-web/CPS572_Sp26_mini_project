from datasets import load_dataset
import json

def prepare_tulu_data():
    print("Loading Tulu-3-SFT-Mixture (~939k rows, will take a minute or two)...")
    # only load the train split!!!!!!!
    dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    
    print("Subsampling 50,000 examples for balance...")
    # shuffle and select the first 50,000 rows to maintain a good balance 
    sampled_dataset = dataset.shuffle(seed=42).select(range(50000))
    
    formatted_data = []
    for row in sampled_dataset:
        formatted_data.append(row["messages"])
        
    print(f"Data preparation complete! Extracted {len(formatted_data)} rows.")
    return formatted_data

if __name__ == "__main__":
    tulu_data = prepare_tulu_data()
    
    output_file = "tulu_train_50k.json"
    print(f"\nSaving to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tulu_data, f, ensure_ascii=False, indent=2)
        
    print(f"50,000 instruction data points saved to {output_file}!")