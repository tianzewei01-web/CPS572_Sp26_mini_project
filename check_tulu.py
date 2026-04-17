from datasets import load_dataset

def check_tulu_structure():
    print("Fetching Tulu-3 dataset info (streaming mode to save time)...")
    # only load the train split!!!!!!!
    dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    
    # extract the first row
    first_row = next(iter(dataset))
    
    print("\n" + "="*30)
    print("Dataset Keys:")
    print(first_row.keys())
    print("="*30 + "\n")
    
    print("First Row Sample:")
    for key, value in first_row.items():
        content_preview = str(value)[:150].replace("\n", " ") 
        print(f"👉 [{key}]: {content_preview}...")

if __name__ == "__main__":
    check_tulu_structure()