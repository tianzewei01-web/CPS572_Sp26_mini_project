from datasets import load_dataset

def check_dataset_structure():
    print("Fetching dataset info (streaming mode to save time)...")
    # streaming=True，just see the beginning
    dataset = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
    
    # extract the first row of data
    first_row = next(iter(dataset))
    
    print("\n" + "="*30)
    print("🔑 Dataset Keys:")
    print(first_row.keys())
    print("="*30 + "\n")
    
    print("First Row Sample:")
    for key, value in first_row.items():
        content_preview = str(value)[:100].replace("\n", " ") 
        print(f"👉 [{key}]: {content_preview}...")

if __name__ == "__main__":
    check_dataset_structure()