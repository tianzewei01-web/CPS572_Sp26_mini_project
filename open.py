from datasets import load_dataset
import json

def prepare_smart_code_data():
    print("Loading OpenCodeInstruct (This might take a while)...")
    dataset = load_dataset("nvidia/OpenCodeInstruct", split="train")
    
    # filter for Python tasks with high test scores
    def is_excellent_python(row):
        # 1. python or not
        output_text = str(row.get('output', ''))
        is_python = "```python" in output_text.lower() or "def " in output_text
        
        # 2. high score or not
        raw_score = row.get('average_test_score')
        try:
            # convert to float, if it's None or not a number
            numeric_score = float(raw_score)
        except (ValueError, TypeError):
            numeric_score = 0.0
            
        # 3. select only those with perfect score 1.0
        is_excellent_score = numeric_score == 1.0
        
        return is_python and is_excellent_score

    print("Filtering for EXCELLENT Python tasks (Score == 1.0)...")
    perfect_dataset = dataset.filter(is_excellent_python)
    
    total_excellent = len(perfect_dataset)
    print(f"We have {total_excellent} excellent Python tasks left.")
    
    if total_excellent == 0:
        print("Warning: 0 data points. Check the filtering logic.")
        return []

    # subsample if we have more than 10k excellent examples
    sample_size = min(10000, total_excellent)
    print(f"Subsampling {sample_size} examples...")
    final_dataset = perfect_dataset.shuffle(seed=42).select(range(sample_size))
    
    # convert to Tinker format
    formatted_data = []
    for row in final_dataset:
        convo = [
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]}
        ]
        formatted_data.append(convo)
        
    print("Data preparation complete!")
    return formatted_data

if __name__ == "__main__":
    code_data = prepare_smart_code_data()
    
    if len(code_data) > 0:
        output_file = "code_train_10k.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(code_data, f, ensure_ascii=False, indent=2)
        print(f"🎉 大功告成！{len(code_data)} 条高质量代码数据已安全保存到 {output_file}!")