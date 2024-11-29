import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

def convert_csv_to_json():
    # Read the CSV file
    df = pd.read_csv("medical_qa_train.csv")
    
    # Convert DataFrame to list of dictionaries
    qa_pairs = []
    for _, row in df.iterrows():
        qa_pair = {
            "question": row["question"],
            "answer": row["answer"]
        }
        qa_pairs.append(qa_pair)
    
    # Split into train and evaluation sets
    train_data, eval_data = train_test_split(
        qa_pairs, 
        test_size=0.1,  # 10% for evaluation
        random_state=42  # For reproducibility
    )
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save train data
    with open("data/medical_qa_train.json", "w", encoding="utf-8") as f:
        json.dump({"data": train_data}, f, indent=2, ensure_ascii=False)
    
    # Save evaluation data
    with open("data/medical_qa_eval.json", "w", encoding="utf-8") as f:
        json.dump({"data": eval_data}, f, indent=2, ensure_ascii=False)
    
    print(f"Created training dataset with {len(train_data)} examples")
    print(f"Created evaluation dataset with {len(eval_data)} examples")

if __name__ == "__main__":
    convert_csv_to_json() 