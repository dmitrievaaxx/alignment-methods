# prepare_gsm8k.py
from datasets import load_dataset
import pandas as pd
import os

def convert_gsm8k_to_parquet(dataset_split, path):
    rows = []
    for ex in dataset_split:
        prompt = "Question: " + ex["question"] + "\nAnswer:"
        target = ex["answer"]
        rows.append({"input": prompt, "output": target})
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} examples to {path}")

def main():
    os.makedirs("data", exist_ok=True)

    # dataset = load_dataset("openai/gsm8k")
    dataset = load_dataset("openai/gsm8k", "main")
    print(dataset['train'][0])


    # train
    convert_gsm8k_to_parquet(dataset["train"], "data/train.parquet")
    # validation
    convert_gsm8k_to_parquet(dataset["test"], "data/val.parquet")

if __name__ == "__main__":
    main()