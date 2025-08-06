import argparse
import math
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Compute perplexity on a specified dataset split.")
    parser.add_argument("--model_path", type=str, required=True, help="Hugging Face model path or ID.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path or name of the HF dataset.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to use.")
    parser.add_argument("--text_column", type=str, default="output", help="Column name containing text.")
    parser.add_argument("--hierarchy_columns", nargs=4, default=["skill", "subskill", "goal", "indicator"], help="List of 4 columns representing hierarchy A > B > C > D.")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--output_csv", type=str, default="perplexity_results.csv", help="Output CSV file.")
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Load and tokenize dataset
    dataset = load_dataset(args.dataset_path, split=args.split)

    def tokenize(example):
        tokens = tokenizer(example[args.text_column], return_tensors="pt", truncation=True, max_length=args.max_length)
        tokens["input_ids"] = tokens["input_ids"][0]
        tokens["attention_mask"] = tokens["attention_mask"][0]
        return tokens

    dataset = dataset.map(tokenize)

    results = []

    for example in tqdm(dataset, desc=f"Computing perplexity on '{args.split}'"):
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(model.device)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss.item()
            perplexity = math.exp(loss)

        # Extract hierarchy columns and add perplexity
        row = [example[col] for col in args.hierarchy_columns]
        row.append(perplexity)
        results.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(results, columns=args.hierarchy_columns + ["perplexity"])
    
    print(df)

    # Sort by hierarchy
    df = df.sort_values(by=args.hierarchy_columns).reset_index(drop=True)

    id_avg = df.groupby(['skill', 'goal', 'subskill', 'indicator'])['perplexity'].mean().reset_index()

    rows = []

    # Group by skill → subskill → goal
    for skill, skill_df in id_avg.groupby('skill'):
        skill_values = []
        for subskill, subskill_df in skill_df.groupby('subskill'):
            subskill_values = []
            for goal, goal_df in subskill_df.groupby('goal'):
                goal_values = []
                for _, row in goal_df.iterrows():
                    rows.append({
                        "level": "indicator",
                        "skill": skill,
                        "subskill": subskill,
                        "goal": goal,
                        "indicator": row["indicator"],
                        "perplexity": row["perplexity"]
                    })
                    goal_values.append(row["perplexity"])
                rows.append({
                    "level": "goal_avg",
                    "skill": skill,
                    "subskill": subskill,
                    "goal": goal,
                    "indicator": "average_" + goal,
                    "perplexity": sum(goal_values) / len(goal_values)
                })
                subskill_values.extend(goal_values)
            rows.append({
                "level": "subskill_avg",
                "skill": skill,
                "subskill": subskill,
                "goal": "",
                "indicator": "average_" + subskill,
                "perplexity": sum(subskill_values) / len(subskill_values)
            })
            skill_values.extend(subskill_values)
        rows.append({
            "level": "skill_avg",
            "skill": skill,
            "subskill": "",
            "goal": "",
            "indicator": "average_" + skill,
            "perplexity": sum(skill_values) / len(skill_values)
        })

    # Convert to DataFrame
    flat_df = pd.DataFrame(rows)

    # Optional: round perplexity for readability
    flat_df['perplexity'] = flat_df['perplexity'].round(4)
    
    print(flat_df)

    # Save to CSV
    flat_df.to_csv(args.output_csv, index=False)
    print(f"Saved results to {args.output_csv}")

if __name__ == "__main__":
    main()
