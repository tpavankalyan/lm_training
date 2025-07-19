from datasets import load_dataset, concatenate_datasets
import os
import json
import re


root_dir = "/home/aiscuser/experiments/tinystories_eval"

res_name = "seed_tinystories_ncp_3"


hf_df = load_dataset(
        "parquet",
        data_files=os.path.join(f"{root_dir}/{res_name}", "*.parquet"),
        streaming=False
    )

def parse_json_string(text):
    try:
        # Remove common markdown wrappers
        cleaned = text.strip()
        cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^```', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'```$', '', cleaned, flags=re.MULTILINE)

        parsed = json.loads(cleaned)
        return {
            "grammar": parsed.get("grammar"),
            "creativity": parsed.get("creativity"),
            "consistency": parsed.get("consistency"),
            "explanation": parsed.get("explanation")
        }

    except Exception as e:
        print("JSON parse failed, fallback to regex.")

        # Fallback: extract values manually using regex
        def extract_int(key):
            match = re.search(rf'"{key}"\s*:\s*(\d+)', cleaned)
            return int(match.group(1)) if match else None

        grammar = extract_int("grammar")
        creativity = extract_int("creativity")
        consistency = extract_int("consistency")

        return {
            "grammar": grammar,
            "creativity": creativity,
            "consistency": consistency,
            "explanation": None
        }

def flatten_parsed_fields(example):
    parsed = parse_json_string(example['answer'])
    return {
        "grammar": parsed.get("grammar"),
        "creativity": parsed.get("creativity"),
        "consistency": parsed.get("consistency"),
        "explanation": parsed.get("explanation"),
    }

hf_df['train'] = hf_df['train'].map(flatten_parsed_fields)

import numpy as np

# Extract scores into lists
scores = hf_df['train'].map(lambda x: {
    "grammar": x["grammar"],
    "creativity": x["creativity"],
    "consistency": x["consistency"]
})

# Convert to separate lists
grammar_scores = [x["grammar"] for x in scores]
creativity_scores = [x["creativity"] for x in scores]
consistency_scores = [x["consistency"] for x in scores]

# Compute means and standard deviations
means = {
    "grammar": np.mean(grammar_scores),
    "creativity": np.mean(creativity_scores),
    "consistency": np.mean(consistency_scores),
}

stds = {
    "grammar": np.std(grammar_scores),
    "creativity": np.std(creativity_scores),
    "consistency": np.std(consistency_scores),
}

# Print results
print("=== Evaluation Summary ===")
for key in ["grammar", "creativity", "consistency"]:
    print(f"{key.capitalize():<12} | Mean: {means[key]:.2f}  | Std: {stds[key]:.2f}")
