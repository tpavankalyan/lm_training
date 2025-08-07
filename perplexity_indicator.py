import argparse
import math
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import gspread
from google.oauth2.service_account import Credentials
import os
from datetime import datetime
import time

def setup_google_sheets():
    """Setup Google Sheets API credentials"""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    
    creds = Credentials.from_service_account_file("/datadrive/pavan/experiments/agentdalal-6ea56a9e5ecc.json", scopes=scope)
    client = gspread.authorize(creds)
    return client

def update_google_sheet(client, spreadsheet_name, worksheet_name, results_df, model_name):
    try:
        spreadsheet = client.open(spreadsheet_name)
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)
            print(f"Created new worksheet: {worksheet_name}")
        
        try:
            existing_data = worksheet.get_all_records()
        except:
            existing_data = []
        
        if not existing_data:
            base_columns = ["level", "skill", "subskill", "goal", "indicator"]
            header_row = base_columns + [f"{model_name}"]
            worksheet.clear()
            worksheet.update(values=[header_row], range_name='A1', value_input_option='RAW')
            time.sleep(1)  # Rate limiting

            data_rows = []
            for _, row in results_df.iterrows():
                data_row = [
                    str(row["level"]) if pd.notna(row["level"]) else "", 
                    str(row["skill"]) if pd.notna(row["skill"]) else "", 
                    str(row["subskill"]) if pd.notna(row["subskill"]) else "", 
                    str(row["goal"]) if pd.notna(row["goal"]) else "", 
                    str(row["indicator"]) if pd.notna(row["indicator"]) else "", 
                    float(row["perplexity"]) if pd.notna(row["perplexity"]) else 0.0
                ]
                data_rows.append(data_row)
            
            chunk_size = 100
            for i in range(0, len(data_rows), chunk_size):
                chunk = data_rows[i:i + chunk_size]
                start_row = i + 2  # +2 because row 1 is header and rows are 1-indexed
                end_row = start_row + len(chunk) - 1
                range_str = f'A{start_row}:F{end_row}'
                worksheet.update(range_str, chunk, value_input_option='RAW')
            
        else:
            existing_df = pd.DataFrame(existing_data)
            
            header_row = worksheet.row_values(1)
            new_col_letter = chr(ord('A') + len(header_row))
            
            worksheet.update(values=[[f'{model_name}_perplexity']], range_name=f'{new_col_letter}1', value_input_option='RAW')
            time.sleep(1)  # Rate limiting
            
            existing_key_to_row = {}
            for idx, row in existing_df.iterrows():
                level = str(row.get("level", "")) if pd.notna(row.get("level", "")) else ""
                skill = str(row.get("skill", "")) if pd.notna(row.get("skill", "")) else ""
                subskill = str(row.get("subskill", "")) if pd.notna(row.get("subskill", "")) else ""
                goal = str(row.get("goal", "")) if pd.notna(row.get("goal", "")) else ""
                indicator = str(row.get("indicator", "")) if pd.notna(row.get("indicator", "")) else ""
                
                key = (level, skill, subskill, goal, indicator)
                existing_key_to_row[key] = idx + 2  # +2 because row numbers are 1-indexed and we skip header
            
            batch_data = []
            for _, row in results_df.iterrows():
                level = str(row["level"]) if pd.notna(row["level"]) else ""
                skill = str(row["skill"]) if pd.notna(row["skill"]) else ""
                subskill = str(row["subskill"]) if pd.notna(row["subskill"]) else ""
                goal = str(row["goal"]) if pd.notna(row["goal"]) else ""
                indicator = str(row["indicator"]) if pd.notna(row["indicator"]) else ""
                
                key = (level, skill, subskill, goal, indicator)
                if key in existing_key_to_row:
                    row_num = existing_key_to_row[key]
                    perplexity_value = float(row["perplexity"]) if pd.notna(row["perplexity"]) else 0.0
                    batch_data.append({
                        'range': f'{new_col_letter}{row_num}',
                        'values': [[perplexity_value]]
                    })
            
            chunk_size = 50  # Reduced chunk size for rate limiting
            for i in range(0, len(batch_data), chunk_size):
                chunk = batch_data[i:i + chunk_size]
                if chunk:
                    try:
                        worksheet.batch_update(chunk, value_input_option='RAW')
                        time.sleep(2)  # Rate limiting - wait 2 seconds between batches
                        print(f"Updated batch {i//chunk_size + 1}/{(len(batch_data)-1)//chunk_size + 1}")
                    except Exception as batch_error:
                        print(f"Batch update failed, falling back to individual updates: {batch_error}")
                        for update_item in chunk:
                            try:
                                worksheet.update(values=update_item['values'], 
                                               range_name=update_item['range'], 
                                               value_input_option='RAW')
                                time.sleep(0.5)  # Rate limiting between individual updates
                            except Exception as individual_error:
                                print(f"Failed to update {update_item['range']}: {individual_error}")
                                continue
        
        print(f"Successfully updated '{spreadsheet_name}' -> '{worksheet_name}' with {model_name} results")
        
    except Exception as e:
        print(f"Error updating Google Sheet: {e}")
        import traceback
        traceback.print_exc()

def compute_perplexity_and_update_sheet(args):
    skill_cols = ["skill", "subskill", "goal", "indicator"]
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
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

        row = [example[col] for col in skill_cols]
        row.append(perplexity)
        results.append(row)
        
    df = pd.DataFrame(results, columns=skill_cols + ["perplexity"])
    id_avg = df.groupby(['skill', 'goal', 'subskill', 'indicator'])['perplexity'].mean().reset_index()

    rows = []
    all_skill_values=[]
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
        all_skill_values.extend(skill_values)

    rows.append({
        "level": "overall_avg",
        "skill": "overall",
        "subskill": "",
        "goal": "",
        "indicator": "average_overall",
        "perplexity": sum(all_skill_values) / len(all_skill_values)
    })
    
    results_df = pd.DataFrame(rows)
    
    spreadsheet_name = "CL_results"
    w_name = args.dataset_path.split("/")[-1].split("_")[0][-1]
    worksheet_name = f"C_{w_name}"

    if spreadsheet_name and worksheet_name:
        print("Setting up Google Sheets connection...")
        client = setup_google_sheets()
        model_name = os.path.basename(args.model_path.rstrip('/'))
        model_name = model_name.replace('/', '_').replace('-', '_')
        update_google_sheet(client, spreadsheet_name, worksheet_name, results_df, model_name)

def main():
    parser = argparse.ArgumentParser(description="Compute perplexity on a specified dataset split and update Google Sheets.")
    parser.add_argument("--model_path", type=str, required=True, help="Hugging Face model path or ID.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path or name of the HF dataset.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to use.")
    parser.add_argument("--text_column", type=str, default="output", help="Column name containing text.")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--output_csv", type=str, default="perplexity_results.csv", help="Output CSV file.")
    args = parser.parse_args()
    
    compute_perplexity_and_update_sheet(args)

if __name__ == "__main__":
    main()