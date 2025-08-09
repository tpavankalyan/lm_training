'''python collect_results.py --model_path Pavankalyan/stage0_training --data_type instruct --split_type val --stage 0'''
'''Requires that gemma outputs be stored in CL_results/outputs/stage{stage}/{data_type}/{split_type}/{model_name}'''
'''Requires the service account json file in the folder as this file'''

import os
import re
import json
from datasets import load_dataset
import argparse
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import time

def setup_google_sheets():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_file(
        "agentdalal-6ea56a9e5ecc.json", 
        scopes=scope
    )
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
            time.sleep(1)

            data_rows = []
            for _, row in results_df.iterrows():
                data_row = [
                    str(row["level"]) if pd.notna(row["level"]) else "", 
                    str(row["skill"]) if pd.notna(row["skill"]) else "", 
                    str(row["subskill"]) if pd.notna(row["subskill"]) else "", 
                    str(row["goal"]) if pd.notna(row["goal"]) else "", 
                    str(row["indicator"]) if pd.notna(row["indicator"]) else "", 
                    float(row["rating"]) if pd.notna(row["rating"]) else 0.0
                ]
                data_rows.append(data_row)
            
            chunk_size = 100
            for i in range(0, len(data_rows), chunk_size):
                chunk = data_rows[i:i + chunk_size]
                start_row = i + 2
                end_row = start_row + len(chunk) - 1
                range_str = f'A{start_row}:F{end_row}'
                worksheet.update(range_str, chunk, value_input_option='RAW')
            
        else:
            existing_df = pd.DataFrame(existing_data)
            header_row = worksheet.row_values(1)
            new_col_letter = chr(ord('A') + len(header_row))
            
            worksheet.update(values=[[f'{model_name}']], range_name=f'{new_col_letter}1', value_input_option='RAW')
            time.sleep(1)
            
            existing_key_to_row = {}
            for idx, row in existing_df.iterrows():
                key = (
                    str(row.get("level", "")),
                    str(row.get("skill", "")),
                    str(row.get("subskill", "")),
                    str(row.get("goal", "")),
                    str(row.get("indicator", "")),
                )
                existing_key_to_row[key] = idx + 2
            
            batch_data = []
            for _, row in results_df.iterrows():
                key = (
                    str(row["level"]) if pd.notna(row["level"]) else "",
                    str(row["skill"]) if pd.notna(row["skill"]) else "",
                    str(row["subskill"]) if pd.notna(row["subskill"]) else "",
                    str(row["goal"]) if pd.notna(row["goal"]) else "",
                    str(row["indicator"]) if pd.notna(row["indicator"]) else ""
                )
                if key in existing_key_to_row:
                    row_num = existing_key_to_row[key]
                    rating_value = float(row["rating"]) if pd.notna(row["rating"]) else 0.0
                    batch_data.append({
                        'range': f'{new_col_letter}{row_num}',
                        'values': [[rating_value]]
                    })
            
            chunk_size = 50
            for i in range(0, len(batch_data), chunk_size):
                chunk = batch_data[i:i + chunk_size]
                if chunk:
                    try:
                        worksheet.batch_update(chunk, value_input_option='RAW')
                        time.sleep(2)
                        print(f"Updated batch {i//chunk_size + 1}/{(len(batch_data)-1)//chunk_size + 1}")
                    except Exception as batch_error:
                        print(f"Batch update failed: {batch_error}")
                        for update_item in chunk:
                            try:
                                worksheet.update(values=update_item['values'], 
                                                 range_name=update_item['range'], 
                                                 value_input_option='RAW')
                                time.sleep(0.5)
                            except Exception as individual_error:
                                print(f"Failed to update {update_item['range']}: {individual_error}")
                                continue
        
        print(f"Successfully updated '{spreadsheet_name}' -> '{worksheet_name}' with {model_name} ratings")
        
    except Exception as e:
        print(f"Error updating Google Sheet: {e}")
        import traceback
        traceback.print_exc()

def extract_details(row):
    seed_data = row['user']
    f1 = '''\nIndex: '''
    f2 = '''**Output Format:**'''
    s = seed_data.find(f1)
    e = seed_data.find(f2)
    q_in = seed_data[s+len(f1):e].strip()
    return {"q_index": q_in}

def parse_json_string(text):
    try:
        cleaned = text.strip()
        cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^```', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'```$', '', cleaned, flags=re.MULTILINE)

        parsed = json.loads(cleaned)
        return {
            "rating": parsed.get("rating"),
            "explanation": parsed.get("explanation")
        }

    except Exception as e:
        def extract_int(key):
            match = re.search(rf'"{key}"\s*:\s*(\d+)', cleaned)
            return int(match.group(1)) if match else None

        rating = extract_int("rating")
        return {
            "rating": rating,
            "explanation": None
        }

def flatten_parsed_fields(example):
    parsed = parse_json_string(example['answer'])
    return {
        "rating": parsed.get("rating")
    }


def get_result_df(args):
    stage = args.stage
    data_type = args.data_type
    split_type = args.split_type
    model_name = os.path.basename(args.model_path.rstrip('/'))
    model_name = model_name.replace('/', '_').replace('-', '_')
    res_path = f"/datadrive/pavan/az_storage/CL_results/outputs/stage{stage}/{data_type}/{split_type}/{model_name}"
    hf_path = f"Pavankalyan/stage{stage}_{data_type}_eval"
    hf_df = load_dataset(
            "parquet",
            data_files=os.path.join(res_path, "*.parquet"),
            streaming=False
        )

    hf_df = hf_df['train'].map(flatten_parsed_fields)
    hf_df = hf_df.remove_columns(['batch_uuid', 'embeddings', 'generated_tokens', 'messages', 'metrics', 'num_generated_tokens', 'num_input_tokens', 'params', 'prompt', 'prompt_token_ids', 'request_id', 'system', 'time_taken_llm'])
    hf_df = hf_df.map(extract_details)
    hf_df = hf_df.remove_columns(['answer', 'generated_text', 'user'])
    hf_df = hf_df.to_pandas()
    
    ds = load_dataset(hf_path, split=split_type)
    df = ds.to_pandas()
    df['q_index'] = df['q_index'].astype(str)
    hf_df['q_index'] = hf_df['q_index'].astype(str)
    df = df.merge(hf_df, on='q_index', how='left')
    df['rating'] = df['rating'].astype(float)
    return df

def compute_ratings_and_update_sheet(args):
    skill_cols = ["skill", "subskill", "goal", "indicator"]
    
    df = get_result_df(args)
    
    id_avg = df.groupby(['skill', 'goal', 'subskill', 'indicator'])['rating'].mean().reset_index()

    rows = []
    all_skill_values = []
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
                        "rating": row["rating"]
                    })
                    goal_values.append(row["rating"])
                rows.append({
                    "level": "goal_avg",
                    "skill": skill,
                    "subskill": subskill,
                    "goal": goal,
                    "indicator": "average_" + goal,
                    "rating": sum(goal_values) / len(goal_values)
                })
                subskill_values.extend(goal_values)
            rows.append({
                "level": "subskill_avg",
                "skill": skill,
                "subskill": subskill,
                "goal": "",
                "indicator": "average_" + subskill,
                "rating": sum(subskill_values) / len(subskill_values)
            })
            skill_values.extend(subskill_values)
        rows.append({
            "level": "skill_avg",
            "skill": skill,
            "subskill": "",
            "goal": "",
            "indicator": "average_" + skill,
            "rating": sum(skill_values) / len(skill_values)
        })
        all_skill_values.extend(skill_values)

    rows.append({
        "level": "overall_avg",
        "skill": "overall",
        "subskill": "",
        "goal": "",
        "indicator": "average_overall",
        "rating": sum(all_skill_values) / len(all_skill_values)
    })
    
    results_df = pd.DataFrame(rows)
    
    spreadsheet_name = "CL_results"
    worksheet_name = f"{args.data_type}_{args.split_type}_stage{args.stage}"

    if spreadsheet_name and worksheet_name:
        print("Setting up Google Sheets connection...")
        client = setup_google_sheets()
        model_name = os.path.basename(args.model_path.rstrip('/'))
        model_name = model_name.replace('/', '_').replace('-', '_')
        update_google_sheet(client, spreadsheet_name, worksheet_name, results_df, model_name)

def main():
    parser = argparse.ArgumentParser(description="Aggregate ratings and update Google Sheets.")
    parser.add_argument("--model_path", type=str, required=True, help="Name for the model column in Sheets.")
    parser.add_argument("--data_type", type=str, required=True, choices=["instruct", "cqa", "csqa"], help="Path or name of the HF dataset.")
    parser.add_argument("--split_type", type=str, default="val", choices=["val", "test"], help="Dataset split to use.")
    parser.add_argument("--stage", type=int, required=True, help="Stage number for the results.")
    args = parser.parse_args()
    
    compute_ratings_and_update_sheet(args)

if __name__ == "__main__":
    main()

    
    
    
    
    