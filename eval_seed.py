from datasets import Dataset, load_dataset
import pickle
import argparse

def prepare_seed_csqa(stage, split_type, model_name, root_dir):
    ds = load_dataset(f"Pavankalyan/stage{stage}_{data_type}_eval_results", split=split_type)
    seeds = []
    for i in range(len(ds)):
        seeds.append({
            'context': ds[i]['context'],
            'instruction': ds[i]['question'],
            'response': ds[i][model_name],
            'stage': stage,
            'age_group': ds[i]['age_group'],
            'skill': ds[i]['skill'],
            'subskill': ds[i]['subskill'],
            'goal': ds[i]['goal'],
            'indicator': ds[i]['indicator'],
            'q_index': ds[i]['q_index']
        })

    with open(f"{root_dir}/stage{stage}/{data_type}/{split_type}/{model_name}.pkl", "wb") as f:
        pickle.dump(seeds, f)

def prepare_seed_cqa(stage, split_type, model_name, root_dir):
    ds = load_dataset(f"Pavankalyan/stage{stage}_cqa_eval_results", split=split_type)
    seeds = []
    for i in range(len(ds)):
        seeds.append({
            'context': ds[i]['context'],
            'question': ds[i]['question'],
            'answer': ds[i][model_name],
            'stage': stage,
            'age_group': ds[i]['age_group'],
            'q_index': ds[i]['q_index']
        })
        
    with open(f"{root_dir}/stage{stage}/{data_type}/{split_type}/{model_name}.pkl", "wb") as f:
        pickle.dump(seeds, f)
        
def prepare_seed_ir(stage, split_type, model_name, root_dir):
    ds = load_dataset(f"Pavankalyan/stage{stage}_instruct_eval_results", split=split_type)
    seeds = []
    for i in range(len(ds)):
        seeds.append({
            'instruction': ds[i]['question'],
            'response': ds[i][model_name],
            'stage': stage,
            'age_group': ds[i]['age_group'],
            'q_index': ds[i]['q_index']
        })
        
    with open(f"{root_dir}/stage{stage}/{data_type}/{split_type}/{model_name}.pkl", "wb") as f:
        pickle.dump(seeds, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save seed for HF")
    parser.add_argument("--model_path", type=str, required=True, help="Name for the model column in Sheets.")
    parser.add_argument("--data_type", type=str, required=True, choices=["instruct", "cqa", "csqa"], help="Path or name of the HF dataset.")
    parser.add_argument("--split_type", type=str, default="val", choices=["val", "test"], help="Dataset split to use.")
    parser.add_argument("--stage", type=int, required=True, help="Stage number for the results.")
    args = parser.parse_args()
    
    root_dir = "/datadrive/pavan/az_storage/CL_results/seed"
    stage = args.stage
    model_name = args.model_path.split("/")[-1].replace('/', '_').replace('-', '_')
    split_type = args.split_type
    data_type = args.data_type
    
    print(f"Preparing seed for {data_type}...")
    if data_type == "instruct":
        prepare_seed_ir(stage, split_type, model_name, root_dir)
    elif data_type == "cqa":
        prepare_seed_cqa(stage, split_type, model_name, root_dir)
    elif data_type == "csqa":
        prepare_seed_csqa(stage, split_type, model_name, root_dir)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
        
        
    
    