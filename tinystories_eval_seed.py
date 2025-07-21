from datasets import Dataset, load_dataset
import pickle

ds = load_dataset("Pavankalyan/TinyStories_eval", split="validation")
root_dir = "/scratch/azureml/cr/j/3ea997ee1d70452ca9a4b196bb1f5556/cap/data-capability/wd/INPUT_asdf/tinystories_data"
# candidates = ["gt", "tinystories_ncp_1", "tinystories_ncp_2", "tinystories_ncp_3", "tinystories_cp_2", "tinystories_cp_1"]
candidates = ["tinystories_33M_rep4"]

#for each candidate, make a list of dictionary, with each dictionary containing the prompt and the candidate text
results = {}
for candidate in candidates:
    results[candidate] = []
    for i in range(len(ds)):
        results[candidate].append({
            "story_beginning": ds[i]["prompt"],
            "story_continuation": ds[i][candidate]
        })
        
    # Save the results to a pkl file
    with open(f"{root_dir}/seed_{candidate}.pkl", "wb") as f:
        pickle.dump(results[candidate], f)