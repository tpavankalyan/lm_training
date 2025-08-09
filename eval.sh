#Upload the model to hf
#Batch inference
#Save the eval seed
#Collect results
python upload_to_hf.py <model_path>
python batched_inference.py --model_path Pavankalyan/checkpoint-13281 --data_type instruct --split_type val --stage 0
python eval_seed.py --model_path Pavankalyan/checkpoint-13281 --data_type instruct --split_type val --stage 0
python collect_results.py --model_path Pavankalyan/checkpoint-13281 --data_type instruct --split_type val --stage 0