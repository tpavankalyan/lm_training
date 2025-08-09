#Upload the model to hf
#Batch inference
#Save the eval seed
#Collect results
python upload_to_hf.py <model_path>
python batched_inference.py --model_path Pavankalyan/stage0_training --data_type instruct --split_type val --stage 2
python eval_seed.py --model_path Pavankalyan/stage0_training --data_type instruct --split_type val --stage 2
python run_inference.py --model_path Pavankalyan/stage0_training --data_type instruct --split_type val --stage 2
python collect_results.py --model_path Pavankalyan/stage0_training --data_type instruct --split_type val --stage 2