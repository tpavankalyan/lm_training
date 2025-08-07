CUDA_VISIBLE_DEVICES=0 python perplexity_indicator.py --model_path Pavankalyan/C_0_padded_11 --dataset_path Pavankalyan/stage0_c_all 
CUDA_VISIBLE_DEVICES=1 python perplexity_indicator.py --model_path Pavankalyan/C_0_padded_11 --dataset_path Pavankalyan/stage1_c_all 
CUDA_VISIBLE_DEVICES=2 python perplexity_indicator.py --model_path Pavankalyan/C_0_padded_11 --dataset_path Pavankalyan/stage2_c_all 
CUDA_VISIBLE_DEVICES=3 python perplexity_indicator.py --model_path Pavankalyan/C_0_padded_11 --dataset_path Pavankalyan/stage3_c_all 
CUDA_VISIBLE_DEVICES=4 python perplexity_indicator.py --model_path Pavankalyan/C_0_padded_11 --dataset_path Pavankalyan/stage4_c_all 

