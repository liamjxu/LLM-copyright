if [ $1 -eq 0 ]; then
python -m src.eval_pred \
    --config paper_experiment \
    --exp_dir local_data/ \
    --model_name gpt-4-turbo  \
    --metrics rejection rouge lcs_score bert_score cos_sim
fi

if [ $1 -eq 1 ]; then
python -m src.eval_pred \
    --config paper_experiment \
    --exp_dir local_data/ \
    --model_name together_ai/meta-llama/Llama-3-8b-chat-hf  \
    --metrics rejection rouge lcs_score bert_score cos_sim  
fi

if [ $1 -eq 2 ]; then
python -m src.eval_pred \
    --config paper_experiment \
    --exp_dir local_data/ \
    --model_name together_ai/meta-llama/Llama-3-70b-chat-hf \
    --metrics rejection rouge lcs_score bert_score cos_sim  
fi

if [ $1 -eq 3 ]; then
python -m src.eval_pred \
    --config paper_experiment \
    --exp_dir local_data/ \
    --model_name together_ai/mistralai/Mistral-7B-Instruct-v0.3 \
    --metrics rejection rouge lcs_score bert_score cos_sim
fi

if [ $1 -eq 4 ]; then
python -m src.eval_pred \
    --config paper_experiment \
    --exp_dir local_data/ \
    --model_name together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --metrics rejection rouge lcs_score bert_score cos_sim
fi

if [ $1 -eq 5 ]; then
python -m src.eval_pred \
    --config paper_experiment \
    --exp_dir local_data/ \
    --model_name together_ai/google/gemma-2-9b-it \
    --metrics rejection rouge lcs_score bert_score cos_sim 
fi