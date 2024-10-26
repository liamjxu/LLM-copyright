if [ $1 -eq 0 ]; then
python -m src.model_infer \
    --config paper_experiment \
    --exp_dir local_data/ \
    --model gpt-4-turbo 
fi

if [ $1 -eq 1 ]; then
python -m src.model_infer \
    --config paper_experiment \
    --exp_dir local_data/ \
    --model together_ai/meta-llama/Llama-3-8b-chat-hf 
fi

if [ $1 -eq 2 ]; then
python -m src.model_infer \
    --config paper_experiment \
    --exp_dir local_data/ \
    --model together_ai/meta-llama/Llama-3-70b-chat-hf
fi

if [ $1 -eq 3 ]; then
python -m src.model_infer \
    --config paper_experiment \
    --exp_dir local_data/ \
    --model together_ai/mistralai/Mistral-7B-Instruct-v0.3
fi

if [ $1 -eq 4 ]; then
python -m src.model_infer \
    --config paper_experiment \
    --exp_dir local_data/ \
    --model together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1
fi

if [ $1 -eq 5 ]; then
python -m src.model_infer \
    --config paper_experiment \
    --exp_dir local_data/ \
    --model together_ai/google/gemma-2-9b-it
fi
