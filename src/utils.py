TASK_TYPES = [
    "extract",
    "repeat",
    "paraphrase",
    "translate",
]


MODELS = [
    'gpt-4-turbo',
    'together_ai/meta-llama/Llama-3-8b-chat-hf',
    'together_ai/meta-llama/Llama-3-70b-chat-hf',
    'together_ai/mistralai/Mistral-7B-Instruct-v0.1',
    'together_ai/mistralai/Mistral-7B-Instruct-v0.3',
    'together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1',
    'together_ai/google/gemma-2-9b-it',
]

# https://docs.together.ai/docs/chat-models
MODEL_CONTEXT_LENGTHS = {
    'gpt-4-turbo': 4096,  # this is to avoid high cost
    'together_ai/meta-llama/Llama-3-8b-chat-hf': 8192,
    'together_ai/meta-llama/Llama-3-70b-chat-hf': 8192,
    'together_ai/mistralai/Mistral-7B-Instruct-v0.1': 8192,
    'together_ai/mistralai/Mistral-7B-Instruct-v0.3': 32768,
    'together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1': 32768,
    'together_ai/google/gemma-2-9b-it': 8192,
}
