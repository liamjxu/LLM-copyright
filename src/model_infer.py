import os
import json
import pandas as pd
import time
from tqdm import tqdm
from litellm import completion, token_counter
from argparse import ArgumentParser
from src.utils import MODELS, MODEL_CONTEXT_LENGTHS


def main(args):

    dataset_df = pd.read_json(os.path.join(args.exp_dir, 'dataset', f'dataset_{args.config}.json'))
    model_name_for_record = args.model.replace('/', '_')

    # Get dataset sample df
    output_path = os.path.join(args.exp_dir, 'output', f"dataset_{args.config}", f'dataset_{args.config}_preds_{model_name_for_record}.json')
    if args.is_continue:
        with open(output_path, 'r') as f:
            res = json.load(f)
        starting_idx = res[-1]['dataset_idx'] + 1 # because there might be previous failures, using len(res) is not reliable
        sample_df = dataset_df.iloc[starting_idx:]  # here we know that our dataset_df index is continuous, because we built it from a json file, 
                                                    # so we can use +1 and not worry about missing indices
    else:
        res = []
        sample_df = dataset_df

    # Inference
    entry_idx = 0 if len(res) == 0 else res[-1]['dataset_idx'] + 1
    for row_idx, row in tqdm(sample_df.iterrows(), total=sample_df.shape[0]):
        tic = time.time()
        def compose_messages(sample, notice, query):
            context = f"{sample}\n{notice}"
            messages = [
                {"content": context, "role": "user"},
                {"content": query, "role": "user"},
            ]
            return messages

        sample = row['source/sample']
        notice = row['notice/notice']
        query = row['query/rewritten_query']
        messages = compose_messages(sample, notice, query)
        token_counts = token_counter(model=args.model, messages=messages)

        while token_counts > MODEL_CONTEXT_LENGTHS[args.model]:
            sample = sample[:int(len(sample) * 0.9)]
            messages = compose_messages(sample, notice, query)
            token_counts = token_counter(model=args.model, messages=messages)

        response_success = False
        try:
            response = completion(
                model=args.model,
                messages = messages,
                temperature=0.0,
                max_retries=5,
                timeout=20
            )
            response_success = True
        except Exception as e:
            error_str = str(e)
            if "524: A timeout occurred" in error_str:
                print(f"Timeout error: {row_idx}")
                toc = time.time()
                res.append({
                    'entry_idx': entry_idx,
                    'model': args.model,
                    'dataset_idx': row_idx,
                    'sample': sample,
                    'notice': notice,
                    'query': query,
                    'pred': "The model experienced an error. ",
                    'prompt_tokens': response.usage['prompt_tokens'],
                    'completion_tokens': response.usage['completion_tokens'],
                    'time': toc - tic,
                    'full_response': str(response),
                })
                
        if response_success:
            toc = time.time()
            res.append({
                'entry_idx': entry_idx,
                'dataset_idx': row_idx,
                'model': args.model,
                'sample': sample,
                'notice': notice,
                'query': query,
                'pred': response['choices'][0]['message']['content'],
                'prompt_tokens': response.usage['prompt_tokens'],
                'completion_tokens': response.usage['completion_tokens'],
                'time': toc - tic,
                'full_response': str(response),
            })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=4, ensure_ascii=False)

        entry_idx += 1

if __name__ == '__main__':

    with open('openai_cred.txt', 'r') as f: openai_cred = f.read()
    with open('togetherai_cred.txt', 'r') as f: togetherai_cred = f.read()
    os.environ["OPENAI_API_KEY"] = openai_cred
    os.environ["TOGETHERAI_API_KEY"] = togetherai_cred

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, choices=["paper_experiment", "benchmark_release"])
    parser.add_argument("--exp_dir", type=str)
    parser.add_argument('--model', type=str, choices=MODELS)
    parser.add_argument('--is_continue', action='store_true')
    args = parser.parse_args()
    main(args)