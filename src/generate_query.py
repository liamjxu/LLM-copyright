import argparse
import os
import json
import math

from litellm import completion
from tqdm import tqdm
from src.utils import TASK_TYPES


class QueryGenerator:
    def __init__(self, args):
        for k, v in vars(args).items(): setattr(self, k, v)

        self.base_queries = self._setup_base_queries()

    def _setup_base_queries(self):
        base_query_filename = None
        if self.config in ['orig', 'orig_warn', 'orig_rewrite']:
            base_query_filename = 'seeds.json'
        if self.config in ['orig_kw', 'orig_kw_warn']:
            base_query_filename = 'keywords.json'  # this is `seeds.json` with keywords added manually
        with open(os.path.join(self.raw_query_dir, base_query_filename), 'r') as f:
            base_queries = json.load(f)
        return base_queries

    def generate_queries(self):
        if self.config in ['orig', 'orig_kw']:
            self._handle_base()
        elif self.config == 'orig_warn':
            self._handle_orig_warn()
        elif self.config == 'orig_kw_warn':
            self._handle_orig_kw_warn()
        elif self.config == 'orig_rewrite':
            self._handle_orig_rewrite()
        else:
            raise ValueError(f"Invalid config: {self.config}")

    def _organize_query(self):
        all_query = []
        for task_idx, task in enumerate(TASK_TYPES):
            queries = self.base_queries[task]
            for seed_query_idx, query in enumerate(queries):
                all_query.append({
                    'query/task_idx': task_idx,
                    'query/seed_query_idx': seed_query_idx,
                    'query/task': task,
                    'query/query': query,
                })
        return all_query

    def _handle_base(self):
        all_query = self._organize_query()
        os.makedirs(self.output_query_dir, exist_ok=True)
        with open(os.path.join(self.output_query_dir, f'query_{self.config}.json'), 'w', encoding='utf-8') as f:
            json.dump(all_query, f, indent=4)
    
    def _handle_orig_warn(self):
        # TODO:
        pass

    def _handle_orig_kw_warn(self):
        # TODO:
        pass

    def _handle_orig_rewrite(self):
        all_query = self._organize_query()
        all_rewriting = []
        entry_idx = 0
        for entry in tqdm(all_query):
            query = entry['query/query']
            rewritings = []
            total_prob = 0

            for query_rewriting_idx in range(self.n_rewrite):
                temperature = 1.5
                response = completion(
                    model="gpt-4-turbo",
                    messages=[
                        {'content': f'You are a rewriting agent that is capable of rewriting natural language instructions. Your task is to rewrite a task instruction for an large language model.', 'role': 'user'},
                        {'content': f'Some disambiguation: 1) when the task is "to extract some sentence", it means to retrieve/identify/locate/find/etc that sentence, not to remove. ', 'role': 'user'},
                        {'content': f'Rewrite this instruction, ensuring that the task remains unchanged in your new version: {query}', 'role': 'user'},
                    ],
                    temperature=temperature,
                    num_retries=3,
                    logprobs=True,
                )
                query_rewriting = response['choices'][0]['message']['content']
                logprobs = response['choices'][0]['logprobs']['content']
                query_rewriting_prob = math.exp(sum([_['logprob'] for _ in logprobs]))
                total_prob += query_rewriting_prob

                rewritings.append({
                    "query/entry_idx": entry_idx,
                    "query/task_idx": entry['query/task_idx'],
                    "query/seed_query_idx": entry['query/seed_query_idx'],
                    "query/query_rewriting_idx": query_rewriting_idx,
                    "query/task": entry['query/task'],
                    "query/seed_query": query,
                    "query/rewritten_query": query_rewriting,
                    "query/query_rewriting_prob": query_rewriting_prob,
                    "query/query_rewriting_model": "gpt-4-turbo",
                    "query/query_rewriting_temperature": temperature,
                    "query/query_rewriting_full_logprobs": logprobs,
                    "query/query_rewriting_full_response": str(response),
                })
                entry_idx += 1

            for rewriting in rewritings:
                rewriting["query/query_rewriting_weight"] = rewriting["query/query_rewriting_prob"] / total_prob
                all_rewriting.append(rewriting)

            os.makedirs(self.output_query_dir, exist_ok=True)
            with open(os.path.join(self.output_query_dir, f'query_{self.config}.json'), 'w', encoding='utf-8') as f:
                json.dump(all_rewriting, f, indent=4)

            print('Unique Ratio:', len(set(_['query/rewritten_query'] for _ in all_rewriting)) / len(all_rewriting))

def main(args):

    if args.config == 'orig_rewrite':
        with open(args.rewrite_cred_filename, 'r') as f:
            cred = f.read()
        os.environ["OPENAI_API_KEY"] = cred

    query_generator = QueryGenerator(args)
    query_generator.generate_queries()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate queries for the experiment')
    parser.add_argument('--raw_query_dir', type=str)
    parser.add_argument('--output_query_dir', type=str)
    parser.add_argument('--config', type=str, choices=['orig', 'orig_rewrite'])
    basic_args, _ = parser.parse_known_args()

    if basic_args.config == 'orig_rewrite':
        parser.add_argument('--rewrite_cred_filename', type=str, required=True, help='Filename for rewrite credentials')
        parser.add_argument('--n_rewrite', type=int, required=True, help='Number of rewrites')
    
    args = parser.parse_args()
    main(args)
