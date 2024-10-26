import os
import json
import random
import argparse
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from litellm import completion
from math import exp


class DatasetGenerator():
    def __init__(self, args):

        for k, v in self._handle_args(args).items(): setattr(self, k, v)

        self._read_data()

    def _handle_args(self, args):
        res = vars(args)
        if args.config in ['paper_experiment', 'experiment_release']:
            res['notice_positions'] = ['middle']
        else:
            raise ValueError(f"Cannot determine notice position for config: {args.config}")
        print("Full Arguments:", res)
        return res

    def _read_data(self):
        with open(self.query_path, 'r') as f:
            self.queries = json.load(f)
        with open(self.source_text_path, 'r') as f:
            self.source_texts = json.load(f)
        self.notices = self._get_notices()

    def _get_notices(self):
        # Read the notice into relational format
        # TODO: decouple this from the dataset generation
        with open(os.path.join(self.notice_dir, "copyright_notices.json"), 'r') as f:
            raw_notices = json.load(f)
        with open(os.path.join(self.notice_dir, "full_content.json"), 'r') as f:
            full_content = json.load(f)

        # common notices, which are the same for all source texts
        notices = [{'notice/level_idx': level_idx,
                    'notice/source_idx': source_idx,
                    'notice/level': level_entry['level'],
                    'notice/source_name': source_entry['content_name'],
                    'notice/notice': level_entry['description']
                    } for level_idx, level_entry in enumerate(raw_notices) for source_idx, source_entry in enumerate(full_content)]

        # get the full_content, which contains the copyright notices for each source text
        original = [{'notice/level_idx': 2,
                     'notice/source_idx': source_idx,
                     'notice/level': 'original',
                     'notice/source_name': source_entry['content_name'],
                     'notice/notice': source_entry['copyright_notice']} for source_idx, source_entry in enumerate(full_content)]

        # combine and save
        res = notices + original
        notice_dir = os.path.join(self.exp_dir, "notice")
        os.makedirs(notice_dir, exist_ok=True)
        with open(os.path.join(notice_dir, f"notice_{self.config}.json"), 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=4, ensure_ascii=False)
        return res
    
    def generate_dataset(self):
        query_df = pd.DataFrame(self.queries)
        notice_df = pd.DataFrame(self.notices)
        source_df = pd.DataFrame(self.source_texts)

        notice_df = notice_df[notice_df['notice/source_name'].isin(source_df['source/source_name'])]
        context_df = pd.merge(notice_df, source_df, left_on='notice/source_name', right_on='source/source_name')
        full_df = pd.merge(context_df, query_df, how='cross')
        print("Number of notice rows used:", len(notice_df), "Number of sources used:", len(source_df))
        print("Number of query rows used:", len(query_df))
        print("Number of data rows generated:", len(full_df))

        full_df = full_df.drop(columns=['source/full_text', 'query/query_rewriting_full_response', 'query/query_rewriting_full_logprobs'])  # cut out to save space
        # f"{sample}\n\n{self.copyright_notice}"
        full_df['prompt'] = full_df.apply(lambda row: f"{row['source/sample']}\n{row['notice/notice']}\n{row['query/rewritten_query']}", axis=1)

        dataset_dir = os.path.join(self.exp_dir, 'dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        full_df.to_json(os.path.join(dataset_dir, f"dataset_{self.config}.json"), orient='records', indent=4, force_ascii=False)

    def get_data_chunk(self):
        res = []
        context_word_lens = [100, 500, 1000]
        context_samples = self.sample_context(word_lens=context_word_lens)
        warned_query = self.get_warned_query()
        if self.notice_position == 'ending':
            full_query = f"{warned_query}\n\n{self.copyright_notice}"
        else:
            full_query = warned_query

        for context_idx, context_sample in enumerate(context_samples):
            res.append({
                'query': full_query,
                'context': context_sample,

                'query_type': self.query_rewriting_entry['query_type'],
                'query_type_idx': self.query_rewriting_entry['query_type_idx'],
                'seed_query': self.query_rewriting_entry['seed_query'],
                'query_rewriting_idx': self.query_rewriting_entry['query_rewriting_idx'],
                'query_rewriting_prob': self.query_rewriting_entry['query_rewriting_prob'],

                'context_source_type': self.context_source_type,
                'context_source_idx': self.context_source_idx,
                
                'notice_type': self.notice_type,
                'notice_position': self.notice_position,
                'copyright_notice_idx': self.copyright_notice_idx,
                'copyright_notice': self.copyright_notice,
                
                'warning_type': self.warning_type,
                'warning_position': self.warning_position,
                'copyright_warning_idx': self.copyright_warning_idx,
                'copyright_warning': self.copyright_warning,
                
                'sample_len': context_word_lens[context_idx],
            })
        return res
    
    def get_warned_query(self) -> str:
        return f"{self.copyright_warning}\n\n{self.query_rewriting_entry['query_rewriting']}"

    def sample_context(self, word_lens: List[int]) -> List[str]:
        res = []
        words = self.context_source_text.split()
        for word_len in word_lens:
            start = random.randint(0, max(0, len(words) - 1 - max(word_lens)))
            sample = ' '.join(words[start:start+word_len])
            if self.notice_position == 'beginning':
                full_sample = f"{self.copyright_notice}\n\n{sample}"
            if self.notice_position == 'middle':
                full_sample = f"{sample}\n\n{self.copyright_notice}"
            else:
                full_sample = f"{sample}"
            res.append(full_sample)
        return res


def main(args):

    dataset_generator = DatasetGenerator(args)
    dataset_generator.generate_dataset()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate datasets for the experiment')
    parser.add_argument("--config", type=str, choices=["paper_experiment", "benchmark_release"])
    parser.add_argument("--exp_dir", type=str)
    parser.add_argument("--query_path", type=str)
    parser.add_argument("--source_text_path", type=str)
    parser.add_argument("--notice_dir", type=str)
    # TODO: add the args for `kw and warn``

    args = parser.parse_args()
    main(args)
