import argparse
import os
import json
import random
random.seed(42)

class SourceTextGenerator:
    def __init__(self, args):
        for k, v in vars(args).items(): setattr(self, k, v)

        self.raw_source_texts = self._read_raw_source_text()

    def _read_raw_source_text(self):
        raw_source_texts = []

        source_types = ['book', 'doc', 'news', 'scripts']
        subclass_mapping = {
            'book': ['2024', 'old'],
            'doc': [''],
            'news': ['BBC', 'NYT', 'Reuters'],
            'scripts': ['']
        }

        for type_idx, source_type in enumerate(source_types):
            within_type_idx = 0
            source_dir = os.path.join(self.raw_source_text_dir, source_type)
            for subclass_idx, subclass in enumerate(subclass_mapping[source_type]):
                for within_subclass_idx, filename in enumerate(os.listdir(os.path.join(source_dir, subclass))):
                    with open(os.path.join(source_dir, subclass, filename), 'r', encoding='iso-8859-1') as f:
                        full_text = f.read()
                    for sample_len in [100, 500, 1000]:
                        raw_source_texts.append({
                            'source/type_idx': type_idx,
                            'source/subclass_idx': subclass_idx,
                            'source/within_type_idx': within_type_idx,
                            'source/within_subclass_idx': within_subclass_idx,
                            'source/type': source_type,
                            'source/subclass': subclass,
                            'source/source_name': filename,  # special naming, for joining purposes
                            'source/sample_len': sample_len,
                            'source/sample': self._get_sample(full_text, sample_len),
                            'source/full_text': full_text
                        })
                    within_type_idx += 1  # this is used when we want to use only the first text from each source type

        return raw_source_texts
    
    def _get_sample(self, text, sample_len):
        words = text.split()
        start = random.randint(0, max(0, len(words) - 1 - sample_len))
        sample = ' '.join(words[start:start+sample_len])
        return sample

    def generate_source_text(self):
        if self.config == 'paper_experiment':
            self._handle_paper_experiment()
        elif self.config == 'benchmark_release':
            self._handle_benchmark_release()
        else:
            raise ValueError(f"Invalid config: {self.config}")

    def _handle_paper_experiment(self):
        # Grab only the first text from each source type
        experiment_source_texts = [entry for entry in self.raw_source_texts if entry['source/within_type_idx'] < 1]
        res = [{**{'source/entry_idx': entry_idx}, **entry} for entry_idx, entry in enumerate(experiment_source_texts)]
        os.makedirs(self.output_source_text_dir, exist_ok=True)
        with open(os.path.join(self.output_source_text_dir, f'source_texts_{self.config}.json'), 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=4)

    def _handle_benchmark_release(self):
        # Use all texts
        res = [{**{'source/entry_idx': entry_idx}, **entry} for entry_idx, entry in enumerate(self.raw_source_texts)]
        os.makedirs(self.output_source_text_dir, exist_ok=True)
        with open(os.path.join(self.output_source_text_dir, f'source_texts_{self.config}.json'), 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=4)


def main(args):

    source_text_generator = SourceTextGenerator(args)
    source_text_generator.generate_source_text()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate source texts for the experiment')
    parser.add_argument('--raw_source_text_dir', type=str)
    parser.add_argument('--output_source_text_dir', type=str)
    parser.add_argument('--config', type=str, choices=['paper_experiment', 'benchmark_release'])

    args = parser.parse_args()
    main(args)
