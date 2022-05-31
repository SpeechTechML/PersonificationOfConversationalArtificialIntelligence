import os 
import argparse
from typing import Literal
from omegaconf import OmegaConf, DictConfig

import pandas as pd

from utils.translator import RusEngTranslator
from utils.task import Task


class TranslationTask:
    def __init__(self, config: DictConfig, kind: Literal['ru_en', 'en_ru']):
        self.config = config
        self.translator = RusEngTranslator(
            mt_config   = config['task']['model'], 
            kind        = kind, 
            device      = config['task']['device']
        )

    def _translate_base(self, base_name: str):
        print(f"Base: {base_name}")

        # Load data config
        data_config = OmegaConf.load(f'configs/data/{base_name}.yaml')

        # Iterate through partitions and translate all texts
        for part, fname in data_config['partitions'].items():
            current_partition_path = os.path.join(data_config['base_path'], fname)
            df = pd.read_csv(current_partition_path)
            
            # Translate texts
            translations = self.translator.translate_texts(
                texts = df['utterance'].values,
                verbose = True,
                desc = f'Processing of {part}'
            )

            # Write translations in another column and resave source partition .csv-file
            df['translated_utterance'] = translations
            df.to_csv(current_partition_path, index=False)

    def run(self):
        for base_name in self.config['datasets']:
            self._translate_base(base_name)
        

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='configs/tasks/nmt.yaml', type=str)
    parser.add_argument('--kind', default='en_ru', type=str, choices=['en_ru', 'ru_en'])
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    task = TranslationTask(config, args.kind)
    task.run()


if __name__ == '__main__':
    main()