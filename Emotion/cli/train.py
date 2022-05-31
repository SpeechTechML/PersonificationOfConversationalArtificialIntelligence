import warnings
warnings.filterwarnings('ignore')

import os
import tqdm
import random
import argparse
from typing import Tuple, Iterable
from omegaconf import OmegaConf, DictConfig

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset
from transformers import Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from utils.base import set_seed, str_to_python
from utils.task import Task
from utils.label_encoder import LabelEncoderWithFitCheck



class TrainTask(Task):
    def __init__(self, config: DictConfig):
        self.config = config
        self.tokenizer = RobertaTokenizer.from_pretrained(
            **config['task']['tokenizer']
        )
        self.model = RobertaForSequenceClassification.from_pretrained(
            **config['task']['model']
        )
        self.le = LabelEncoderWithFitCheck(
            config['task']['label_encoder']['save_dir']
        )


    def _tokenize_function(self, examples):
        return self.tokenizer(examples['text'], padding='max_length', truncation=True)


    def _compute_metrics(self, eval_pred: Tuple[Iterable, Iterable]):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'weighted_f1': f1_score(labels, predictions, average='weighted')
        }
        return metrics


    def _preprocess_dataset(self, df_dataset: pd.DataFrame) -> pd.DataFrame:
        """ Renames columns and encodes labels """

        # Edit columns and order
        df_dataset.rename(columns={
            'emo': 'label', 
            'emotion': 'label',
            'dialogue_id': 'dia_id',
            'utterance_id': 'utt_id',
            'translated_utterance_aug': 'utterance_aug'
        }, inplace=True)
        df_dataset.sort_values(by=['dia_id', 'utt_id'], inplace=True)

        # Encode labels
        if self.le.is_fitted:
            df_dataset['label'] = self.le.transform(df_dataset['label'])
        else:
            df_dataset['label'] = self.le.fit_transform(df_dataset['label'])

        # Convert columns with '_aug' to pythonic list notation (instead of pandas-converted string)
        df_dataset['utterance_aug'] = df_dataset['utterance_aug'].apply(str_to_python)

        return df_dataset


    def _extract_dialogue_contexts(
        self, 
        df_dataset: pd.DataFrame, 
        ds_name: str, 
        aug_factor: int = 1
        ) -> pd.DataFrame:
        """ 
        Extracts dialogues contexts for each utterance and corresponding targets
        using augmentations (if aug_factor > 1). 
        """

        dset_config = self.config['task']['dataset']
        context_size = dset_config['dialogue_context_size']
        sep_token = dset_config['sep_token']
        contexts, labels = [], []

        # Iterate through augmentation cycles
        for aug_iter in range(aug_factor):
            print(f"Aug iteration: {aug_iter + 1}/{aug_factor}")

            for i in tqdm.tqdm(range(df_dataset.shape[0]), desc=ds_name):
                curr_row =  df_dataset.iloc[i,:]
                current_dialogue_id = curr_row['dia_id']
                current_context = []

                # Iterate through context
                for j in range(context_size)[::-1]:
                    curr_index = i - j

                    # Check if index is not out of bounds
                    if curr_index >= 0:
                        row = df_dataset.iloc[curr_index,:]

                        # Check if current context's dia_id is the same as original's one
                        if row['dia_id'] == current_dialogue_id:
                            # utterance = row['utterance'] # <-- For ENG
                            utterance = row['translated_utterance'] # <-- For RUS

                            if aug_factor > 1:
                                try:
                                    if random.random() < 0.5:
                                        # Randomly choice from presented augmentations 
                                        # in column 'utterance_aug' for current utterance.
                                        utterance = random.choice(row['utterance_aug'])
                                except IndexError:
                                    pass
                                
                            current_context.append(utterance)
                
                current_context = f' {sep_token} '.join(current_context)
                contexts.append(current_context)
                labels.append(curr_row['label'])

        # Return resulting df
        res_df = pd.DataFrame(columns=['text', 'label'])
        res_df['text'] = contexts
        res_df['label'] = labels
        return res_df


    def _setup_datasets(self):
        """ Prepares train and eval datasets """

        dset_dict = {'train': None, 'eval': None}
        for base_name in self.config['datasets']:
            ds_config = OmegaConf.load(f'configs/data/{base_name}.yaml')

            # Iterates through partitions in dataset
            for partition in dset_dict.keys():
                partition_csv_path = os.path.join(
                    ds_config['base_path'], ds_config['partitions'][partition])
                df_dataset = pd.read_csv(partition_csv_path)

                # Preprocess current DataFrame and convert it to dict form
                aug_factor = self.config['task']['dataset']['aug_factor'] if partition == 'train' else 1
                df_dataset = self._preprocess_dataset(df_dataset)
                df_dataset = self._extract_dialogue_contexts(
                    df_dataset  = df_dataset, 
                    ds_name     = f"{base_name}/{partition}", 
                    aug_factor  = aug_factor
                )
                df_data = df_dataset.loc[:,['text', 'label']].to_dict('list')

                # Concatenate to already prepared data
                if dset_dict[partition] is None:
                    dset_dict[partition] = df_data
                else:
                    for k, v in df_data.items():
                        dset_dict[partition][k] += v

        # Convert to batched HF-dataset
        for part in dset_dict.keys():
            dset_dict[part] = Dataset.from_dict(dset_dict[part]) \
                .map(
                    self._tokenize_function, 
                    batched     = True, 
                    batch_size  = self.config['task']['dataset']['batch_size']
                ) \
                .shuffle()
        return dset_dict


    def _setup_task(self) -> Trainer:
        """ 
        Composes previous stages 
        (dataset preprocessing, dialogue contexts extraction, etc.)
        and returns Trainer object.
        """

        print("Datasets preparation.")
        datasets = self._setup_datasets()
        self.le.save()

        training_args = TrainingArguments(**self.config['task']['training_args'])
    
        print("Training.")
        trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset = datasets['train'],
            eval_dataset = datasets['eval'],
            compute_metrics = self._compute_metrics
        )
        return trainer


    def run(self, random_seed: int):
        set_seed(random_seed)
        trainer = self._setup_task()
        trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='configs/tasks/emo.yaml', type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    task = TrainTask(config)
    task.run(451)


if __name__ == '__main__':
    main()