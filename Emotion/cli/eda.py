import os
import argparse
from typing import List
from omegaconf import DictConfig, OmegaConf

import pandas as pd

from utils.task import Task


class EDATask(Task):
    def __init__(self, config: DictConfig):
        pass

    def _plot_label_dist(self, dataframes_list: List[pd.DataFrame], save_path: str):
        pass

    def run(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='configs/tasks/eda.yaml', type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    task = EDATask(config)
    task.run(451)


if __name__ == '__main__':
    main()