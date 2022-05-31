import os
import argparse

import pandas as pd


def parse_data(in_path: str, out_path: str):
    in_df = pd.read_csv(in_path)
    in_df.columns = in_df.columns.str.lower()

    dirname = os.path.dirname(out_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    in_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=str)
    parser.add_argument('-o', required=True, type=str)
    args = parser.parse_args()

    parse_data(args.i, args.o)
