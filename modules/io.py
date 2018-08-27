import csv
import pandas as pd


class IO:

    def read_tsv(path):
        with open(path, 'r') as tsv:
            content = [x for x in csv.reader(tsv, delimiter='\t')]
        return content

    def read_pkl(path):
        return pd.read_pickle(path)
