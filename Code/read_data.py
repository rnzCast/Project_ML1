import pandas as pd
from pathlib import Path


class ReadData:
    def __init__(self, file_name):
        self.file_name = file_name
    """Read the csv file for train"""

    def read_csv(self):
        path = (str(Path(__file__).parents[1]) + '/Data/')
        df = pd.read_csv(path + self.file_name, header=0, low_memory=False)
        return df

    def read_pickle(self):
        path = (str(Path(__file__).parents[1]) + '/Data/')
        df = pd.read_pickle(path + self.file_name)
        return df

class SplitData:
    def __init__(self, df):
        self.df = df

    def split_csv(self):
        df_features = self.df.iloc[:, 0:17]
        df_targets = self.df.iloc[:, 17:]
        return df_features, df_targets


class SaveDf:
    def __init__(self, df, name):
        self.df = df
        self.name = name

    def save_df(self):
        dir_base = (str(Path(__file__).parents[1]) + '/Data/')
        return self.df.to_pickle(dir_base + self.name + '.pickle')




