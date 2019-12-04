import pandas as pd


class ReadData:
    def __init__(self, path, file_name):
        self.path = path
        self.file_name = file_name
    """Read the csv file for train"""

    def slit_csv(self):
        df = pd.read_csv(self.path + self.file_name, header=0, low_memory=False)
        df_features = df.iloc[:, 0:24]
        df_targets = df.iloc[:, 24:]
        return df_features, df_targets

    def read_csv(self):
        df = pd.read_csv(self.path + self.file_name, header=0, low_memory=False)

        return df

"""Class for save a DF as pickle file in data"""
class SaveDf:
    def __init__(self, dir_base, df_files, df_name):
        self.dir_base = dir_base
        self.df_files = df_files
        self.df_name = df_name

    def save_dataframe(self):
        data_df = pd.DataFrame(self.df_files)
        return data_df.to_pickle(self.dir_base+'/' + self.df_name + '.pickle')
