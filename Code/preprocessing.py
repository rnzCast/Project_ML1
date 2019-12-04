import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


class Preprocess:

    def __init__(self, df):
        self.df = df

    def drop_columns(self):
        df = self.df.drop(['ncodpers',
                           'fecha_dato',
                           'ult_fec_cli_1t',
                           'conyuemp',
                           'fecha_alta',
                           'nomprov'], axis=1)
        return df

