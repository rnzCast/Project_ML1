import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


class Eda:

    def __init__(self, df):
        self.df = df

    def cor_map(self):
        plt.figure(figsize=(24, 24))
        cor = self.df.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        return plt

    def cor_map2(self):
        corr = self.df.corr()
        ax = sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        return plt.show()

    def check_null_values(self):
        return print(((self.df.isnull().sum() / len(self.df)).sort_values())*100, '\n')

    def check_na(self):
        return print(self.df.isna().sum(), '\n')

    def check_features(self):
        for j in range(self.df.shape[1]):
            print(self.df.columns[j] + ':')
            print(self.df.iloc[:, j].value_counts(), end='\n\n')

    def check_target(self):
        print('Unique values and number for target')
        return print(self.df.value_counts())


class CategoricalChecker:

    def __init__(self, df, dtype):
        self.df = df
        self.dtype = dtype

    def categorical_feature_checker(self):
        """
        Parameters
        ----------
        df : dataframe
        dtype : the type of the feature

        Returns
        ----------
        The categorical features and their number of unique value
        """

        feature_number = [[feature, self.df[feature].nunique()]
                          for feature in self.df.columns
                          if self.df[feature].dtype.name == self.dtype]

        print('%-30s' % 'CATEGORICAL FEATURES', 'NUMBER OF UNIQUE VALUES')
        for feature, number in sorted(feature_number, key=lambda x: x[1]):
            print('%-30s' % feature, number)

        return feature_number





