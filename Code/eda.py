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
        return plt.show()

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


    def feature_importance_rfc(self):
        X = self.df.iloc[:, 0:18]
        y = self.df.iloc[:, 18:19]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
        clf.fit(X_train, y_train)
        feat_labels = X.columns
        for feature in zip(feat_labels, clf.feature_importances_):
            print(feature)


    def check_null_values(self):
        return print(((self.df.isnull().sum() / len(self.df)).sort_values())*100)


class CategoricalChecker:

    def __init__(self, df, target, dtype):
        self.df = df
        self.target = target
        self.dtype = dtype

    def categorical_feature_checker(self):
        """
        Parameters
        ----------
        df : dataframe
        target : the target
        dtype : the type of the feature

        Returns
        ----------
        The categorical features and their number of unique value
        """

        feature_number = [[feature, self.df[feature].nunique()]
                          for feature in self.df.columns
                          if feature != self.target and self.df[feature].dtype.name == self.dtype]

        print('%-30s' % 'Categorical feature', 'Number of unique value')
        for feature, number in sorted(feature_number, key=lambda x: x[1]):
            print('%-30s' % feature, number)

        return feature_number


