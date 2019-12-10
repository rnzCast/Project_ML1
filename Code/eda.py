import matplotlib.pyplot as plt
import seaborn as sns


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

    def check_null_values(self):
        print("CHECK NULL VALUES - Percentages\n")
        return print(((self.df.isnull().sum() / len(self.df)).sort_values())*100, '\n')

    def check_na(self):
        print('CHECK NAN VALUES: \n')
        return print(self.df.isna().sum(), '\n')

    def check_features(self):
        for j in range(self.df.shape[1]):
            print(self.df.columns[j] + ':')
            print(self.df.iloc[:, j].value_counts(), end='\n\n')

    def check_target(self):
        print('Unique values and number for target')
        return print(self.df.value_counts())

    def change_dtypes_features(self):
        self.df['pensions_nom'] = self.df['pensions_nom'].map({'1.0': '1', '0.0': '0'})
        self.df['payroll'] = self.df['payroll'].map({'1.0': '1', '0.0': '0'})
        self.df['savings_acct'] = self.df['savings_acct'].astype(object)
        self.df['guarantees'] = self.df['guarantees'].astype(object)
        self.df['curr_acct'] = self.df['curr_acct'].astype(object)
        self.df['derivada_acct'] = self.df['derivada_acct'].astype(object)
        self.df['payroll_acct'] = self.df['payroll_acct'].astype(object)
        self.df['jr_acct'] = self.df['jr_acct'].astype(object)
        self.df['mas_particular_acct'] = self.df['mas_particular_acct'].astype(object)
        self.df['particular_acct'] = self.df['particular_acct'].astype(object)
        self.df['particular_plus_acct'] = self.df['particular_plus_acct'].astype(object)
        self.df['short_term_dep'] = self.df['short_term_dep'].astype(object)
        self.df['med_term_dep'] = self.df['med_term_dep'].astype(object)
        self.df['long_term_dep'] = self.df['long_term_dep'].astype(object)
        self.df['e_acct'] = self.df['e_acct'].astype(object)
        self.df['funds'] = self.df['funds'].astype(object)
        self.df['mortgage'] = self.df['mortgage'].astype(object)
        self.df['pensions_plan'] = self.df['pensions_plan'].astype(object)
        self.df['loans'] = self.df['loans'].astype(object)
        self.df['taxes'] = self.df['taxes'].astype(object)
        self.df['securities'] = self.df['securities'].astype(object)
        self.df['home_acct'] = self.df['home_acct'].astype(object)
        self.df['payroll'] = self.df['payroll'].astype(object)
        self.df['pensions_nom'] = self.df['pensions_nom'].astype(object)
        self.df['direct_debit'] = self.df['direct_debit'].astype(object)
        return self.df

    def count_unique_values(self):
        for j in range(self.df.shape[1]):
            return [print(self.df.columns[j] + ': '),
                    print(self.df.iloc[:, j].value_counts(), end='\n\n')]


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





