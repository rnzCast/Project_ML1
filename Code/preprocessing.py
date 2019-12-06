import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


class Preprocess:

    def __init__(self, df):
        self.df = df

    def drop_columns(self):
        self.df = self.df.drop(['ncodpers',
                           'fecha_dato',
                           'ult_fec_cli_1t',
                           'conyuemp',
                           'fecha_alta',
                           'nomprov'], axis=1)
        return self.df

    def rename_targets(self):
        self.df = self.df.rename(columns={
            'ind_ahor_fin_ult1': 'savings_acct',
            'ind_aval_fin_ult1': 'guarantees',
            'ind_cco_fin_ult1': 'curr_acct',
            'ind_cder_fin_ult1': 'derivada_acct',
            'ind_cno_fin_ult1': 'payroll_acct',
            'ind_ctju_fin_ult1': 'jr_acct',
            'ind_ctma_fin_ult1': 'mas_particular_acct',
            'ind_ctop_fin_ult1': 'particular_acct',
            'ind_ctpp_fin_ult1': 'particular_plus_acct',
            'ind_deco_fin_ult1': 'short_term_dep',
            'ind_deme_fin_ult1': 'med_term_dep',
            'ind_dela_fin_ult1': 'long_term_dep',
            'ind_ecue_fin_ult1': 'e_acct',
            'ind_fond_fin_ult1': 'funds',
            'ind_hip_fin_ult1': 'mortgage',
            'ind_plan_fin_ult1': 'pensions_plan',
            'ind_pres_fin_ult1': 'loans',
            'ind_reca_fin_ult1': 'taxes',
            'ind_tjcr_fin_ult1': 'credit_card',
            'ind_valo_fin_ult1': 'securities',
            'ind_viv_fin_ult1': 'home_acct',
            'ind_nomina_ult1': 'payroll',
            'ind_nom_pens_ult1': 'pensions_nom',
            'ind_recibo_ult1': 'direct_debit'
        })

        return self.df

    def encode_features(self):
        self.df = pd.get_dummies(self.df)
        return self.df

    def encode_target(self):
        le = LabelEncoder()
        self.df = self.df.apply(le.fit_transform)
        return self.df

    def count_feature_values(self):
        """
         Counts how many values are in one feature
        """
        for col in self.df:
             print('FEATURE COUNTER: \n')
             print(self.df[col].value_counts(), '\n')


class FeatureImportanceRfc:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def feature_importance(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.5, random_state=0)
        clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
        clf.fit(X_train, y_train)
        feat_labels = self.X.columns
        for feature in zip(feat_labels, clf.feature_importances_):
            print(feature)

# class FeatureImportanceRfc:
#
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y
#
#     def feature_importance(self):
#         rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
#         rnd_clf.fit(self.X, self.y)
#         print(rnd_clf.feature_importances_)
        # for name, importance in zip(self.X, rnd_clf.feature_importances_):
        #     print(name, "=", importance)

    # def print_feature_importance(self):
    # features = iris['feature_names']
    # importances = rnd_clf.feature_importances_
    # indices = np.argsort(importances)
    #
    # plt.title('Feature Importances')
    # plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    # plt.yticks(range(len(indices)), [features[i] for i in indices])
    # plt.xlabel('Relative Importance')
    # plt.show()

class FsChi2:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def chi2(self):
        # Two features with highest chi-squared statistics are selected
        chi2_features = SelectKBest(chi2, k=2)
        X_kbest_features = chi2_features.fit_transform(self.X, self.y)
        return X_kbest_features
