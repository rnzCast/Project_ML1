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
        df = self.df.drop(['ncodpers',
                           'fecha_dato',
                           'ult_fec_cli_1t',
                           'conyuemp',
                           'fecha_alta',
                           'nomprov'], axis=1)
        return df

    def rename_cols(self):
        df = self.df.rename(columns ={'ind_ahor_fin_ult1': 'savings_acct', 'ind_aval_fin_ult1': 'guarantees',
                                'ind_cco_fin_ult1':'curr_acct', 'ind_cder_fin_ult1': 'derivada_acct',
                                'ind_cno_fin_ult1': 'payroll_acct', 'ind_ctju_fin_ult1': 'jr_acct',
                                'ind_ctma_fin_ult1': 'mas_particular_acct', 'ind_ctop_fin_ult1': 'particular_acct',
                                'ind_ctpp_fin_ult1': 'particular_plus_acct', 'ind_deco_fin_ult1': 'short_term_dep',
                                'ind_deme_fin_ult1': 'med_term_dep', 'ind_dela_fin_ult1': 'long_term_dep',
                                'ind_ecue_fin_ult1': 'e_acct', 'ind_fond_fin_ult1': 'funds',
                                'ind_hip_fin_ult1': 'mortgage', 'ind_plan_fin_ult1': 'pensions',
                                'ind_pres_fin_ult1': 'loans', 'ind_reca_fin_ult1': 'taxes',
                                'ind_tjcr_fin_ult1': 'credit_card', 'ind_valo_fin_ult1': 'securities',
                                'ind_viv_fin_ult1': 'home_acct', 'ind_nomina_ult1': 'payroll',
                                'ind_nom_pens_ult1': 'pensions', 'ind_recibo_ult1': 'direct_debit'})

        return df



    def encode_features(self):
        df = pd.get_dummies(self.df)
        return df

    def encode_target(self):
        le = LabelEncoder()
        y = le.fit_transform(self.df)

        return y



class FS_chi2:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def chi2(self):

        # Two features with highest chi-squared statistics are selected
        chi2_features = SelectKBest(chi2, k=2)
        X_kbest_features = chi2_features.fit_transform(self.X, self.y)
        return X_kbest_features
