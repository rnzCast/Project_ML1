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
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

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
        print('FEATURE COUNTER: \n')
        for col in self.df:
             print(self.df[col].value_counts(), '\n')

    def drop_col_feature_selection(self):
        print(self.df.shape)

        self.df = self.df.drop(columns=[
            'pais_residencia_AL',
            'tipodom',
            'canal_entrada_KGN',
            'age_106',
            'canal_entrada_KAD',
            'pais_residencia_AD',
            'pais_residencia_AE',
            'pais_residencia_SE',
            'pais_residencia_NI',
            'pais_residencia_EE',
            'pais_residencia_BG',
            'pais_residencia_SA',
            'pais_residencia_CA',
            'pais_residencia_GT',
            'pais_residencia_EC',
            'pais_residencia_PR',
            'pais_residencia_CR',
            'pais_residencia_IE',
            # second run of RFC
            'pais_residencia_IL',
            'pais_residencia_IN',
            'pais_residencia_SV',
            'canal_entrada_K00',
            'pais_residencia_PL',
            'pais_residencia_SN',
            'pais_residencia_MR',
            'pais_residencia_MZ',
            'canal_entrada_KGC',
            'pais_residencia_TW',
            # third run of RFC
            'pais_residencia_BE',
            'pais_residencia_CZ',
            'pais_residencia_PY',
            'pais_residencia_GA',
            'canal_entrada_KFV',
            'pais_residencia_HN',
            # fourth run
            'pais_residencia_AR',
            'pais_residencia_ET',
            'pais_residencia_GR',
            'pais_residencia_RU',
            'pais_residencia_UA',
            'canal_entrada_KBG',
            # fifth run
            'pais_residencia_DO'
        ])

        print(self.df.shape)
        return self.df


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

    def train_random_forest_classifier(self):
        pipe_rf = Pipeline([('StandardScaler', StandardScaler()),
                            ('RandomForestClassifier', RandomForestClassifier())])
        pipe_rf = pipe_rf.fit(self.X, self.y)
        return pipe_rf

    def plot_random_forest_classifier(self, pipe_rf):
        # Convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(pipe_rf.named_steps['RandomForestClassifier'].feature_importances_,
                                  self.X.columns)

        # Sort the array in descending order of the importances
        f_importances = f_importances.sort_values(ascending=False)
        print(f_importances)

        # Draw the bar Plot from f_importances
        f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(64, 36), rot=45, fontsize=4)
        # Show the plot
        plt.tight_layout()
        plt.show()


class Chi2:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def chi2(self):
        chi_scores = chi2(self.X, self.y)
        return chi_scores

    def plot_chi2(self, chi_scores):
        p_values = pd.Series(chi_scores[1], index=self.X.columns)
        p_values.sort_values(ascending=False, inplace=True)
        p_values.plot.bar()
        return plt.show()


class DataFrameImputer:

    def __init__(self, df):
        self.df = df
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean of column.
        """

    def fit(self):
        self.fill = pd.Series([self.df[c].value_counts().index[0]
                               if self.df[c].dtype == np.dtype('O') else self.df[c].mean() for c in self.df],
                              index=self.df.columns)

        return self.df

    def transform(self):
        self.df = self.df.fillna(self.fill)
        return self.df


class Oversampling:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def oversampler(self):
        ros = RandomOverSampler(random_state=0)
        self.X, self.y = ros.fit_sample(self.X, self.y)

        return self.X, self.y