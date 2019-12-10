import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt


class Preprocess:

    def __init__(self, df):
        self.df = df

    def drop_columns(self):
        self.df = self.df.drop(['ncodpers',
                           'fecha_dato',
                           'ult_fec_cli_1t',
                           'conyuemp',
                           'fecha_alta',
                           'tipodom',
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
            'ind_recibo_ult1': 'direct_debit',

        })

        return self.df

    def encode_features(self):
        self.df = pd.get_dummies(self.df)
        return self.df

    def encode_target(self):
        le = LabelEncoder()
        self.df = le.fit_transform(self.df)
        return self.df

    def count_feature_values(self):
        """
         Counts how many values are in one feature
        """
        print('FEATURE COUNTER: \n')
        for col in self.df:
             print(self.df[col].value_counts(), '\n')

    def drop_col_feature_selection(self):
        self.df = self.df.drop(columns=[
            'canal_entrada_KFM',
            'canal_entrada_KCF',
            'canal_entrada_KCE',
            'canal_entrada_KBD',
            'canal_entrada_K00',
            'canal_entrada_KEH',
            'canal_entrada_KCP',
            'canal_entrada_KFB',
            'canal_entrada_KDH',
            'canal_entrada_KAU',
            'canal_entrada_KDN',
            'ind_empleado_S',
            'canal_entrada_KDP',
            'canal_entrada_KEU',
            'pais_residencia_DE',
            'canal_entrada_KDB',
            'pais_residencia_IT',
            'pais_residencia_ES',
            'canal_entrada_KCS',
            'canal_entrada_KBN',
            'canal_entrada_KHP',
            'canal_entrada_KDI',
            'canal_entrada_KFV',
            'canal_entrada_KEF',
            'indresi_N',
            'indresi_S',
            'canal_entrada_KFE',
            'canal_entrada_KEQ',
            'canal_entrada_KEI',
            'canal_entrada_KCT',
            'canal_entrada_KBW',
            'canal_entrada_KBY',
            'canal_entrada_KEA',
            'canal_entrada_KEC',
            'guarantees_0',
            'canal_entrada_KDS',
            'canal_entrada_KBV',
            'canal_entrada_KCR',
            'canal_entrada_KCQ',
            'canal_entrada_KEK',
            'canal_entrada_KBX',
            'canal_entrada_KDF',
            'canal_entrada_KDZ',
            'canal_entrada_KDA',
            'indrel_1mes_1',
            'indrel_1mes_3',
            'canal_entrada_KCX',
            'canal_entrada_KCJ',
            'cod_prov_31.0',
            'cod_prov_20.0',
            'canal_entrada_KEV',
            'canal_entrada_KDD',
            'canal_entrada_KDV',
            'canal_entrada_KDX',
            'canal_entrada_KDY',
            'canal_entrada_KCV',
            'canal_entrada_KCO',
            'cod_prov_1.0',
            'tiprel_1mes_P',
            'canal_entrada_KDG',
            'canal_entrada_KGW',
            'canal_entrada_KDC',
            'canal_entrada_KEW',
            'canal_entrada_KBU',
            'canal_entrada_KFH',
            'canal_entrada_KBS',
            'canal_entrada_KEO',
            'indrel_99.0',
            'savings_acct_0',
            'canal_entrada_KCK',
            'canal_entrada_KBJ',
            'canal_entrada_KDM',
            'canal_entrada_KBM',
            'cod_prov_48.0',
            'canal_entrada_KDU',
            'canal_entrada_KFI',
            'canal_entrada_KGY',
            'canal_entrada_KDQ',
            'canal_entrada_KCN',
            'canal_entrada_KCU',
            'canal_entrada_KDT',
            'canal_entrada_004',
            'canal_entrada_KDE',
            'canal_entrada_KBL',
            'canal_entrada_KBB',
            'canal_entrada_KAC',
            'canal_entrada_KEB',
            'canal_entrada_KAV',
            'canal_entrada_KED',
            'canal_entrada_KEM',
            'canal_entrada_KFJ',
            'canal_entrada_KFG',
            'canal_entrada_KGV',
            'canal_entrada_KBF',
            'canal_entrada_KFF',
            'canal_entrada_KCL',
            'canal_entrada_KCD',
            'canal_entrada_KAL',
            'ind_empleado_A',
            'canal_entrada_KEL',
            'ind_empleado_B',
            'canal_entrada_KFU',
            'canal_entrada_KFK',
            'canal_entrada_KAO',
            'canal_entrada_KAI',
            'canal_entrada_KCM',
            'savings_acct_1',
            'canal_entrada_KBR',
            'canal_entrada_KEG',
            'canal_entrada_KFN',
            'canal_entrada_KEZ',
            'canal_entrada_KCA',
            'canal_entrada_KHF',
            'indrel_1.0',
            'canal_entrada_KAN',
            'canal_entrada_KEN',
            'canal_entrada_KDO',
            'guarantees_1',
            'canal_entrada_KBG',
            'canal_entrada_KAK',
            'cod_prov_49.0',
            'canal_entrada_KAH',
            'short_term_dep_0',
            'canal_entrada_KFP',
            'cod_prov_52.0',
            'canal_entrada_KHD',
            'indfall_N',
            'short_term_dep_1',
            'canal_entrada_KGX',
            'canal_entrada_KAD',
            'canal_entrada_KBO',
            'canal_entrada_KAP',
            'derivada_acct_1',
            'canal_entrada_KES',
            'cod_prov_42.0',
            'canal_entrada_KAM',
            'derivada_acct_0',
            'canal_entrada_KCB',
            'canal_entrada_KBH',
            'cod_prov_44.0',
            'cod_prov_51.0',
            'canal_entrada_KEJ',
            'canal_entrada_KCG',
            'canal_entrada_KBQ',
            'ind_empleado_F',
            'canal_entrada_KDR',
            'canal_entrada_KFT',
            'canal_entrada_KHO',
            'canal_entrada_KFS',
            'canal_entrada_KHQ',
            'loans_0',
            'canal_entrada_KAB',
            'canal_entrada_KBZ',
            'canal_entrada_007',
            'cod_prov_27.0',
            'canal_entrada_KAW',
            'canal_entrada_KAE',
            'med_term_dep_1',
            'canal_entrada_KAR',
            'cod_prov_32.0',
            'canal_entrada_KHM',
            'canal_entrada_KAF',
            'med_term_dep_0',
            'canal_entrada_013',
            'cod_prov_25.0',
            'indfall_S',
            'jr_acct_1',
            'canal_entrada_KHN',
            'cod_prov_34.0',
            'cod_prov_5.0',
            'cod_prov_22.0',
            'canal_entrada_KCI',
            'canal_entrada_KAQ',
            'ind_empleado_N',
            'canal_entrada_KHL',
            'canal_entrada_KCH',
            'cod_prov_16.0',
            'canal_entrada_KEY',
            'canal_entrada_KAJ',
            'canal_entrada_KHC',
            'cod_prov_7.0',
            'cod_prov_14.0',
            'mortgage_1',
            'ind_nuevo_0.0',
            'cod_prov_19.0',
            'mortgage_0',
            'canal_entrada_RED',
            'cod_prov_24.0',
            'cod_prov_37.0',
            'cod_prov_43.0',
            'cod_prov_38.0',
            'canal_entrada_KAS',
            'cod_prov_12.0',
            'home_acct_1',
            'cod_prov_23.0',
            'home_acct_0',
            'cod_prov_26.0',
            'cod_prov_9.0',
            'canal_entrada_KCC',
            'cod_prov_21.0',
            'cod_prov_13.0',
            'cod_prov_17.0',
            'cod_prov_4.0',
            'cod_prov_2.0',
            'canal_entrada_KAY',
            'canal_entrada_KFD',
            'cod_prov_10.0',
            'canal_entrada_KAZ',
            'canal_entrada_KAA',
            'cod_prov_40.0',
            'particular_plus_acct_1',
            'securities_0',
            'securities_1',
            'funds_1',
            'funds_0',
            'indext_S',
            'indext_N',
            'cod_prov_29.0',
            'cod_prov_11.0',
            'cod_prov_3.0',
            'cod_prov_35.0',
            'cod_prov_50.0',
            'jr_acct_0',
            'cod_prov_33.0',
            'cod_prov_15.0',
            'pensions_plan_0',
            'pensions_plan_1',
            'cod_prov_47.0',
            'canal_entrada_KHK',
            'cod_prov_18.0',
            'mas_particular_acct_0',
            'cod_prov_36.0',
            'cod_prov_39.0',
            'cod_prov_45.0',
            'cod_prov_30.0',
            'loans_1',
            'mas_particular_acct_1',
            'ind_nuevo_1.0',
            'canal_entrada_KAG',
            'cod_prov_6.0',
        ])
        return self.df

    def remove_outliers(self):
        self.df = self.df[self.df['renta'] < 1000000]
        self.df = self.df[self.df['age'] > 10]
        self.df = self.df[self.df['age'] < 80]
        self.df = self.df[self.df['antiguedad'] > -800000]
        return self.df

    def change_dtypes(self):
        self.df['age'] = self.df['age'].astype(np.float64)
        self.df['ind_nuevo'] = self.df['ind_nuevo'].astype(object)
        self.df['antiguedad'] = self.df['antiguedad'].astype(np.float64)
        self.df['indrel'] = self.df['indrel'].astype(object)
        self.df['indrel_1mes'] = self.df['indrel_1mes'].map({'1.0': '1', '3.0': '3', '1': '1'})
        self.df['cod_prov'] = self.df['cod_prov'].astype(object)
        self.df['ind_actividad_cliente'] = self.df['ind_actividad_cliente'].astype(object)
        return self.df


class CheckOutliers():
    def __init__(self, df, feature):
        self.df = df
        self.feature = feature

    def check_outliers(self):
        sns.boxplot(x=self.df[self.feature])
        plt.show()


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
        f_importances = pd.Series(pipe_rf.named_steps['RandomForestClassifier'].feature_importances_,
                                  self.X.columns)

        f_importances = f_importances.sort_values(ascending=False)
        print(f_importances)

        f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(64, 36), rot=45, fontsize=10)
        plt.tight_layout()
        plt.show()


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