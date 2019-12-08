"""
SANTANDER PRODUCT RECOMMENDATION - MACHINE LEARNING I
AUTHORS:
    Adwoa Brako
    Aira Domingo
    Renzo Castagnino
"""
from Code import read_data as rd
from Code import preprocessing as pre
from Code import visualization as vis
from Code import eda
from Code import models
import pandas as pd
import numpy as np
from tabulate import tabulate


def main():

    ##########################################
    # READ DATA
    ##########################################

    # """Read the train file"""
    # read_data = rd.ReadData(file_name='train.csv')
    # df = read_data.read_csv()
    #
    # df_1 = df.loc[df['ind_tjcr_fin_ult1'] == 1]
    # df_1 = df_1.sample(n=250000, random_state=1)
    #
    # df_0 = df.loc[df['ind_tjcr_fin_ult1'] == 0]
    # df_0 = df_0.sample(n=250000, random_state=1)
    #
    # df = df_1.append(df_0)

    ##########################################
    # PREPROCESSING - CLEANING
    ##########################################
    # """DROP IRRELEVANT COLUMNS"""
    # df = pre.Preprocess(df).drop_columns()
    #
    # """RENAME DF"""
    # df = pre.Preprocess(df).rename_targets()
    #
    # """REPLACE NULL WITH NAN"""
    # df = df.replace('', np.nan)
    #
    # """REMOVE ROWS WITH MISSING VALUES"""
    # df = df.dropna(axis=0)
    #
    # """CHANGE DATA TYPES"""
    # df['age'] = df['age'].astype(np.float64)
    # df['ind_nuevo'] = df['ind_nuevo'].astype(object)
    # df['antiguedad'] = df['antiguedad'].astype(np.float64)
    # df['indrel'] = df['indrel'].astype(object)
    # df['indrel_1mes'] = df['indrel_1mes'].map({'1.0': '1', '3.0': '3', '1': '1'})
    # df['cod_prov'] = df['cod_prov'].astype(object)
    # df['ind_actividad_cliente'] = df['ind_actividad_cliente'].astype(object)
    #
    # """ Split features and targets in different pickle files"""
    #
    # # print(tabulate(df.sort_values('renta', ascending=False).head(2000), headers=df.columns, tablefmt="grid"), '\n')
    # df = df[df['renta'] < 1000000]
    # df = df[df['age'] > 10]
    # df = df[df['age'] < 80]
    # df = df[df['antiguedad'] > -800000]


    """CHECK OUTLIERS"""
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.boxplot(x=df['antiguedad'])
    # plt.show()
    #
    #
    # split_df = rd.SplitData(df)
    # X, targets = split_df.split_csv()


    ###########################################
    # SAVE DATA
    ###########################################

    """SAVE A TEMPORARY DF FILE"""
    # save_df = rd.SaveDf(X, name='X_500k')
    # save_df.save_df()
    #
    # save_df = rd.SaveDf(targets, name='targets_500k')
    # save_df.save_df()

    ##########################################
    # READ TEMPORARY DATAFRAMES
    ##########################################
    X = rd.ReadData(file_name='X_500k.pickle').read_pickle()
    targets = rd.ReadData(file_name='targets_500k.pickle').read_pickle()

    ##########################################
    # EDA:  NULL VALUES - NAN - CATEGORICAL CHECK
    ##########################################
    """NULL VALUES IN EACH FEATURE"""
    # print("CHECK NULL VALUES - Percentages\n")
    # eda_process = eda.Eda(X)
    # eda_process.check_null_values()

    """CHECK NAN"""
    # print('CHECK NAN VALUES: \n')
    # eda_process.check_na()

    """CATEGORICAL CHECKER"""
    # cat_check = eda.CategoricalChecker(X, 'object')
    # cat_check.categorical_feature_checker()
    # print()

    #########################################
    # PREPROCESSING - IMPUTATION
    ##########################################

    """Check NA Values"""
    # impute = pre.DataFrameImputer(X_500)
    # X_500 = impute.fit()
    # X_500 = impute.transform()

    # print('Check missing values:\n')
    # print(X_500.isna().sum(), '\n')

    ##########################################
    # PREPROCESSING - FEATURE SELECTION
    ##########################################

    """ENCODE CATEGORICAL FEATURES"""
    X = pre.Preprocess(X).encode_features()
    # pre.Preprocess(X).count_feature_values()
    # print(tabulate(X.head(20), headers=X.columns, tablefmt="grid"), '\n')

    """ENCODE TARGET"""
    targets = pre.Preprocess(targets).encode_target()
    # pre.Preprocess(targets).count_feature_values()

    X = pre.Preprocess(X).drop_col_feature_selection()

    """FEATURE IMPORTANCE"""
    feature_importance = pre.FeatureImportanceRfc(X, targets['credit_card'])
    pipe_ft = feature_importance.train_random_forest_classifier()
    feature_importance.plot_random_forest_classifier(pipe_ft)

    """CORRELATION MAP"""
    eda.Eda(pd.concat([X, targets['credit_card']], axis=1)).cor_map()


    """CHI SQUARED FEATURE SELECTION"""
    # chi_squared = pre.Chi2(X2, targets2['curr_acct'])
    # chi_scores = chi_squared.chi2()
    # print(chi_scores)
    # chi_squared.plot_chi2(chi_scores)

    ##########################################
    # PREPROCESSING - DROP IRRELEVANT FEATURES - TARGET: curr_acct
    ##########################################
    # preprocess = pre.Preprocess(X2)
    # X2 = preprocess.drop_col_feature_selection()


    ##########################################
    # SAVE DATA CLEAN
    ##########################################
    """SAVE A TEMPORARY DF FILE"""
    # rd.SaveDf(X, name='X_500K_clean').save_df()
    # rd.SaveDf(targets, name='targets_500K_clean').save_df()

    ##########################################
    # READ CLEAN DATA
    ##########################################
    # X_500 = rd.ReadData(file_name='X_500K_clean.pickle').read_pickle()
    # targets_500 = rd.ReadData(file_name='targets_500K_clean.pickle').read_pickle()






    ##########################################
    # TRAIN TEST SPLIT
    ##########################################
    # """DEFINE y"""
    # y = targets2['curr_acct']
    # print(y.value_counts())

    # X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3, random_state=0)

    ##########################################
    # OVERSAMPLING
    ##########################################
    # print(y.value_counts())
    #
    # oversampler = pre.Oversampling(X_train, y_train)
    # X_train, y_train = oversampler.oversampler()
    # print(pd.DataFrame(data=y_train, columns=['curr_acct'])['curr_acct'].value_counts())


    ##########################################
    # HYPERPARAMETER TUNING
    ##########################################

    """CLASSIFIER DICTIONARY"""
    # clfs = models.classifer_dict()


    """PIPELINE DICTIONARY"""
    # pipe_clfs = models.pipeline_dict(clfs)

    """PARAMETER GRIDS"""
    # param_grids = models.create_param_grids()

    """HYPERPARAMETER TUNING ONE MODEL"""
    modelname = 'lr'
    # hyper_tuning_one = models.HyperparameterOneModel(pipe_clfs,param_grids,X_train, y_train)
    # best_score_param_estimators = hyper_tuning_one.tune_one_model()

    """HYPERPARAMETER TUNING ALL MODELS"""
    # hyper_tuning = models.HyperparameterTuning(pipe_clfs, param_grids, X_train, y_train)
    # best_score_param_estimators = hyper_tuning.best_parameters_gs()


    # ##########################################
    # # HYPERPARAMETER TUNING
    # ##########################################
    """MODEL SELECTION"""
    # models_params = models.ModelSelection(best_score_param_estimators)
    # best_score_param_estimators = models_params.select_best()

    """PRINT BEST PARAMETERS FOR ALL MODELS"""
    # get_params = models.ModelSelection(best_score_param_estimators)
    # get_params.print_models_params()

    """Heatmap"""

main()
