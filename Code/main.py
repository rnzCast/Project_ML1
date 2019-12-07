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

    """Read the train file"""
    # read_data = rd.ReadData(file_name='train.csv')
    # df = read_data.read_csv()
    # df = df.sample(n=500000, random_state=1)

    """ Split features and targets in different pickle files"""
    # split_df = rd.SplitData(df)
    # X, targets = split_df.split_csv()

    ###########################################
    # SAVE DATA
    ###########################################

    # """SAVE A TEMPORARY DF FILE"""
    # save_df = rd.SaveDf(df, name='df')
    # save_df.save_df()

    # save_df = rd.SaveDf(X, name='X_complete')
    # save_df.save_df()
    #
    # save_df = rd.SaveDf(targets, name='targets_complete')
    # save_df.save_df()

    ##########################################
    # READ TEMPORARY DATAFRAMES
    ##########################################
    X = rd.ReadData(file_name='X_500k.pickle').read_pickle()
    targets = rd.ReadData(file_name='targets_500k.pickle').read_pickle()

    # print(tabulate(X.head(20), headers=X.columns, tablefmt="grid"))
    # print(tabulate(targets.head(20), headers=targets.columns, tablefmt="grid"))

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

    ##########################################
    # PREPROCESSING - CLEANING
    ##########################################
    """DROP IRRELEVANT COLUMNS"""
    # X = pre.Preprocess(X).drop_columns()
    # print(tabulate(X.head(20), headers=X.columns, tablefmt="grid"), '\n')

    """RENAME TARGETS"""
    # targets = pre.Preprocess(targets).rename_targets()
    # print(tabulate(targets.head(20), headers=targets.columns, tablefmt="grid"), '\n')

    ##########################################
    # SAVE DATA CLEAN
    ##########################################
    """SAVE A TEMPORARY DF FILE"""
    # rd.SaveDf(X, name='X_500K_clean').save_df()
    # rd.SaveDf(targets, name='targets_500K_clean').save_df()

    ##########################################
    # READ CLEAN DATA
    ##########################################
    X_500 = rd.ReadData(file_name='X_500K_clean.pickle').read_pickle()
    targets_500 = rd.ReadData(file_name='targets_500K_clean.pickle').read_pickle()

    ##########################################
    # PREPROCESSING - IMPUTATION
    ##########################################

    """Check NA Values"""
    impute = pre.DataFrameImputer(X_500)
    X_500 = impute.fit()
    X_500 = impute.transform()

    print('Check missing values:\n')
    print(X_500.isna().sum(), '\n')

    ##########################################
    # PREPROCESSING - FEATURE SELECTION
    ##########################################

    """ENCODE CATEGORICAL FEATURES"""
    X_500 = pre.Preprocess(X_500).encode_features()
    # pre.Preprocess(X_500).count_feature_values()
    # print(tabulate(X2.head(20), headers=X2.columns, tablefmt="grid"), '\n')

    """ENCODE TARGET"""
    targets_500 = pre.Preprocess(targets_500).encode_target()
    # pre.Preprocess(targets2).count_feature_values()

    """FEATURE IMPORTANCE"""
    # feature_importance = pre.FeatureImportanceRfc(X_500, targets_500['curr_acct'])
    # pipe_ft = feature_importance.train_random_forest_classifier()
    # feature_importance.plot_random_forest_classifier(pipe_ft)

    """CORRELATION MAP"""
    eda.Eda(pd.concat([X_500, targets_500['curr_acct']], axis=1)).cor_map()


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
