"""
SANTANDER PRODUCT RECOMMENDATION
MACHINE LEARNING I
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
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate


dir_base = (str(Path(__file__).parents[1]) + '/Data/')


def main():

    ##########################################
    # READ DATA
    ##########################################

    # """Read the train file"""
    # file_name = 'train.csv'
    # read_data = rd.ReadData(dir_base, file_name)
    # df = read_data.read_csv()
    # X, targets = read_data.split_csv()
    #
    # """Temporary gets Df with 200,000 rows"""
    # X = X.iloc[0:200000]
    # targets = targets.iloc[0:200000]
    # df = df.iloc[0:200000]

    ###########################################
    # SAVE DATA
    ###########################################

    # """SAVE A TEMPORARY DF FILE"""
    # name = 'df'
    # save_df = rd.SaveDf(dir_base, df, name)
    # save_df.save_dataframe()
    #
    # name = 'X'
    # save_df = rd.SaveDf(dir_base, X, name)
    # save_df.save_dataframe()
    #
    # name = 'targets'
    # save_df = rd.SaveDf(dir_base, targets, name)
    # save_df.save_dataframe()

    ##########################################
    # READ TEMPORARY DATAFRAMES
    ##########################################
    # X = pd.read_pickle(dir_base+"/X.pickle")
    # targets = pd.read_pickle(dir_base+"/targets.pickle")

    # print(tabulate(X.head(20), headers=X.columns, tablefmt="grid"))
    # print(tabulate(targets.head(20), headers=targets.columns, tablefmt="grid"))

    ##########################################
    # EDA
    ##########################################

    # """NULL VALUES IN EACH FEATURE"""
    # print("CHECK NULL VALUES - Percentages\n")
    # eda_process = eda.Eda(X)
    # eda_process.check_null_values()
    #
    # """CHECK NAN"""
    # print('CHECK NAN VALUES: \n')
    # eda_process.check_na()
    #
    # """CATEGORICAL CHECKER"""
    # cat_check = eda.CategoricalChecker(X, 'object')
    # cat_check.categorical_feature_checker()
    # print()

    ##########################################
    # PREPROCESSING - CLEANING
    ##########################################

    # """DROP IRRELEVANT COLUMNS"""
    # preprocess = pre.Preprocess(X)
    # X = preprocess.drop_columns()
    # print("DROP IRRELEVANT COLUMNS TABLE \n")
    # print(tabulate(X.head(20), headers=X.columns, tablefmt="grid"), '\n')
    #
    """RENAME TARGETS"""
    # preprocess = pre.Preprocess(targets)
    # targets = preprocess.rename_targets()
    # print("RENAMED TARGETS \n")
    # print(tabulate(targets.head(20), headers=targets.columns, tablefmt="grid"), '\n')

    ##########################################
    # SAVE DATA CLEAN
    ##########################################

    # """SAVE A TEMPORARY DF FILE"""
    # name = 'X2'
    # save_df = rd.SaveDf(dir_base, X, name)
    # save_df.save_dataframe()

    # name = 'targets2'
    # save_df = rd.SaveDf(dir_base, targets, name)
    # save_df.save_dataframe()

    ##########################################
    # READ CLEAN DATA
    ##########################################
    X2 = pd.read_pickle(dir_base+"/X2.pickle")
    targets2 = pd.read_pickle(dir_base+"/targets2.pickle")
    # print(tabulate(targets2.head(1000), headers=targets2.columns, tablefmt="grid"), '\n')

    ##########################################
    # PREPROCESSING - FEATURE SELECTION
    ##########################################

    """ENCODE CATEGORICAL FEATURES"""
    encode_x = pre.Preprocess(X2)
    X2 = encode_x.encode_features()
    # encode_x.count_feature_values()
    print(tabulate(X2.head(20), headers=X2.columns, tablefmt="grid"), '\n')

    """ENCODE TARGET"""
    encode_y = pre.Preprocess(targets2)
    targets2 = encode_y.encode_target()
    # encode_y.count_feature_values()

    """FEATURE IMPORTANCE"""
    # feature_importance = pre.FeatureImportanceRfc(X2, targets2['savings_acct'])
    # feature_importance.feature_importance()

    """CHI SQUARED FEATURE SELECTION"""
    # chi = pre.FsChi2(X, y)
    # X_kbest = chi.chi2()
    # print('Original feature number:', X.shape[1])
    # print('Reduced feature number:', X_kbest.shape[1])


    """CLASSIFIER DICTIONARY"""
    clfs = models.classifer_dict()

    """PIPELINE DICTIONARY"""
    pipe_clfs = models.pipeline_dict(clfs)

    """PARAMETER GRIDS"""
    param_grids = models.create_param_grids()

    """HYPERPARAMETER TUNING"""
    hyper_tuning = models.HyperparameterTuning(pipe_clfs, param_grids, X, y)
    best_score_param_estimators = hyper_tuning.best_parameters_gs()

    """MODEL SELECTION"""
    models_params = models.ModelSelection(best_score_param_estimators)
    best_score_param_estimators = models_params.select_best()

    """PRINT BEST PARAMETERS FOR ALL MODELS"""
    get_params = models.ModelSelection(best_score_param_estimators)
    get_params.print_models_params()

    """Heatmap"""

main()
