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
    # df = df.sample(n=10000, random_state=1)

    """ Split features and targets in different pickle files"""
    # split_df = rd.SplitData(df)
    # X, targets = split_df.split_csv()

    ###########################################
    # SAVE DATA
    ###########################################

    # """SAVE A TEMPORARY DF FILE"""
    # save_df = rd.SaveDf(df, name='df)
    # save_df.save_df()

    # save_df = rd.SaveDf(X, name='X_complete')
    # save_df.save_df()
    #
    # save_df = rd.SaveDf(targets, name='targets_complete')
    # save_df.save_df()

    ##########################################
    # READ TEMPORARY DATAFRAMES
    ##########################################
    X = rd.ReadData(file_name='X_complete.pickle').read_pickle()
    targets = rd.ReadData(file_name='targets_complete.pickle').read_pickle()

    # print(tabulate(X.head(20), headers=X.columns, tablefmt="grid"))
    # print(tabulate(targets.head(20), headers=targets.columns, tablefmt="grid"))

    ##########################################
    # EDA
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
    #
    ##########################################
    # PREPROCESSING - CLEANING
    ##########################################

    """DROP IRRELEVANT COLUMNS"""
    preprocess = pre.Preprocess(X)
    X = preprocess.drop_columns()
    # print("DROP IRRELEVANT COLUMNS TABLE \n")
    # print(tabulate(X.head(20), headers=X.columns, tablefmt="grid"), '\n')
    #
    """RENAME TARGETS"""
    preprocess = pre.Preprocess(targets)
    targets = preprocess.rename_targets()
    # print("RENAMED TARGETS \n")
    # print(tabulate(targets.head(20), headers=targets.columns, tablefmt="grid"), '\n')
    #
    ##########################################
    # SAVE DATA CLEAN
    ##########################################

    """SAVE A TEMPORARY DF FILE"""
    save_df = rd.SaveDf(X, name='X_complete2')
    save_df.save_df()
    #
    save_df = rd.SaveDf(targets, name='targets_complete2')
    save_df.save_df()
    #
    ##########################################
    # READ CLEAN DATA
    ##########################################
    X2 = rd.ReadData(file_name='X_complete2.pickle').read_pickle()
    targets2 = rd.ReadData(file_name='targets_complete2.pickle').read_pickle()
    # print(tabulate(targets2.head(1000), headers=targets2.columns, tablefmt="grid"), '\n')

    ##########################################
    # PREPROCESSING - IMPUTATION
    ##########################################

    """Check NA Values"""
    impute = pre.DataFrameImputer(X2)
    X2 = impute.fit()
    X2 = impute.transform()

    # print('Check missing values:\n')
    # print(X2.isna().sum(), '\n')

    ##########################################
    # PREPROCESSING - FEATURE SELECTION
    ##########################################

    """ENCODE CATEGORICAL FEATURES"""
    X2 = pre.Preprocess(X2).encode_features()
    # pre.Preprocess(X2).count_feature_values()

    # print(tabulate(X2.head(20), headers=X2.columns, tablefmt="grid"), '\n')

    """ENCODE TARGET"""
    targets2 = pre.Preprocess(targets2).encode_target()
    # pre.Preprocess(targets2).count_feature_values()

    """FEATURE IMPORTANCE"""
    # feature_importance = pre.FeatureImportanceRfc(X2, targets2['curr_acct'])
    # pipe_ft = feature_importance.train_random_forest_classifier()
    # feature_importance.plot_random_forest_classifier(pipe_ft)

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
    # OVERSAMPLING
    ##########################################

    # """DEFINE y"""
    y = targets2['curr_acct']
    print(y.value_counts())
    #
    oversampler = pre.Oversampling(X2, y)
    X2, y = oversampler.oversampler()
    print(pd.DataFrame(data=y, columns=['curr_acct'])['curr_acct'].value_counts())


    ##########################################
    # HYPERPARAMETER TUNING
    ##########################################

    """CLASSIFIER DICTIONARY"""
    clfs = models.classifer_dict()


    """PIPELINE DICTIONARY"""
    pipe_clfs = models.pipeline_dict(clfs)

    """PARAMETER GRIDS"""
    param_grids = models.create_param_grids()

    """HYPERPARAMETER TUNING"""
    hyper_tuning = models.HyperparameterTuning(pipe_clfs, param_grids, X2, y)
    best_score_param_estimators = hyper_tuning.best_parameters_gs()


    # ##########################################
    # # HYPERPARAMETER TUNING
    # ##########################################
    """MODEL SELECTION"""
    models_params = models.ModelSelection(best_score_param_estimators)
    best_score_param_estimators = models_params.select_best()

    """PRINT BEST PARAMETERS FOR ALL MODELS"""
    get_params = models.ModelSelection(best_score_param_estimators)
    get_params.print_models_params()

    """Heatmap"""

main()
