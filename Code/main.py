"""
SANTANDER PRODUCT RECOMMENDATION - MACHINE LEARNING I
AUTHORS:
    Adwoa Brako
    Aira Domingo
    Renzo Castagnino

NOTES:
    - Data can be downloaded from: https://www.kaggle.com/c/santander-product-recommendation/data
    - the csv files needs to be renamed as: test.csv , train.csv

    The  code needs to be run in 3  parts:
    1. run all the code until the part: SAVE DATA 1.

    2. Comment all the code until the step SAVE DATA 1, and run the second part where
    READ TEMPORARY DATAFRAMES until the part SAVE Features DF.

    3. Run only the code from READ 'CLEAN DATA - STEP 3' and bellow.
"""
from Code import read_data as rd
from Code import preprocessing as pre
from Code import predict_rfc as rfc
from Code import eda
from Code import models
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import train_test_split


def main():

    ##########################################
    # READ DATA
    ##########################################

    """Read the data file"""
    # read_data = rd.ReadData(file_name='train.csv')
    # df = read_data.read_csv()

    """Create a sample of 250,000 for positive targets"""
    # df_1 = df.loc[df['ind_tjcr_fin_ult1'] == 1]
    # df_1 = df_1.sample(n=250000, random_state=1)
    #
    """Create a sample of 250,000 for negative targets"""
    # df_0 = df.loc[df['ind_tjcr_fin_ult1'] == 0]
    # df_0 = df_0.sample(n=250000, random_state=1)
    #
    """combined again the df"""
    # df = df_1.append(df_0)

    ##########################################
    # CLEANING DATA
    ##########################################
    """DROP IRRELEVANT COLUMNS"""
    # df = pre.Preprocess(df).drop_columns()
    #
    """RENAME DF"""
    # df = pre.Preprocess(df).rename_targets()
    #
    """REPLACE NULL WITH NAN"""
    # df = df.replace('', np.nan)
    #
    """REMOVE ROWS WITH MISSING VALUES"""
    # df = df.dropna(axis=0)
    #
    """CHANGE DATA TYPES"""
    # df = pre.Preprocess(df).change_dtypes()
    #
    """CHECK OUTLIERS"""
    # pre.CheckOutliers(df, feature='antiguedad').check_outliers()
    #
    """REMOVE OUTLIERS"""
    # df = pre.Preprocess(df).remove_outliers()

    ###########################################
    # SAVE DATA - STEP 1
    ###########################################

    # target = 'credit_card'
    # X = df.drop(columns=['credit_card'])
    # y = df[target]

    """SAVE A TEMPORARY DF FILE"""
    # save_df = rd.SaveDf(X, name='X_cc')
    # save_df.save_df()

    # save_df = rd.SaveDf(y, name='y_cc')
    # save_df.save_df()

    ##########################################
    # READ TEMPORARY DATAFRAMES - STEP 2
    ##########################################
    # X = rd.ReadData(file_name='X_cc.pickle').read_pickle()
    # y = rd.ReadData(file_name='y_cc.pickle').read_pickle()
    # print(tabulate(X.head(10), headers=X.columns, tablefmt="grid"), '\n')

    ##########################################
    # EDA:  NULL VALUES - NAN - CATEGORICAL CHECK
    ##########################################
    """NULL VALUES IN EACH FEATURE"""
    # eda.Eda(X).check_null_values()

    """CHECK NAN"""
    # eda.Eda(X).check_na()

    """CHANGE FEATURES TYPES"""
    # X = eda.Eda(X).change_dtypes_features()

    """CATEGORICAL CHECKER"""
    # eda.CategoricalChecker(X, 'object').categorical_feature_checker()

    """COUNT UNIQUE VALUES PER FEATURE"""
    # eda.Eda(X).count_unique_values()

    ##########################################
    # PREPROCESSING - FEATURE SELECTION
    ##########################################

    """ENCODE CATEGORICAL FEATURES"""
    # X = pre.Preprocess(X).encode_features()
    # pre.Preprocess(X).count_feature_values()

    """DROP IRRELEVANT FEATURES - VALIDATED WITH FEATURE IMPORTANCE RFC"""
    # X = pre.Preprocess(X).drop_col_feature_selection()

    """FEATURE IMPORTANCE"""
    # feature_importance = pre.FeatureImportanceRfc(X, y)
    # pipe_ft = feature_importance.train_random_forest_classifier()
    # feature_importance.plot_random_forest_classifier(pipe_ft)

    """CORRELATION MAP"""
    # eda.Eda(pd.concat([X, y], axis=1)).cor_map()
    # eda.Eda(pd.concat([X, targets['credit_card']], axis=1)).cor_map()

    ##########################################
    # SAVE Features DF
    ##########################################
    """SAVE A TEMPORARY DF FILE"""
    # rd.SaveDf(X, name='X_cc_encoded').save_df()

    ##########################################
    # READ CLEAN DATA - STEP 3
    ##########################################
    X = rd.ReadData(file_name='X_cc_encoded.pickle').read_pickle()
    y = rd.ReadData(file_name='y_cc.pickle').read_pickle()

    ##########################################
    # TRAIN TEST SPLIT
    ##########################################
    """ENCODE TARGET"""
    y = pre.Preprocess(y).encode_target()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    ##########################################
    # OVERSAMPLING
    ##########################################
    oversampler = pre.Oversampling(X_train, y_train)
    X_train, y_train = oversampler.oversampler()

    ##########################################
    # HYPERPARAMETER TUNING
    ##########################################
    """CLASSIFIER DICTIONARY"""
    # clfs = models.classifer_dict()

    """PIPELINE DICTIONARY"""
    # pipe_clfs = models.pipeline_dict(clfs)

    """PARAMETER GRIDS"""
    # param_grids = models.create_param_grids()

    """HYPERPARAMETER TUNING ALL MODELS"""
    # hyper_tuning = models.HyperparameterTuning(pipe_clfs, param_grids, X, y)
    # best_score_param_estimators = hyper_tuning.best_parameters_gs()

    """HYPERPARAMETER TUNING ONE MODEL
    Due to computacional limitations, we used the pipeline to train
    1 model at the time.
    'lr': LogisticRegression
    'mlp': MLPClassifier
    'dt': DecisionTreeClassifier
    'rf': RandomForestClassifier
    'xgb': XGBClassifier
    """
    # modelname = 'rf'
    # hyper_tuning_one = models.HyperparameterOneModel(pipe_clfs, param_grids, X_train, y_train, modelname)
    # best_score_param_estimators = hyper_tuning_one.tune_one_model()

    # # ##########################################
    # MODEL SELECTION
    # # ##########################################
    """MODEL SELECTION"""
    # models_params = models.ModelSelection(best_score_param_estimators)
    # best_score_param_estimators = models_params.select_best()

    """PRINT BEST PARAMETERS FOR ALL MODELS"""
    # get_params = models.ModelSelection(best_score_param_estimators)
    # get_params.print_models_params()

    """ RUN BEST MODEL"""
    rfc.PredictRfc(X_train, y_train, X_test, y_test).predict_rfc()


main()
