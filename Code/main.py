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
print(dir_base)


def main():

    """Read the train file"""
    # file_name = 'train.csv'
    # read_data = rd.ReadData(dir_base, file_name)
    # df = read_data.read_csv()
    # df_features, df_targets = read_data.slit_csv()
    #
    """Temporary gets Df with 200,000 rows"""
    # temp_features = df_features.iloc[0:200000]
    # temp_target = df_targets.iloc[0:200000]
    # temp_df = df.iloc[0:200000]
    #
    #
    """SAVE A TEMPORARY DF FILE"""
    # name = 'temp_df'
    # save_df = rd.SaveDf(dir_base, temp_df, name)
    # save_df.save_dataframe()
    #
    # name = 'temp_features'
    # save_df = rd.SaveDf(dir_base, temp_features, name)
    # save_df.save_dataframe()
    #
    # name = 'temp_targets'
    # save_df = rd.SaveDf(dir_base, temp_target, name)
    # save_df.save_dataframe()


    """READ THE TEMPORARY DF"""
    df_temp = pd.read_pickle(dir_base+"/temp_df.pickle")
    # df_features = pd.read_pickle(dir_base+"/temp_features.pickle")
    # df_targets = pd.read_pickle(dir_base+"/temp_targets.pickle")

    # print(tabulate(df_features.head(20), headers=df_features.columns, tablefmt="grid"))
    # print(tabulate(df_targets.head(20), headers=df_targets.columns, tablefmt="grid"))
    # prep = pre.Preprocess(df_temp)
    # df_temp = prep.drop_columns()


    """CORRELATION MAP (HEAT)"""
    # target = df_temp.iloc[:, 0:19]
    # print(target['tipodom'].value_counts())
    # run_eda = eda.Eda(target)
    # run_eda.cor_map()


    """FEATURE IMPORTANCE"""
    # run_eda.feature_importance_rfc()

    """CATEGORICAL CHECKER"""
    # cat_check = eda.CategoricalChecker(df_temp, 'ind_ahor_fin_ult1', 'object')
    # cat_check.categorical_feature_checker()
    # print()


    """NULL VALUES IN EACH FEATURE"""
    # print("CHECK NULL VALUES - %")
    # null = eda.Eda(df_temp)
    # null.check_null_values()


    """CHECK NAN"""
    # print('CHECK NAN')
    # nan = eda.Eda(df_temp)
    # nan.check_na()


    """DROP IRRELEVANT COLUMNS"""
    preprocess = pre.Preprocess(df_temp)
    df_temp = preprocess.drop_columns()
    # print(tabulate(df_temp.head(20), headers=df_temp.columns, tablefmt="grid"))


    """RENAME COLUMNS"""
    preprocess = pre.Preprocess(df_temp)
    df_temp = preprocess.rename_cols()


    """DEFINE TARGET"""
    target = 'curr_acct'


    """DEFINE X & Y"""
    X = df_temp.iloc[:, :18]
    y = df_temp[target]


    """CHECK CATEGORICAL FEATURES"""
    # feature_check = eda.Eda(X)
    # feature_check.check_features()

    """CHECK CATEGORICAL TARGET"""
    target_check = eda.Eda(y)
    target_check.check_target()


    """CHANGE DTYPES"""
    # X = X.astype({'age': 'float'}).dtypes
    # X = X.astype({'antiguedad': 'float'}).dtypes


    """CHI SQUARED FEATURE SELECTION"""
    chi = pre.FS_chi2(X, y)
    X_kbest = chi.chi2()
    print('Original feature number:', X.shape[1])
    print('Reduced feature number:', X_kbest.shape[1])




    """ENCODE CATEGORICAL FEATURES"""
    # encode = pre.Preprocess(X)
    # X = encode.encode_features()
    # print(tabulate(df_temp.head(20), headers=df_temp.columns, tablefmt="grid"))
    # print(X.dtypes)


    """ENCODE TARGET"""
    # encode = pre.Preprocess(y)
    # y = encode.encode_target()
    # pd.DataFrame(data=y, columns=[target])[target].value_counts()


    """CLASSIFIER DICTIONARY"""
    # clfs = models.classifer_dict()


    """PIPELINE DICTIONARY"""
    # pipe_clfs = models.pipeline_dict()


main()
