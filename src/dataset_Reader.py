import pandas as pd


def dataset_Reader():
    """
    :return: The Dataset in the csv format
    """
    return pd.read_csv("https://www.neuraldesigner.com/files/datasets/bank_churn.csv", sep=";", index_col=None)
