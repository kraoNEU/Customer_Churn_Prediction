import pandas as pd


def dataset_Reader():
    """
    :return: The Dataset in the csv format
    """

    df = pd.read_csv("https://www.nteuraldesigner.com/files/datasets/bank_churn.csv", sep=";", index_col=None)
    df.to_csv("dataset/Bank_Churn_Prediction.csv")
    return df
