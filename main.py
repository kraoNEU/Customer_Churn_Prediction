import pandas as pd

from src import dataset_Reader
from src import feature_Selection

dataFrame_Churn_Prediction = dataset_Reader.dataset_Reader()

# Getting the features_X of the Dataframe
features_X = dataFrame_Churn_Prediction.drop('churn', axis=1)

# features_X dataframe has got 'string' values which cannot be processed my feature selectors, convert them to identifiers
features_X = pd.get_dummies(features_X)

# target_Label for the Target Label
target_y = dataFrame_Churn_Prediction["churn"]

# target_Label dataframe has got 'string' values which cannot be processed my feature selectors, convert them to identifiers
target_y = pd.get_dummies(target_y)

print(feature_Selection.feature_Selection(features_X, target_y))
