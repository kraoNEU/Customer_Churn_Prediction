from src import dataset_Reader
from src import feature_Selection

dataFrame_Churn_Prediction = dataset_Reader.dataset_Reader()

X = dataFrame_Churn_Prediction.drop('churn', axis=1)
y = dataFrame_Churn_Prediction["churn"]
print(feature_Selection.feature_Selection(X, y))
