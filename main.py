import pandas as pd
from sklearn.model_selection import train_test_split
from src import dataset_Reader
from src import feature_Selection
from src import lightGBM_Classifier
from src import neuralNet_Classifier
from src import svm_Classifier

# Conda ENV has a tendency to be more verbose therefore, igmoring.
import warnings
warnings.filterwarnings("ignore")


# Calling the DataFrame to read and to pass the values
bank_Churn_DataFrame = dataset_Reader.dataset_Reader()

# Getting all the Feature Vectors for the Dataset
X = bank_Churn_DataFrame[['customer_id', 'credit_score', 'age',
                          'tenure', 'balance', 'products_number', 'credit_card', 'active_member',
                          'estimated_salary']]

# Getting the Target Labels
y = bank_Churn_DataFrame['churn']

# features_X dataframe has got 'string' values which cannot be processed my feature selectors, convert them to identifiers
features_X = pd.get_dummies(X)

# target_Label for the Target Label
target_y = bank_Churn_DataFrame["churn"]

# target_Label dataframe has got 'string' values which cannot be processed my feature selectors, convert them to identifiers
target_y = pd.get_dummies(target_y)

print("-------------------------------------------------Feature Selection-------------------------------------------\n")
# Prints out all the Important Features according to the XGBoost Feature Selection
print(feature_Selection.feature_Selection(features_X, target_y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("-------------------------------------------------LightGBM Classifier-----------------------------------------\n")
# Calling the various classifiers:
# Calling the LightGBM Class
lightGBM_Classifier.lightGBM_Classifier(X_train, X_test, y_train, y_test)

print("-------------------------------------------------Neural Net Classifier---------------------------------------\n")
# Calling the Neural Network Class
neuralNet_Classifier.neuralNetwork_Classifier(X_train, X_test, y_train, y_test)

print("-------------------------------------------------SVM Classifier----------------------------------------------\n")
# Calling the SVM Classifier Class
svm_Classifier.svm_Classifier(X_train, X_test, y_train, y_test)
print("-------------------------------------------------------------------------------------------------------------\n")
