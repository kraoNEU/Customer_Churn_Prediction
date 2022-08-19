import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings("ignore")

bank_Churn_DataFrame = pd.read_csv('../dataset/Bank_Churn_Prediction.csv')

bank_Churn_DataFrame = bank_Churn_DataFrame.drop(['Unnamed: 0'], axis=1)

X = bank_Churn_DataFrame[['customer_id', 'credit_score', 'age',
                          'tenure', 'balance', 'products_number', 'credit_card', 'active_member',
                          'estimated_salary']]
y = bank_Churn_DataFrame['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

svmclassifier_Linear = SVC(kernel='linear')
svmclassifier_Linear.fit(X_train, y_train)
y_pred = svmclassifier_Linear.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
