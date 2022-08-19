import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

bank_Churn_DataFrame = pd.read_csv('../dataset/Bank_Churn_Prediction.csv')

bank_Churn_DataFrame = bank_Churn_DataFrame.drop(['Unnamed: 0'], axis=1)

X = bank_Churn_DataFrame[['customer_id', 'credit_score', 'age',
                          'tenure', 'balance', 'products_number', 'credit_card', 'active_member',
                          'estimated_salary']]
y = bank_Churn_DataFrame['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

NN = MLPClassifier()

# Step 5
# Training the model on the training data and labels
NN.fit(X_train, y_train)

# Step 6
# Testing the model i.e. predicting the labels of the test data.
y_pred = NN.predict(X_test)

# Step 7
# Evaluating the results of the model
accuracy = accuracy_score(y_test, y_pred) * 100
confusion_mat = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', confusion_mat)
print('\nTrue Positives(TP) = ', confusion_mat[0, 0])
print('\nTrue Negatives(TN) = ', confusion_mat[1, 1])
print('\nFalse Positives(FP) = ', confusion_mat[0, 1])
print('\nFalse Negatives(FN) = ', confusion_mat[1, 0])


# Step 8
# Printing the Results
print("Accuracy for Neural Network is:", accuracy)
print("Confusion Matrix")
print(confusion_mat)
