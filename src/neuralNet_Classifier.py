from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def neuralNetwork_Classifier(X_train, X_test, y_train, y_test):
    """
    :param X_train: Feature Set for Training
    :param X_test: Feature Set for Testing
    :param y_train: Target Label for Training
    :param y_test: Target Label for Testing
    :return: Returns precision, recall, f1-score, support, True Positives(TP), True Negatives(TN), False Positives(FP),
    False Negatives(FN), accuracy, macro avg, weighted avg
    """

    NN = MLPClassifier()

    # Training the model on the training data and labels
    NN.fit(X_train, y_train)

    # Testing the model i.e. predicting the labels of the test data.
    y_pred = NN.predict(X_test)

    accuracy = accuracy_score(y_pred, y_test) * 100
    print('Neural Network Model accuracy score: {0:0.4f}'.format(accuracy))

    print('Test set score: {:.4f}'.format(NN.score(X_test, y_test)))

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix:\n', cm)
    print('True Positives(TP) = ', cm[0, 0])
    print('True Negatives(TN) = ', cm[1, 1])
    print('False Positives(FP) = ', cm[0, 1])
    print('False Negatives(FN) = ', cm[1, 0])

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                             index=['Predict Positive:1', 'Predict Negative:0'])

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    print(classification_report(y_test, y_pred))

    plt.title("Confusion Matrix for Neural Network")
    plt.savefig("img/neuralnet_confusion_matrix.png")
    plt.show()
