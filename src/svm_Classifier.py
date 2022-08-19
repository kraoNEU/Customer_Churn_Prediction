from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def svm_Classifier(X_train, X_test, y_train, y_test):
    """
    :param X_train: Feature Set for Training
    :param X_test: Feature Set for Testing
    :param y_train: Target Label for Training
    :param y_test: Target Label for Testing
    :return: Returns precision, recall, f1-score, support, True Positives(TP), True Negatives(TN), False Positives(FP),
    False Negatives(FN), accuracy, macro avg, weighted avg
    """

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    accuracy = accuracy_score(y_pred, y_test) * 100
    print('SVM Model accuracy score: {0:0.4f}'.format(accuracy))

    print(f'Test set score: {0}'.format(svclassifier.score(X_test, y_pred)))

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

    plt.title("Confusion Matrix for SVM")
    plt.savefig("img/svm_confusion_matrix.png")
    plt.show()
