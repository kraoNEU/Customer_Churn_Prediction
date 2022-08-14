from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


def feature_Selection(X, y):
    """
    :param X: Dataset Features
    :param y: Dataset Target Label
    :return: Returns Selected Random Forest Selected Features of Importance
    """
    rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100))
    rf_selector.fit(X, y)
    rf_support = rf_selector.get_support()
    rf_feature = X.loc[:, rf_support].columns.tolist()
    print(str(len(rf_feature)), 'selected features')
    return rf_feature
