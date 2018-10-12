from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier


# TODO: Estimators loop
def process_data(data, y_name):
    Y_train = data[[y_name]].values.ravel()
    X_train = data.drop([y_name], axis=1)

    kf = KFold(n_splits=5, shuffle=True)
    clf = GradientBoostingClassifier()

    print cross_val_score(clf, X_train, Y_train, cv=kf)

