from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from src.utils import out


def process_gradient_boosting(n_estimators, step, kf, X_train, Y_train):
    qualities = []

    for i in range(step, n_estimators, step):
        clf = GradientBoostingClassifier(n_estimators=n_estimators)
        mean_quality = sum(cross_val_score(clf, X_train, Y_train, cv=kf)) / 5
        qualities.append(mean_quality)

        print("Trees quantity: " + str(i))
        print("Quality on cross validation: " + str(mean_quality))

    return qualities


def get_best_quality_trees_number(qualities, step):
    max_quality = max(qualities)
    trees_quantity = (qualities.index(max_quality) + 1) * step
    return trees_quantity, max_quality


def process_data(data, y_name):
    Y_train = data[[y_name]].values.ravel()
    X_train = data.drop([y_name], axis=1)
    kf = KFold(n_splits=5, shuffle=True)

    n_estimators = 100
    gb_trees_step = 10
    gb_qualities = process_gradient_boosting(n_estimators, gb_trees_step, kf, X_train, Y_train)

    trees_quantity, max_quality = get_best_quality_trees_number(gb_qualities, gb_trees_step)

    print(" ")
    print("Maximum quality on GB: " + str(max_quality) + "; By trees number: " + str(trees_quantity))
