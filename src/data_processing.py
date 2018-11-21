from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from src.utils import out
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def process_gradient_boosting(n_estimators, step, kf, X_train, Y_train):
    qualities = []

    for i in range(step, n_estimators, step):
        clf = GradientBoostingClassifier(n_estimators=n_estimators)
        mean_quality = sum(cross_val_score(clf, X_train, Y_train, cv=kf)) / 5
        qualities.append(mean_quality)

        print("Trees quantity: " + str(i))
        print("Quality on cross validation: " + str(mean_quality))

    return qualities

def process_linear_regression(fit_intercept, normalize, X_train, Y_train, kf):
    qualities = []
    clf = LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
    mean_quality = sum(cross_val_score(clf, X_train, Y_train, cv=kf)) / 5 
    print("LR Quality on cross validation: " + str(mean_quality))

def process_rf(n_estimators, max_depth, X_train, Y_train, kf):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
    mean_quality = sum(cross_val_score(clf, X_train, Y_train, cv=kf)) / 5 
    print("RF Quality on cross validation: " + str(mean_quality))

def process_MLP(X_train, Y_train, kf):
    mlp = MLPClassifier(solver='sgd', max_iter=500)
    mean_quality = sum(cross_val_score(mlp, X_train, Y_train, cv=kf)) / 5 
    print("MLP Quality on cross validation: " + str(mean_quality))

def get_best_quality_trees_number(qualities, step):
    max_quality = max(qualities)
    trees_quantity = (qualities.index(max_quality) + 1) * step
    return trees_quantity, max_quality

def process_data(data, X_train, Y_train ,y_name):
    kf = KFold(n_splits=5, shuffle=True)
    
    n_estimators = 100
    gb_trees_step = 10
    gb_qualities = process_gradient_boosting(n_estimators, gb_trees_step, kf, X_train, Y_train)

    trees_quantity, max_quality = get_best_quality_trees_number(gb_qualities, gb_trees_step)
    
    print(" ")
    print("Maximum quality on GB: " + str(max_quality) + "; By trees number: " + str(trees_quantity))

    #process_linear_regression(False, False, X_train, Y_train, kf)

    #process_rf(300, 2, X_train, Y_train, kf)

    #process_MLP(X_train, Y_train, kf)
