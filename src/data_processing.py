import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import numpy as np


def process_gradient_boosting(n_estimators, step, kf, X, y):
    qualities = []

    for i in range(step, n_estimators, step):

        start_time = datetime.datetime.now()
        nest = n_estimators
        clf = GradientBoostingClassifier(n_estimators=nest)
        qualities_n_estimators = []
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = clf.fit(x_train, y_train)
            predictions = clf.predict_proba(x_test)[:, 1]
            qualities_n_estimators.append(roc_auc_score(y_test, predictions))

        mean_quality = np.mean(qualities_n_estimators)
        qualities.append(mean_quality)

        print("Trees quantity: " + str(i))
        print('Time:', datetime.datetime.now() - start_time)
        print("Quality AUC-ROC: " + str(mean_quality))

    return qualities


def get_best_quality_trees_number(qualities, step):
    max_quality = max(qualities)
    trees_quantity = (qualities.index(max_quality) + 1) * step
    return trees_quantity, max_quality


def process_data(X, y):

    # 5)

    # Зафиксируйте генератор разбиений для кросс-валидации по 5 блокам (KFold)

    kf = KFold(n_splits=5, shuffle=True)

    # Оцените качество градиентного бустинга (GradientBoostingClassifier) с помощью данной кросс-валидации,
    # попробуйте при этом разное количество деревьев (как минимум протестируйте следующие значения для количества
    # деревьев: 10, 20, 30).

    n_estimators = 40
    gb_trees_step = 10
    gb_qualities = process_gradient_boosting(n_estimators, gb_trees_step, kf, X, y)
    trees_quantity, max_quality = get_best_quality_trees_number(gb_qualities, gb_trees_step)

    # Как долго проводилась кросс-валидация для градиентного бустинга с 30 деревьями?
    # Ответ: 0:00:57.463733
    # Какое качество при этом получилось?
    # Ответ: Качество по AUC - ROC: 0.6997023016875593

    print("\n Maximum quality on GB: " + str(max_quality) + "; By trees number: " + str(trees_quantity))

    # Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге?
    # Ответ: Имеет смысл, так как увелечение числа деревьев даёт прирост качества по AUC - ROC.

    # Что бы вы предложили делать, чтобы ускорить его обучение при увеличении количества деревьев?
    # Ответ:
    # 1) Уменьшение максимальной глубины деревьев (параметр max_depth)
    # 2) Использование подбвыборки для обучения и кросс-валидации
    # 3) Удаление столбцов, feature_importance которых равняется 0 при текущем колл-ве деревьев.
