import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
from joblib import dump, load
import pandas as pd


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


def process_data_gb(X, y):
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


# ----------------------------------------------------------------------------------------------------------------------

def process_lr(kf, X, y, c_list_iterator, model_name, dump_clf):
    qualities = []
    for i, c in enumerate(c_list_iterator):
        start_time = datetime.datetime.now()
        clf = LogisticRegression(penalty='l2', C=c, solver="lbfgs", max_iter=200)
        qualities_c = []
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = clf.fit(x_train, y_train)
            predictions = clf.predict_proba(x_test)[:, 1]
            qualities_c.append(roc_auc_score(y_test, predictions))

        mean_quality = round(np.mean(qualities_c), 5)
        qualities.append(mean_quality)

        if dump_clf:
            dump(clf, "./models/" + model_name + ".joblib")

    print("C: " + str(c))
    print('Time:', datetime.datetime.now() - start_time)
    print("Quality AUC-ROC: " + str(mean_quality))

    return qualities


class CList:

    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        self.c = self.start
        return self

    def __next__(self):
        if self.c <= self.stop:
            x = self.c
            self.c += self.step
            return x
        else:
            raise StopIteration


def get_best_quality_c(qualities, start, step):
    max_q = max(qualities)
    best_c = (qualities.index(max_q) + start) * step
    return max_q, best_c


def process_data_lr(X, y):
    kf = KFold(n_splits=5, shuffle=True)
    start, stop, step = 1, 5, 1
    c_estimated = True  # False, если нужно подобрать заново
    best_c = 3.8
    if c_estimated:
        start, stop, step = best_c, best_c, best_c
    c_list_iterator = iter(CList(start, stop, step))
    # ------------------------------------------------------------------------------------------------------------------

    # 1.
    # Какое качество получилось у логистической регрессии над всеми исходными признаками?
    # Как оно соотносится с качеством градиентного бустинга?
    # Чем вы можете объяснить эту разницу?
    # Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?

    model_name = 'lr'
    qualities = process_lr(kf, X, y, c_list_iterator, model_name, c_estimated)

    max_q, best_c = get_best_quality_c(qualities, start, step)

    print("Max quality: ", max_q)  # Без масштабирования: 0.51602. С масштабированием: 0.71024
    if not c_estimated:
        print("Best C: ", best_c)  # 3.8

    # Ответ:
    # Качество у логистической регрессии над всеми исходными признаками: 0.71024
    # Выше качества градиентного бустинга (0.6997) на 0.01054
    # Логистическая лучше находит зависимости в текущих данных.
    # Логистическая регрессия работает быстрее. (время Бустинг / регрессия: 0:00:57.463733 / 0:00:00.739182)
    # ------------------------------------------------------------------------------------------------------------------

    # 2.
    # 2. Среди признаков в выборке есть категориальные, которые мы использовали как числовые, что вряд ли является
    # хорошей идеей. Категориальных признаков в этой задаче одиннадцать: lobby_type и r1_hero, r2_hero, ..., r5_hero,
    # d1_hero, d2_hero, ..., d5_hero. Уберите их из выборки, и проведите кросс-валидацию для логистической регрессии на
    # новой выборке с подбором лучшего параметра регуляризации.
    # Изменилось ли качество?
    # Чем вы можете это объяснить?

    # Качество: 0.71483 при C = 1.1
    # Разница в качестве незначительна, т.к у признаков маленькие веса => итоговое значение принадлежности к классу не
    # зависит от данных признаков в данной форме.
    # ------------------------------------------------------------------------------------------------------------------

    # 3.
    # Какое получилось качество при добавлении "мешка слов" по героям?
    # Улучшилось ли оно по сравнению с предыдущим вариантом? Чем вы можете это объяснить?

    # Качество: 0.75201 при С = 3.8000000000000003
    # После использование мешка слов качество заметно улучшилось. Это связано с тем, что комманда имеет больший шанс
    # выйграть, если в ней присутствуют определенные герои.


def test_lr(X, data_x):
    clf = load('./models/lr.joblib')
    predictions_df = pd.DataFrame(clf.predict_proba(X))
    predictions_df.columns = ['dire_win', 'radiant_win']
    predictions_df.index = data_x.index
    print("Mean:\n", predictions_df.mean())
    print("Max prob. radiant win: ", max(predictions_df['radiant_win']))  # 0.9966559346546404
    print("Min prob. radiant win: ", min(predictions_df['radiant_win']))  # 0.00956029948527895
    result = predictions_df.drop(['dire_win'], axis=1)
    result.to_csv('./data/result.csv')
