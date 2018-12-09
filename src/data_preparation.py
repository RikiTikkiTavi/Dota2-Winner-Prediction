import pandas as pd
import numpy as np
from src.utils import out
from scipy import stats
from sklearn.preprocessing import StandardScaler


def remove_cheat_props(data, columns_to_drop):
    data = data.drop(columns_to_drop, axis=1)
    return data


def handle_emitions(data):
    data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]
    return data


def detect_empty_columns(data):
    values_full = data.count()
    rows_quantity = data.shape[0]
    columns_with_empty = []
    for column, length in values_full.iteritems():
        if rows_quantity - length > 0:
            columns_with_empty.append(column)
    return columns_with_empty


def handle_empty(data, columns_with_empty, value):
    data[columns_with_empty] = data[columns_with_empty].fillna(value=value, inplace=False)
    return data


def scale_data(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    X_train = scaler.transform(X_train)
    return X_train


def prepare_data(y_name):
    # 1) Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше.
    # Удалите признаки, связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).
    features = pd.read_csv('./data/features.csv', index_col='match_id')
    features = features.sample(n=round(len(features.index) / 5))
    columns_to_drop = ['duration', 'tower_status_radiant', 'tower_status_dire',
                       'barracks_status_radiant',
                       'barracks_status_dire']
    data = remove_cheat_props(features, columns_to_drop)
    # ------------------------------------------------------------------------------------------------------------------

    # 2.Проверьте выборку на наличие пропусков с помощью функции count(), которая для каждого столбца показывает
    # число заполненных значений. Много ли пропусков в данных? Запишите названия признаков, имеющих пропуски,
    # и попробуйте для любых двух из них дать обоснование, почему их значения могут быть пропущены.

    columns_with_empty = detect_empty_columns(data)
    out("Empty columns: ", columns_with_empty)

    # Пропущенные признаки:
    # ['first_blood_time', 'first_blood_team', 'first_blood_player1', 'first_blood_player2', 'radiant_bottle_time',
    # 'radiant_courier_time', 'radiant_flying_courier_time', 'radiant_first_ward_time', 'dire_bottle_time',
    # 'dire_courier_time', 'dire_flying_courier_time', 'dire_first_ward_time']

    # Обоснование пропущенных значений:
    # 'first_blood_time', 'first_blood_team' отсутствуют, так как событие "Первая Кровь" не произошло
    # ------------------------------------------------------------------------------------------------------------------

    # 3) Замените пропуски на нули с помощью функции fillna()

    data = handle_empty(data, columns_with_empty, 0)
    # ------------------------------------------------------------------------------------------------------------------

    # 4) Какой столбец содержит целевую переменную? Запишите его название.

    # Название столбца с целевой переменной: radiant_win
    # ------------------------------------------------------------------------------------------------------------------

    y = data[[y_name]].values.ravel()
    X = data.drop([y_name], axis=1).values

    # X_train = scale_data(X_train)
    # data = handleEmitions(data)
    return X, y
