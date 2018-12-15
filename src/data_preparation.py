import pandas as pd
import numpy as np
from src.utils import out
from scipy import stats
from sklearn.preprocessing import StandardScaler


def remove_cheat_props(data, columns_to_drop):
    data = data.drop(columns_to_drop, axis=1)
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


def data_initial_prep(divider):
    features = pd.read_csv('./data/features.csv', index_col='match_id')
    features = features.sample(n=round(len(features.index) / divider))
    columns_to_drop = ['duration', 'tower_status_radiant', 'tower_status_dire',
                       'barracks_status_radiant',
                       'barracks_status_dire']
    return remove_cheat_props(features, columns_to_drop)


def prepare_data_gb(y_name):
    # 1) Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше.
    # Удалите признаки, связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).
    data = data_initial_prep(6)
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

    return X, y


# ----------------------------------------------------------------------------------------------------------------------

def remove_categorial(data):
    data = data.drop(['lobby_type'], axis=1)
    for i in range(1, 6):
        data = data.drop(["r" + str(i) + "_hero"], axis=1)
        data = data.drop(["d" + str(i) + "_hero"], axis=1)
    return data


def count_unic_hero_id(data):
    hero_columns = []
    for i in range(1, 6):
        hero_columns.append("r" + str(i) + "_hero")
        hero_columns.append("d" + str(i) + "_hero")
    return len(pd.unique(data[hero_columns].values.ravel()))


def process_categorial(heroes_quantity, x_data, X):
    X_pick = np.zeros((x_data.shape[0], heroes_quantity + 4))

    for i, match_id in enumerate(x_data.index):
        for p in range(5):
            X_pick[i, x_data.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            X_pick[i, x_data.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    return np.hstack((X, X_pick))


def prepare_data_lr(y_name):
    data = data_initial_prep(1)
    columns_with_empty = detect_empty_columns(data)
    data = handle_empty(data, columns_with_empty, 0)
    x_data = data.drop([y_name], axis=1)
    # ------------------------------------------------------------------------------------------------------------------

    # 3. Сколько различных идентификаторов героев существует в данной игре
    heroes_quantity = count_unic_hero_id(data)
    print("Unical heroes id quantity", heroes_quantity)  # 108
    # ------------------------------------------------------------------------------------------------------------------

    # 2. Среди признаков в выборке есть категориальные, которые мы использовали как числовые, что вряд ли является
    # хорошей идеей. Категориальных признаков в этой задаче одиннадцать: lobby_type и r1_hero, r2_hero, ..., r5_hero,
    # d1_hero, d2_hero, ..., d5_hero. Уберите их из выборки, и проведите кросс-валидацию для логистической регрессии
    # на новой выборке с подбором лучшего параметра регуляризации.
    # data = remove_categorial(data)
    # ------------------------------------------------------------------------------------------------------------------

    y = data[[y_name]].values.ravel()
    X = x_data.values
    # ------------------------------------------------------------------------------------------------------------------

    # Scale data
    X = scale_data(X)
    # ------------------------------------------------------------------------------------------------------------------

    # 4.
    # Воспользуемся подходом "мешок слов" для кодирования информации о героях.
    # Пусть всего в игре имеет N различных героев. Сформируем N признаков, при этом i-й будет равен нулю, если i-й
    # герой не участвовал в матче; единице, если i-й герой играл за команду Radiant; минус единице, если i-й герой
    # играл за команду Dire. Ниже вы можете найти код, который выполняет данной преобразование. Добавьте полученные
    # признаки к числовым, которые вы использовали во втором пункте данного этапа.
    X = process_categorial(heroes_quantity, x_data, X)
    # ------------------------------------------------------------------------------------------------------------------

    return X, y
