import pandas as pd


def out(h, v):
    print(h + ":")
    print(v)
    print("----")
    print(" ")


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


def prepare_data():
    features = pd.read_csv('./data/features.csv', index_col='match_id')
    columns_to_drop = ['duration', 'tower_status_radiant', 'tower_status_dire',
                       'barracks_status_radiant',
                       'barracks_status_dire']
    data = remove_cheat_props(features, columns_to_drop)
    columns_with_empty = detect_empty_columns(data)
    out("Empty columns", columns_with_empty)
    data = handle_empty(data, columns_with_empty, 0)

    return data
