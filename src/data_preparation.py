import pandas as pd
import numpy as np
from src.utils import out
from scipy import stats
from sklearn.preprocessing import StandardScaler

def remove_cheat_props(data, columns_to_drop):
    data = data.drop(columns_to_drop, axis=1)
    return data

def handleEmitions(data):
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

def scaleData(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    X_train = scaler.transform(X_train)
    return X_train

def prepare_data(y_name):
    features = pd.read_csv('./data/features.csv', index_col='match_id')
    features = features.sample(n=round(len(features.index)/6))
    columns_to_drop = ['duration', 'tower_status_radiant', 'tower_status_dire',
                       'barracks_status_radiant',
                       'barracks_status_dire']
    data = remove_cheat_props(features, columns_to_drop)
    columns_with_empty = detect_empty_columns(data)
    out("Empty columns: ", columns_with_empty)
    data = handle_empty(data, columns_with_empty, 0)
    
    Y_train = data[[y_name]].values.ravel()
    X_train = data.drop([y_name], axis=1)
   
    X_train = scaleData(X_train)
    # data = handleEmitions(data)
    return data, X_train, Y_train
