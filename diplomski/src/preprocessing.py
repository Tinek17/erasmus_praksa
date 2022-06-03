import numpy
import pandas
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from src import *


def prep_data(var_name, site_name, PM, rest_data):
    temp_y = PM[var_name]
    temp_y.name = var_name
    temp_temporal = rest_data[temporal]
    temp_x = rest_data.filter(like=site_name, axis=1)
    temp_x = pd.concat([temp_x, temp_temporal], axis=1)
    return temp_x, temp_y


def prep_data_satelite(temp_x, site, data):
    weather_data = temp_x.filter(like=site, axis=1)
    temp_x = temp_x.drop(columns=weather_data.columns, axis=1)
    temp_x = pd.concat([temp_x, data], axis=1)
    return temp_x


def split_data(x_data, y_data):
    train_x = x_data['2018-01-01':'2020-01-02']
    test_x = x_data['2020-01-03':'2020-03-15']
    train_y = y_data['2018-01-01':'2020-01-02']
    test_y = y_data['2020-01-03':'2020-03-15']
    return train_x, test_x, train_y, test_y


def traffic_for_station(site_temp, data_temp):
    traffic_temp = data_temp.filter(like=site_temp, axis=1)
    return traffic_temp


def split_data_nn_of_rf(X_train_temp, X_test_temp, key):
    if key == 'nn':
        sc = StandardScaler()
        X_train_temp_scaled = sc.fit_transform(X_train_temp)
        X_train_df = pd.DataFrame(X_train_temp_scaled, columns=X_train_temp.columns)
        X_test_temp_scaled = sc.transform(X_test_temp)
        X_test_df = pd.DataFrame(X_test_temp_scaled, columns=X_test_temp.columns)
        return X_train_df, X_test_df
    else:
        return X_train_temp,X_test_temp
