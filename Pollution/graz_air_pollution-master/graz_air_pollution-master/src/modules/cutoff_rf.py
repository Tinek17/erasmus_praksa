import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime

AREAS = ['D_', 'N_', 'O_', 'S_', 'W_']  # station abbrevation
add_names = {  # station full name
    'D_': 'DonBosco_',
    'N_': 'Nord_',
    'O_': 'Ost_',
    'W_': 'West_',
    'S_': 'Sud_'}

col_names = [['DonBosco_', 'D_'], ['Nord_', 'N_'], ['Ost_', 'O_'], ['Sud_', 'S_'], ['West_', 'W_']]

factors = ['NO2', 'PM10K', 'O3']  # pollutants

# weather features
features = ['Temp', 'RH', 'Pressure', 'Winddirection', 'Windspeed', 'Precip']
# weekday features
features2 = ['weekday_Friday', 'weekday_Monday', 'weekday_Saturday', 'weekday_Sunday', 'weekday_Thursday',
             'weekday_Tuesday', 'weekday_Wednesday']

# season features
features3 = ['season_fall', 'season_spring', 'season_summer', 'season_winter']

holidays = ['holiday', 'holiday_school', 'holiday_sunday']


def holiday_sunday(df_o):
    """
    Returns new vector 'hol_sun' where value 1 represents if holiday falls on Sunday, and value 0 if not
    :param df_o: dataframe with columns 'holiday' and 'weekday_Sunday'
    :return: vector 'hol_sun' with values 1 or 0
    """
    mask_ind = []
    lista = [element for tupl in np.where(df_o['holiday'] == 1) for element in tupl]
    for i in lista:
        hol_ind = df_o.iloc[i].name
        if df_o.loc[hol_ind]['weekday_Sunday'] == 1:
            mask_ind.append(i)

    hol_sun = [0] * len(df_o.index)
    df_hol_sun = pd.DataFrame(index=df_o.index, data=hol_sun).rename(columns={0: 'hol_sun'})
    for i in mask_ind:
        df_hol_sun.iloc[i]['hol_sun'] = 1
    return df_hol_sun['hol_sun']


def month(df_o):
    """
    Returns new vector 'month' with values 1 to 12 (1=January, 12=December, etc.)
    :param df_o: dataframe with column 'month_Jan', 'month_Feb' etc.
    :return: vector 'month' with values 1 to 12
    """
    months_convert = {
        'month_Apr': 4,
        'month_Aug': 8,
        'month_Dec': 12,
        'month_Feb': 2,
        'month_Jan': 1,
        'month_Jul': 7,
        'month_Jun': 6,
        'month_Mar': 3,
        'month_May': 5,
        'month_Nov': 11,
        'month_Oct': 10,
        'month_Sep': 9}
    for c in months_convert.keys():
        df_o[c] = [months_convert[c] if i else i for i in df_o[c]]
        df_o['month'] = df_o[months_convert.keys()].sum(axis=1)
    return df_o['month']


def cut_off(df, percent):
    """
    Calculates quantile of each column based on a defined percent, and all values above calculated quantile fills with
    quantile in order to exclude outliers.
    :param df: df with datetime index
    :param percent: percent of values in column are lower than calculated quantile
    :return: df with outliers replaced by quantiles
    """
    df = df.copy()
    cols = [k for k in [i + j for j in factors for i in add_names.keys()]
            if k in df.columns]
    targets = df[cols]
    quantiles = targets.quantile(percent)
    for c in cols:
        df[c] = [i if i < quantiles[c] else quantiles[c] for i in df[c]]
    return df


def random_forest(df_o, factor,
                  cutoff, randomstate,
                  s1='2020-01-03', e1='2020-03-10',
                  features2=None, features3=None,
                  others=None,
                  train_date=None):
    """
    Prediction with random forest model for each station pollutant with df with rmse, mae and r2 scores for each station
    pollutant as a result.
    :param df_o: data frame with pollutants and meteo data
    :param factor: pollutant
    :param cutoff: percent of cut off
    :param randomstate: random state
    :param s1: start of prediction
    :param e1: end of prediction
    :param features2: weekday features
    :param features3: season features
    :param others: month and holidays features
    :param train_date: date of training data
    :return: df with rmse, mae and r2 scores for defined pollutant
    """
    if features2 is None:
        features2 = features2
    if features3 is None:
        features3 = features3
    if others is None:
        others = ['month', 'holiday', 'holiday_school', 'holiday_sunday']

    results = pd.DataFrame(index=['RMSE', 'MAE', 'R2'])
    for a, b in col_names:
        # print(a,b)
        if b + factor not in df_o.columns: continue
        cols = [i for i in df_o.columns if a in i]

        cols += features2 + features3 + others
        label_cols = [i for i in df_o.columns if b in i]

        df = df_o[cols + label_cols + ['year']]
        check_NA = df.isna().sum()
        if check_NA[check_NA > 0].shape[0] > 0:
            # print(f'check NA\n{check_NA}')
            df = df.dropna(how='any')

        if train_date is None:
            train = df[df['year'] < 2020]
        else:
            train = df.loc[:str(datetime.date.fromisoformat(s1) - datetime.timedelta(days=1))]
        test = df.loc[s1:e1]

        # cutoffs
        train = cut_off(train, cutoff)

        X_train = train[cols]
        y_train = train[b + factor]
        # display(y_train)

        X_test = test[cols]
        y_test = test[b + factor]
        # display(y_test)

        clf = RandomForestRegressor(random_state=randomstate)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # display(y_pred)
        y_pred = pd.Series(y_pred, index=y_test.index)
        # display(y_pred)
        mse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[b + factor] = [mse, mae, r2]
    return results


def random_forest_pollutants(df_o, percent, randomstate):
    """
    Rmse, mae and r2 scores of random forest prediction for all stations and pollutants.
    :param df_o: data frame with pollutants and meteo data
    :param percent: percent of cut off
    :param randomstate: random state
    :return: df with rmse, mae and r2 scores for all pollutants and stations
    """
    # print(f'percent: {percent}, random state: {randomstate}')
    df = pd.DataFrame()
    for factor in factors:
        # print(factor)
        r = random_forest(df_o=df_o, factor=factor,
                          cutoff=percent, randomstate=randomstate,
                          features2=features2, features3=features3,
                          others=['month', 'holiday', 'holiday_school', 'holiday_sunday'],
                          train_date=None)
        df = pd.concat([df, r], axis=1)
    df.index = df.index + '_' + str(percent) + '_' + str(randomstate)
    return df


def cutoff_errors(df_o, cutoffs, randomstates):
    """
    Rmse, mae and r2 scores of random forest prediction for all stations and pollutants and several cutoffs as mean of
    several random states.
    :param df_o: data frame with pollutant and meteo data
    :param cutoffs: several cutoffs
    :param randomstates: several random states
    :return: data frame with rmse, mae and r2 score for each cutoff and station pollutant
    """
    df_co = {}
    for percent in cutoffs:
        df_co[percent] = {}
        for randomstate in randomstates:
            df_co[percent][randomstate] = random_forest_pollutants(percent=percent, df_o=df_o, randomstate=randomstate)
    tmp = {}
    for percent in df_co.keys():
        tmp[percent] = pd.concat(df_co[percent], axis=0)

    final = pd.concat(tmp, axis=0)
    final.index = final.index.get_level_values(2)

    # mean of random states
    cutoff_list = [str(i) + '_' for i in cutoffs]
    tmp = {}
    for a in cutoff_list:
        tmp[a] = final.loc[[i for i in final.index if a in i]]

    t = {}
    for key in tmp.keys():
        t[key] = {}
        for b in ['RMSE', 'MAE', 'R2']:
            t[key][b] = tmp[key].loc[[i for i in tmp[key].index if b in i]].mean()

    df = {}
    for key in tmp.keys():
        df[key] = pd.DataFrame(columns=t[key]['RMSE'].index, index=t[key].keys(),
                               data=[t[key]['RMSE'], t[key]['MAE'], t[key]['R2']])
        df[key].index += '_' + key

    final_mean = pd.concat(df).droplevel(0)
    return final_mean


def r2_compare(final_mean):
    """
    Gives best cutoff r2 score and value of r2 score for each station pollutant.
    :param final_mean: data frame with rmse, mae and r2 score for each cutoff and station pollutant
    :return: data frame with best cutoff r2 score for each station pollutant
    """
    final_mean_r2 = final_mean.loc[[i for i in final_mean.index if 'R2_' in i]]
    # for each pollutant write cutoff with maximum r2 and r2 value
    final_mean_r2_max = pd.concat([final_mean_r2.idxmax()], axis=1)
    final_mean_r2_max.columns = ['R2']
    val_max = pd.concat([final_mean_r2.max()], axis=1)
    val_max.columns = ['R2']
    final_mean_r2_max['R2_val'] = val_max['R2']
    # final_mean_r2_max = final_mean_r2_max.sort_values('R2_val')
    print(final_mean_r2_max['R2'].value_counts())
    return final_mean_r2_max
