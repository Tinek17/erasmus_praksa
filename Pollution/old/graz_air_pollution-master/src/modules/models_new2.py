import pandas as pd
import numpy as np
import datetime
from bayes_opt import BayesianOptimization
from prophet import Prophet
from prophet.utilities import regressor_coefficients
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

AREAS = ['D_', 'N_', 'O_', 'S_', 'W_']  # station abbrevation
add_names = {  # station full name
    'D_': 'DonBosco_',
    'N_': 'Nord_',
    'O_': 'Ost_',
    'W_': 'West_',
    'S_': 'Sud_'}

col_names = [['DonBosco_', 'D_'], ['Nord_', 'N_'], ['Ost_', 'O_'], ['Sud_', 'S_'], ['West_', 'W_']]

factors = ['NO2', 'PM10K', 'O3']  # pollutants
factors_NO2 = ['NO2']

# weather features
features = ['Temp', 'RH', 'Pressure', 'Winddirection', 'Windspeed', 'Precip']
# weekday features
features2 = ['weekday_Friday', 'weekday_Monday', 'weekday_Saturday', 'weekday_Sunday', 'weekday_Thursday',
             'weekday_Tuesday', 'weekday_Wednesday']
# season features
features3 = ['season_fall', 'season_spring', 'season_summer', 'season_winter']

holidays = ['holiday', 'holiday_school', 'holiday_sunday']
weekday_convert = {
    'weekday_Sunday': 0,
    'weekday_Monday': 1,
    'weekday_Tuesday': 2,
    'weekday_Wednesday': 3,
    'weekday_Thursday': 4,
    'weekday_Friday': 5,
    'weekday_Saturday': 6}
season_convert = {
    'season_spring': 1,
    'season_summer': 2,
    'season_fall': 3,
    'season_winter': 4}


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


# RandomForestRegressor
def rf_predict(df_o, factor, params,
               cutoff, randomstate, others,
               s1='2020-01-03', e1='2020-03-10',
               features2=None, features3=None,
               init_points=10, n_iter=5,
               train_date=None):
    """
        Prediction with random forest model for each station pollutant with df with rmse, mae and r2 scores for each station
        pollutant as a result.
        :param n_iter:
        :param init_points:
        :param params:
        :param df_o: data frame with pollutants and meteo data
        :param factor: pollutant
        :param cutoff: percent of cut off
        :param randomstate: random state
        :param s1: start of prediction
        :param e1: end of prediction
        :param features2: weekday features
        :param features3: season features
        :param others: list of columns as additional features #month and holidays features
        :param train_date: date of training data
        :return: df with rmse, mae and r2 scores for defined pollutant
        """
    if features2 is None:
        features2 = features2
    if features3 is None:
        features3 = features3
    # if others is None:
    #   others = ['month', 'holiday', 'holiday_school', 'holiday_sunday']

    y_test = {}
    y_pred = {}
    for a, b in col_names:
        print(a, b)
        params_gbm = params.copy()
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

        X_test = test[cols]
        y_test[b + factor] = test[b + factor]

        def gbm_cl_bo(max_depth,
                      n_estimators):
            params_gbm = {'max_depth': round(max_depth), 'n_estimators': round(n_estimators)}
            scores = cross_val_score(RandomForestRegressor(**params_gbm, random_state=randomstate),
                                     X_train, y_train, cv=5).mean()
            score = scores.mean()
            return score

        gbm_bo = BayesianOptimization(gbm_cl_bo, params_gbm,
                                      random_state=randomstate)
        gbm_bo.maximize(init_points=init_points, n_iter=n_iter)

        params_gbm = gbm_bo.max['params']
        params_gbm['max_depth'] = round(params_gbm['max_depth'])
        params_gbm['n_estimators'] = round(params_gbm['n_estimators'])
        params_gbm['random_state'] = randomstate
        print(params_gbm)

        clf = RandomForestRegressor(**params_gbm)
        clf.fit(X_train, y_train)
        y_pred[b + factor] = clf.predict(X_test)
        y_pred[b + factor] = pd.Series(y_pred[b + factor], index=y_test[b + factor].index)

        y_pred_all = pd.concat(y_pred, axis=1)
        y_pred_all = y_pred_all.add_suffix('_pred')

        y_test_all = pd.concat(y_test, axis=1)

    df_all = pd.concat([y_test_all, y_pred_all], axis=1)
    df_all = df_all.reindex(sorted(df_all.columns), axis=1)
    return df_all


def rf_prediction(df_o, percent, randomstate, params_gbm, factors, add_feat):
    """

    :param add_feat:
    :param factors:
    :param params_gbm:
    :param df_o:
    :param percent:
    :param randomstate:
    :return:
    """
    # print(f'percent: {percent}, random state: {randomstate}')
    df = pd.DataFrame()
    # for factor in ['O3']:
    for factor in factors:
        print(factor)
        r = rf_predict(df_o=df_o, factor=factor, params=params_gbm,
                       cutoff=percent, randomstate=randomstate,
                       features2=features2, features3=features3,
                       others=add_feat,
                       train_date=None, init_points=10, n_iter=5)
        df = pd.concat([df, r], axis=1)
    return df


# ExtraTreesRegressor
def extree_predict(df_o, factor, params,
                   cutoff, randomstate, others,
                   s1='2020-01-03', e1='2020-03-10',
                   features2=None, features3=None,
                   init_points=10, n_iter=5,
                   train_date=None):
    """
        Prediction with random forest model for each station pollutant with df with rmse, mae and r2 scores for each station
        pollutant as a result.
        :param n_iter:
        :param init_points:
        :param params:
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
    # if others is None:
    #   others = ['month', 'holiday', 'holiday_school', 'holiday_sunday']

    y_test = {}
    y_pred = {}
    for a, b in col_names:
        print(a, b)
        params_gbm = params.copy()
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

        X_test = test[cols]
        y_test[b + factor] = test[b + factor]

        def gbm_cl_bo(max_depth,
                      n_estimators):
            params_gbm = {'max_depth': round(max_depth), 'n_estimators': round(n_estimators)}
            scores = cross_val_score(ExtraTreesRegressor(**params_gbm, random_state=randomstate),
                                     X_train, y_train, cv=5).mean()
            score = scores.mean()
            return score

        gbm_bo = BayesianOptimization(gbm_cl_bo, params_gbm,
                                      random_state=randomstate)
        gbm_bo.maximize(init_points=init_points, n_iter=n_iter)

        params_gbm = gbm_bo.max['params']
        params_gbm['max_depth'] = round(params_gbm['max_depth'])
        params_gbm['n_estimators'] = round(params_gbm['n_estimators'])
        params_gbm['random_state'] = randomstate
        print(params_gbm)

        clf = ExtraTreesRegressor(**params_gbm)
        clf.fit(X_train, y_train)
        y_pred[b + factor] = clf.predict(X_test)
        y_pred[b + factor] = pd.Series(y_pred[b + factor], index=y_test[b + factor].index)

        y_pred_all = pd.concat(y_pred, axis=1)
        y_pred_all = y_pred_all.add_suffix('_pred')

        y_test_all = pd.concat(y_test, axis=1)

    df_all = pd.concat([y_test_all, y_pred_all], axis=1)
    df_all = df_all.reindex(sorted(df_all.columns), axis=1)
    return df_all


def extree_prediction(df_o, percent, randomstate, params_gbm, factors, add_feat):
    """

    :param factors:
    :param add_feat:
    :param params_gbm:
    :param df_o:
    :param percent:
    :param randomstate:
    :return:
    """
    # print(f'percent: {percent}, random state: {randomstate}')
    df = pd.DataFrame()
    # for factor in ['O3']:
    for factor in factors:
        print(factor)
        r = extree_predict(df_o=df_o, factor=factor, params=params_gbm,
                           cutoff=percent, randomstate=randomstate,
                           features2=features2, features3=features3,
                           others=add_feat,
                           train_date=None, init_points=10, n_iter=5)
        df = pd.concat([df, r], axis=1)
    return df


# XGBRegressor
def xgb_predict(df_o, factor, params,
                cutoff, randomstate, others,
                s1='2020-01-03', e1='2020-03-10',
                features2=None, features3=None,
                init_points=10, n_iter=5,
                train_date=None):
    """
        Prediction with random forest model for each station pollutant with df with rmse, mae and r2 scores for each station
        pollutant as a result.
        :param n_iter:
        :param init_points:
        :param params:
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
    # if others is None:
    # others = ['month', 'holiday', 'holiday_school', 'holiday_sunday']

    y_test = {}
    y_pred = {}
    for a, b in col_names:
        print(a, b)
        params_gbm = params.copy()
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

        X_test = test[cols]
        y_test[b + factor] = test[b + factor]

        def gbm_cl_bo(max_depth,
                      n_estimators,
                      eta,
                      subsample,
                      colsample_bytree
                      ):
            params_gbm = {'max_depth': round(max_depth), 'n_estimators': round(n_estimators), 'eta': eta,
                          'subsample': subsample, 'colsample_bytree': colsample_bytree}
            scores = cross_val_score(XGBRegressor(**params_gbm, random_state=randomstate),
                                     X_train, y_train, cv=5).mean()
            score = scores.mean()
            return score

        gbm_bo = BayesianOptimization(gbm_cl_bo, params_gbm,
                                      random_state=randomstate)
        gbm_bo.maximize(init_points=init_points, n_iter=n_iter)

        params_gbm = gbm_bo.max['params']
        params_gbm['max_depth'] = round(params_gbm['max_depth'])
        params_gbm['n_estimators'] = round(params_gbm['n_estimators'])
        params_gbm['random_state'] = randomstate
        print(params_gbm)

        clf = XGBRegressor(**params_gbm)
        clf.fit(X_train, y_train)
        y_pred[b + factor] = clf.predict(X_test)
        y_pred[b + factor] = pd.Series(y_pred[b + factor], index=y_test[b + factor].index)

        y_pred_all = pd.concat(y_pred, axis=1)
        y_pred_all = y_pred_all.add_suffix('_pred')

        y_test_all = pd.concat(y_test, axis=1)

        df_all = pd.concat([y_test_all, y_pred_all], axis=1)
        df_all = df_all.reindex(sorted(df_all.columns), axis=1)
    return df_all


def xgb_prediction(df_o, percent, randomstate, params_gbm, factors, add_feat):
    """

    :param add_feat:
    :param factors:
    :param params_gbm:
    :param df_o:
    :param percent:
    :param randomstate:
    :return:
    """
    # print(f'percent: {percent}, random state: {randomstate}')
    df = pd.DataFrame()
    # for factor in ['O3']:
    for factor in factors:
        print(factor)
        r = xgb_predict(df_o=df_o, factor=factor, params=params_gbm,
                        cutoff=percent, randomstate=randomstate,
                        features2=features2, features3=features3,
                        others=add_feat,
                        train_date=None, init_points=10, n_iter=5)
        df = pd.concat([df, r], axis=1)
    return df


# Prophet
def prophet_predict(df_org, factor, periods=None, holidays=None,
                    features=None, features2=None, features3=False,
                    seasonality_mode='additive',
                    s1=None, e1=None):
    """
    Result is df with y true and y predicted for all stations and pollutants.
    :param df_org: df with datetime index
    :param factor: one pollutant in factors (['NO2', 'PM10K', 'O3'])
    :param periods: number of days for prediction
    :param holidays: public or school holiday
    :param features: meteorological data (Temp, RH, etc.)
    :param features2: weekday data (Mon, Tue, etc.)
    :param features3: station+meteo_feature values squared
    :param seasonality_mode: 'additive' (default) or 'multiplicative'
    :param s1: start date of training data
    :param e1: end date of training data
    :return: df with y true and y predicted for pollutant for all stations
    """
    forecast = {}
    coeff = {}
    if periods is None:
        periods = (datetime.datetime.strptime(e1, '%Y-%m-%d') -
                   datetime.datetime.strptime(s1, '%Y-%m-%d')).days + 1  # number of days for prediction

    for area in AREAS:  # stations abbrevations
        print(area)
        if area == 'O_':
            a = df_org.iloc[:, -5:].dropna(how='all').index  # for O_ features data (Temp, RH etc.) starts in 2017,
            # and not in 2010 like for other stations (AREAS)
            df_org = df_org.loc[a].copy()  # because of that, all data for O_ starts when features available (from 2017)
            print(df_org.shape)
        else:
            df_org = df_org.copy()  # other stations are fine and all data start with 2010-01-01
        if features is not None:
            try:
                add_features = [add_names[area] + feature for feature in features  # list of available features
                                if add_names[area] + feature in df_org.columns  # (Temp, RH etc.) for each station
                                ]
            except:
                pass
        # find appropriate features3 list
        if features3:
            features3 = [f + '_2' for f in
                         add_features]  # features3 before were seasons (season_fall, season_spring etc.)
            # now it's a list of add_features with suffix _2
        col = area + factor
        if col not in df_org: continue

        if holidays is not None:  # if holidays present include them in Prophet
            m = Prophet(holidays=holidays, seasonality_mode=seasonality_mode)
        else:
            m = Prophet(  # if holidays aren't present include only seasonality_mode
                #                changepoint_prior_scale=0.01,
                #                holidays_prior_scale=0.25,
                #                mcmc_samples=300,
                #                yearly_seasonality=10,
                #                daily_seasonality=False,
                seasonality_mode=seasonality_mode
            )

        if s1 and e1:  # if start and end date defined, take df_org until end date (e1),
            df = df_org[:e1][[col]].reset_index()  # save one df for each col (station+pollutant), and reset index
        else:
            df = df_org[[col]].reset_index()  # if start and end date not defined, save one df for each col
            # (station+pollutant) and reset index
        df.columns = ['ds',
                      'y']  # rename date column as 'ds' and station+pollutant as 'y' because that will be predicted

        # add features to data and to additional regressor
        if features is not None:  # features are Temp, RH etc.
            for add_feature in add_features:  # add_features are station+features
                if s1 and e1:
                    df[add_feature] = df_org[:e1][add_feature].values  # add from df_original station+features
                else:
                    df[add_feature] = df_org[add_feature].values
                m.add_regressor(add_feature,  # prior_scale=0.5,     # column with regressor value added which is called
                                standardize=True)  # station+feature
        if features2 is not None:  # features2 are weekdays, here they are also added as regressor columns
            for f in features2:
                if s1 and e1:
                    df[f] = df_org[:e1][f].values
                else:
                    df[f] = df_org[f].values
                m.add_regressor(f,  # prior_scale=0.5,
                                standardize=True)
        if features3 and len(features3) > 0:
            for f in features3:
                if s1 and e1:
                    df[f] = df_org[:e1][f].values
                else:
                    df[f] = df_org[f].values
                m.add_regressor(f,  # prior_scale=0.5,
                                standardize=True)

        if s1 and e1:
            train = df.set_index('ds', drop=True)[:s1].iloc[:-1, :].reset_index()  # take train data untill start date
            # for prediction
        else:
            train = df.iloc[:-periods, :]  # if start date of prediction not defined exclude last days calculated in
            # periods

        # remove NAs
        check_NA = train.isna().sum()
        if check_NA[check_NA > 0].shape[0] > 0:
            print(f'check NA\n{check_NA}')
            train = train.dropna(how='any').reset_index(drop=True)

        df = pd.concat([train, df.set_index('ds', drop=True).loc[s1:].reset_index()],  # concat train data and test data
                       ignore_index=True
                       )

        m.fit(train)

        future = m.make_future_dataframe(periods=periods)
        future = future[future['ds'].isin(df['ds'])]

        if features is not None:  # weather features Temp, RH etc.
            coeff[col] = regressor_coefficients(m)
            for add_feature in add_features:  # station+weather_features
                future[add_feature] = df[add_feature].values
        if features2 is not None:  # weekday features
            for f in features2:
                future[f] = df[f].values
        if features3 and len(features3) > 0:
            for f in features3:
                future[f] = df[f].values

        for c in future.columns:
            if future[c].isna().sum() != 0:
                future[c] = future[c].fillna(future[c].mean())  # fill nans with mean
        forecast[col] = m.predict(future)  # make a prediction
        #         if plot:
        #             fig1 = m.plot(forecast)
        #             plt.title(col)
        #             plt.show()
        forecast[col] = forecast[col].set_index('ds').join(df[['ds', 'y']].set_index('ds')).iloc[-periods:, :]

        forecast_all = pd.concat(forecast, axis=1)
    return forecast_all


def prophet_prediction(df_o, percent, s1, e1, factors):
    """

    :param factors:
    :param df_o:
    :param percent:
    :param s1:
    :param e1:
    :return:
    """
    df_org = cut_off(df_o, percent)
    forecast = {}
    # coefs = {}

    for method in ['f0_add', 'f0_mul', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6']:
        forecast_col = {}
        for factor in factors:
            if method == 'f0_add':
                print(method)
                forecast_col[factor] = prophet_predict(df_org=df_org, factor=factor, s1=s1, e1=e1)
            if method == 'f0_mul':
                print(method)
                forecast_col[factor] = prophet_predict(df_org=df_org, factor=factor, s1=s1, e1=e1,
                                                       seasonality_mode='multiplicative')
            if method == 'f1':
                print(method)
                forecast_col[factor] = prophet_predict(df_org=df_org, factor=factor, s1=s1, e1=e1,
                                                       features=features,
                                                       seasonality_mode='multiplicative')
            if method == 'f2':
                print(method)
                forecast_col[factor] = prophet_predict(df_org=df_org, factor=factor, s1=s1, e1=e1,
                                                       features=features, features2=features2,
                                                       seasonality_mode='multiplicative')
            if method == 'f3':
                print(method)
                holiday_normal = pd.DataFrame(df_org[df_org['holiday'] == 1].index, columns=['ds'])
                holiday_normal['holiday'] = 'holiday'
                holiday_school = pd.DataFrame(df_org[df_org['holiday_school'] == 1].index, columns=['ds'])
                holiday_school['holiday'] = 'holiday_school'
                a = pd.concat([holiday_normal, holiday_school]).sort_values('ds')
                a = a.groupby('ds')['holiday'].agg([len, list])
                a['holiday'] = [i[0] if len(i) == 1 else 'holiday' for i in a['list']]
                holidays = a[['holiday']].reset_index()
                holidays['lower_window'] = 0
                holidays['upper_window'] = 1

                forecast_col[factor] = prophet_predict(df_org=df_org, factor=factor, s1=s1, e1=e1,
                                                       features=features, holidays=holidays,
                                                       seasonality_mode='multiplicative')
            if method == 'f4':
                print(method)
                forecast_col[factor] = prophet_predict(df_org=df_org, factor=factor, s1=s1, e1=e1,
                                                       features=features, features2=features2 + features3,
                                                       holidays=holidays,
                                                       seasonality_mode='multiplicative')
            if method == 'f5':
                print(method)
                for c in features2:
                    df_org[c] = [weekday_convert[c] if i else i for i in df_org[c]]
                df_org['dayoftheweek'] = df_org[features2].sum(axis=1)
                for c in season_convert.keys():
                    df_org[c] = [season_convert[c] if i else i for i in df_org[c]]
                df_org['season'] = df_org[season_convert.keys()].sum(axis=1)
                # Cross day of the week and season in different way
                df_org['cross1'] = df_org['season'] * df_org['dayoftheweek']
                df_org['cross2'] = df_org['season'] ** 2 + df_org['dayoftheweek'] ** 2
                df_org['cross3'] = df_org['season'] ** 2
                df_org['cross4'] = df_org['dayoftheweek'] ** 2
                crosses = ['cross1', 'cross2', 'cross3', 'cross4']

                forecast_col[factor] = prophet_predict(df_org=df_org, factor=factor, s1=s1, e1=e1,
                                                       features=features,
                                                       features2=['dayoftheweek', 'season', 'dayofyear'] + crosses,
                                                       holidays=holidays,
                                                       seasonality_mode='multiplicative')
            if method == 'f6':
                print(method)
                # new cols defined as station+weather feature
                cols = [c for c in [j + i for j in add_names.values() for i in features] if c in df_org.columns]
                # assign new columns in df_org as squared weather features
                for c in cols:
                    df_org[c + '_2'] = df_org[c] ** 2

                forecast_col[factor] = prophet_predict(df_org=df_org, factor=factor, s1=s1, e1=e1,
                                                       features=features,
                                                       features2=['dayoftheweek', 'season', 'dayofyear'] + crosses,
                                                       features3=True, holidays=holidays,
                                                       seasonality_mode='multiplicative')

        need_cols = ['y', 'yhat']
        forecast[method] = pd.concat(forecast_col, axis=1).droplevel(0, axis=1)  # drop first level of column names
        forecast[method].drop(columns=[i for i in forecast[method].columns.get_level_values(1)
                                       # keep only 'y' (y_true) and 'yhat' (y_pred)
                                       if i not in need_cols], level=1, inplace=True)

    df_pred = pd.concat(forecast, axis=1)
    return df_pred


# ===========================================
# Random Forest for Ost station and NO2 and PM10K only
# ost only
def rf_predict_Ost(df_o, factor, params,
                   cutoff, randomstate, others,
                   s1='2020-01-03', e1='2020-03-10',
                   features2=None, features3=None,
                   init_points=10, n_iter=5,
                   train_date=None):
    """
        Prediction with random forest model for each station pollutant with df with rmse, mae and r2 scores for each station
        pollutant as a result.
        :param n_iter:
        :param init_points:
        :param params:
        :param df_o: data frame with pollutants and meteo data
        :param factor: pollutant
        :param cutoff: percent of cut off
        :param randomstate: random state
        :param s1: start of prediction
        :param e1: end of prediction
        :param features2: weekday features
        :param features3: season features
        :param others: list of columns as additional features #month and holidays features
        :param train_date: date of training data
        :return: df with rmse, mae and r2 scores for defined pollutant
        """
    if features2 is None:
        features2 = features2
    if features3 is None:
        features3 = features3
    # if others is None:
    #   others = ['month', 'holiday', 'holiday_school', 'holiday_sunday']

    y_test = {}
    y_pred = {}
    # col_names = [['DonBosco_', 'D_'], ['Nord_', 'N_'], ['Ost_', 'O_'], ['Sud_', 'S_'], ['West_', 'W_']]
    for a, b in [['Ost_', 'O_']]:
        print(a, b)
        params_gbm = params.copy()
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

        X_test = test[cols]
        y_test[b + factor] = test[b + factor]

        def gbm_cl_bo(max_depth,
                      n_estimators):
            params_gbm = {'max_depth': round(max_depth), 'n_estimators': round(n_estimators)}
            scores = cross_val_score(RandomForestRegressor(**params_gbm, random_state=randomstate),
                                     X_train, y_train, cv=5).mean()
            score = scores.mean()
            return score

        gbm_bo = BayesianOptimization(gbm_cl_bo, params_gbm,
                                      random_state=randomstate)
        gbm_bo.maximize(init_points=init_points, n_iter=n_iter)

        params_gbm = gbm_bo.max['params']
        params_gbm['max_depth'] = round(params_gbm['max_depth'])
        params_gbm['n_estimators'] = round(params_gbm['n_estimators'])
        params_gbm['random_state'] = randomstate
        print(params_gbm)

        clf = RandomForestRegressor(**params_gbm)
        clf.fit(X_train, y_train)
        y_pred[b + factor] = clf.predict(X_test)
        y_pred[b + factor] = pd.Series(y_pred[b + factor], index=y_test[b + factor].index)

        y_pred_all = pd.concat(y_pred, axis=1)
        y_pred_all = y_pred_all.add_suffix('_pred')

        y_test_all = pd.concat(y_test, axis=1)

    df_all = pd.concat([y_test_all, y_pred_all], axis=1)
    df_all = df_all.reindex(sorted(df_all.columns), axis=1)
    return df_all


def rf_prediction_Ost(df_o, percent, randomstate, params_gbm, factors, add_feat):
    """

    :param add_feat:
    :param factors:
    :param params_gbm:
    :param df_o:
    :param percent:
    :param randomstate:
    :return:
    """
    # print(f'percent: {percent}, random state: {randomstate}')
    df = pd.DataFrame()
    # for factor in ['O3']:
    for factor in factors:
        print(factor)
        r = rf_predict_Ost(df_o=df_o, factor=factor, params=params_gbm,
                           cutoff=percent, randomstate=randomstate,
                           features2=features2, features3=features3,
                           others=add_feat,
                           train_date=None, init_points=10, n_iter=5)
        df = pd.concat([df, r], axis=1)
    return df
