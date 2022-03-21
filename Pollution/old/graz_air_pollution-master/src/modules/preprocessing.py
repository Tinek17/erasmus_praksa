# Notebook 01_DataProcessing

import glob
import pandas as pd
from workalendar.europe import Austria

SEASONS = {
    'winter': [12, 1, 2],
    'spring': [3, 4, 5],
    'summer': [6, 7, 8],
    'fall': [9, 10, 11]
}

MONTHS = {
    month: season for season in SEASONS.keys()
    for month in SEASONS[season]
}

mo = {
    1: 'Jan',
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec'
}

STATIONS = {
    'DonBosco': 'D',
    'Nord': 'N',
    'Sud': 'S',
    'West': 'W',
    'Ost': 'O'
}

# If in reference stated that school holidays start at weekend, here is first week day taken as a start of school holidays
# Some holidays not marked as school holidays because it's one day of not going to the school (could be fixed with holiday==1)
# reference: https://www.schulferien.org/oesterreich/kalender/steiermark/2010/
school_hol = dict({
    ('2010-01-01', '2010-01-06'): 1,
    ('2010-02-15', '2010-02-19'): 1,
    ('2010-03-29', '2010-04-06'): 1,
    ('2010-05-24', '2010-05-25'): 1,
    ('2010-07-12', '2010-09-10'): 1,
    ('2010-12-24', '2010-12-31'): 1,

    ('2011-01-03', '2011-01-06'): 1,
    ('2011-02-21', '2011-02-25'): 1,
    ('2011-04-18', '2011-04-26'): 1,
    ('2011-06-13', '2011-06-14'): 1,
    ('2011-07-11', '2011-09-09'): 1,
    ('2011-12-26', '2011-12-30'): 1,

    ('2012-01-02', '2012-01-06'): 1,
    ('2012-02-20', '2012-02-24'): 1,
    ('2012-04-02', '2012-04-10'): 1,
    ('2012-05-28', '2012-05-29'): 1,
    ('2012-07-09', '2012-09-07'): 1,
    ('2012-12-24', '2012-12-31'): 1,

    ('2013-01-01', '2013-01-04'): 1,
    ('2013-02-18', '2013-02-22'): 1,
    ('2013-03-25', '2013-04-02'): 1,
    ('2013-05-20', '2013-05-21'): 1,
    ('2013-07-08', '2013-09-06'): 1,
    ('2013-12-23', '2013-12-31'): 1,

    ('2014-01-01', '2014-01-06'): 1,
    ('2014-02-17', '2014-02-21'): 1,
    ('2014-04-14', '2014-04-22'): 1,
    ('2014-06-09', '2014-06-10'): 1,
    ('2014-07-07', '2014-09-05'): 1,
    ('2014-12-24', '2014-12-31'): 1,

    ('2015-01-01', '2015-01-06'): 1,
    ('2015-02-16', '2015-02-20'): 1,
    ('2015-03-30', '2015-04-07'): 1,
    ('2015-05-25', '2015-05-26'): 1,
    ('2015-07-13', '2015-09-11'): 1,
    ('2015-12-24', '2015-12-31'): 1,

    ('2016-01-01', '2016-01-06'): 1,
    ('2016-02-15', '2016-02-19'): 1,
    ('2016-03-21', '2016-03-29'): 1,
    ('2016-05-06', '2016-05-06'): 1,
    ('2016-05-16', '2016-05-17'): 1,
    ('2016-05-27', '2016-05-27'): 1,
    ('2016-07-11', '2016-09-09'): 1,
    ('2016-12-26', '2016-12-30'): 1,

    ('2017-01-02', '2017-01-06'): 1,
    ('2017-02-20', '2017-02-24'): 1,
    ('2017-04-10', '2017-04-18'): 1,
    ('2017-06-05', '2017-06-06'): 1,
    ('2017-07-10', '2017-09-08'): 1,
    ('2017-12-25', '2017-12-29'): 1,

    ('2018-01-01', '2018-01-05'): 1,
    ('2018-02-19', '2018-02-23'): 1,
    ('2018-03-19', '2018-03-19'): 1,
    ('2018-03-26', '2018-04-03'): 1,
    ('2018-05-21', '2018-05-22'): 1,
    ('2018-07-09', '2018-09-07'): 1,
    ('2018-12-24', '2018-12-31'): 1,

    ('2019-01-01', '2019-01-04'): 1,
    ('2019-02-18', '2019-02-22'): 1,
    ('2019-03-19', '2019-03-19'): 1,
    ('2019-04-15', '2019-04-23'): 1,
    ('2019-05-10', '2019-05-11'): 1,
    ('2019-07-08', '2019-09-06'): 1,
    ('2019-12-23', '2019-12-31'): 1,

    ('2020-01-01', '2020-01-06'): 1,
    ('2020-02-17', '2020-02-21'): 1,
    ('2020-03-19', '2020-03-19'): 1,
    ('2020-04-06', '2020-04-14'): 1,
    ('2020-06-01', '2020-06-02'): 1,
    ('2020-07-13', '2020-09-11'): 1,
    ('2020-10-27', '2020-10-30'): 1,
    ('2020-12-24', '2020-12-31'): 1,

    ('2021-01-01', '2021-01-06'): 1,
    ('2021-02-08', '2021-02-12'): 1,
    ('2021-03-19', '2021-03-19'): 1,
    ('2021-03-29', '2021-04-05'): 1,
    ('2021-07-12', '2021-09-10'): 1,
    ('2021-10-27', '2021-10-29'): 1,
    ('2021-12-24', '2021-12-31'): 1,
})


def import_data(path):
    """
    Imports all data available in folder.
    :param path: folder location with data files
    :return: one data frame from all files
    """
    filenames = glob.glob(path + '/*.xls')
    # Dictionary of xls files
    df_dict = {}
    for f in filenames:
        df_dict[f[63:-26]] = pd.read_excel(f, skiprows=4, names=['date', 'time', str(f[63:-26])],
                                           index_col='date').drop(columns='time')
    df = pd.concat(df_dict, axis=1)
    # Index as datetime
    df.index = pd.to_datetime(df.index, dayfirst=True)
    # Sort index
    df = df.sort_index()
    # Drop duplicated column names
    df.columns = df.columns.droplevel(level=0)
    # Rename station names to their abbrevations
    for key, value in zip(STATIONS.keys(), STATIONS.values()):
        df.columns = df.columns.str.replace(key, value)
    return df


def import_data_year2010(path):
    """
    Imports all data available in folder and filter data from date 2010-01-01.
    :param path: folder location with data files
    :return: one data frame from all files from date 2010-01-01
    """
    filenames = glob.glob(path + '/*.xls')
    # Dictionary of xls files
    df_dict = {}
    for f in filenames:
        df_dict[f[63:-26]] = pd.read_excel(f, skiprows=4, names=['date', 'time', str(f[63:-26])],
                                           index_col='date').drop(columns='time')
    df = pd.concat(df_dict, axis=1)
    # Index as datetime
    df.index = pd.to_datetime(df.index, dayfirst=True)
    # Sort index
    df = df.sort_index()
    # Drop duplicated column names
    df.columns = df.columns.droplevel(level=0)
    # Take data from 2010-01-01
    df_2010 = df['2010-01-01':]
    # Drop PM10 because STBK10K will be used as a measure for PM10, and rename STBK10K to PM10K
    cols = [c for c in df.columns if c[-4:] != 'PM10']
    df_2010 = df_2010[cols]
    df_2010.columns = df_2010.columns.str.replace('STBK10K', 'PM10K')
    # Drop stations Gries and Lustbuhl because for 2010 many missing data present
    df_2010 = df_2010.drop(columns=['Gries_NO2', 'Gries_PM10K', 'Lustbuhel_O3', 'Lustbuhel_PM10K'])
    # Rename station names to their abbrevations
    for key, value in zip(STATIONS.keys(), STATIONS.values()):
        df_2010.columns = df_2010.columns.str.replace(key, value)
    return df_2010


def import_meteo_year2010(path):
    """
    Imports all data available in folder and filter data from date 2010-01-01.
    :param path: folder location with data files
    :return: one data frame from all files from date 2010-01-01
    """
    filenames = glob.glob(path + '/*.xls')
    # Dictionary of xls files
    df_dict = {}
    for f in filenames:
        df_dict[f[93:-26]] = pd.read_excel(f, skiprows=4, names=['date', 'time', str(f[93:-26])],
                                           index_col='date').drop(columns='time')
    df = pd.concat(df_dict, axis=1)
    # Index as datetime
    df.index = pd.to_datetime(df.index, dayfirst=True)
    # Sort index
    df = df.sort_index()
    # Drop duplicated column names
    df.columns = df.columns.droplevel(level=0)
    # Take data from 2010-01-01
    df_2010 = df['2010-01-01':]
    return df_2010


def temporal_feat(df):
    """
    Adds temporal features (year, month, day of year, week day and season) as new columns to the data frame.
    :param df: data frame with datetime index
    :return: data frame with added temporal features as new columns
    """
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['month'].replace(mo, inplace=True)
    df['dayofyear'] = df.index.to_series().apply(lambda x: x.strftime('%j'))
    df['weekday'] = df.index.to_series().apply(lambda x: x.strftime('%A'))
    df['season'] = df.index.month.map(MONTHS)
    return df


def temporal_dummies(df, temporal_columns: list):
    """
    From temporal features as column returns data frame with temporal features as dummies (one unique value in temporal
    features column becomes new column).
    :param df: data frame with temporal features as columns
    :param temporal_columns: list of columns wanted as dummies
    :return: data frame with temporal features as dummies columns
    """
    temporal = pd.get_dummies(df[temporal_columns])
    df = df.drop(temporal_columns, axis=1)  # drop columns which become dummies
    df_temp = pd.concat([df, temporal], axis=1)
    return df_temp


def public_holidays(df):
    """
    From data frame with datetime index returns data frame with new column holiday where value is 1 if it was public
    holiday for that datetime in Austria and 0 if not.
    :param df: data frame with datetime index
    :return: data frame with column holiday with values 1 (public holiday in Austria) and 0 (no public holiday in Austria)
    """
    hol = Austria()
    years = df.index.year.unique()
    at_hol = []
    for year in years:
        for date in hol.holidays(year):
            at_hol.append(str(date[0]))
    df['holiday'] = [1 if i.split()[0] in at_hol else 0 for i in df.index.astype(str)]
    return df


def school_holidays(df):
    """
    From data frame with datetime index returns data frame with new column holiday_school where value is 1 if it was
    school holiday for that datetime in Austria and 0 if not.
    Source https://www.schulferien.org/oesterreich/kalender/steiermark/2010/
    :param df: data frame with datetime index
    :return: data frame with column holiday_school with values 1 (school holiday in Austria) and 0 (no school holiday in
    Austria)
    """
    period = {x: v for (k1, k2), v in school_hol.items() for x in pd.period_range(k1, k2, freq='D')}
    days = []
    for i in period:
        days.append(i.strftime('%Y-%m-%d'))
    df['holiday_school'] = [1 if i.split()[0] in days else 0 for i in df.index.astype(str)]
    return df
