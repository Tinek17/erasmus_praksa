import pandas as pd
import numpy as np
import time
from math import sqrt
import pickle
import os
import itertools
import pickle
from sys import modules
from os import listdir

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import warnings



# model imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


meteo = ['DonBosco_Temp_mean', 'DonBosco_RH_mean', 'Nord_Temp_mean',
       'Nord_RH_mean', 'Nord_Pressure_mean', 'Nord_Precip_mean',
       'Nord_Radiation_mean', 'Nord_Winddirection_mean', 'Nord_Windspeed_mean',
       'Nord_peek_wind_speed_mean', 'Ost_Temp_mean', 'Ost_RH_mean',
       'Ost_Pressure_mean', 'Ost_Winddirection_mean', 'Ost_Windspeed_mean',
       'Ost_peek_wind_speed_mean', 'S_PM10K_mean', 'Sud_Temp_mean',
       'Sud_RH_mean', 'Sud_Winddirection_mean', 'Sud_Windspeed_mean',
       'Sud_peek_wind_speed_mean', 'West_Temp_mean', 'West_RH_mean',
       'West_Winddirection_mean', 'West_Windspeed_mean',
       'West_peek_wind_speed_mean','DonBosco_Temp_min', 'DonBosco_RH_min', 'Nord_Temp_min', 'Nord_RH_min',
       'Nord_Pressure_min', 'Nord_Precip_min', 'Nord_Radiation_min',
       'Nord_Winddirection_min', 'Nord_Windspeed_min',
       'Nord_peek_wind_speed_min', 'Ost_Temp_min', 'Ost_RH_min',
       'Ost_Pressure_min', 'Ost_Winddirection_min', 'Ost_Windspeed_min',
       'Ost_peek_wind_speed_min', 'S_PM10K_min', 'Sud_Temp_min', 'Sud_RH_min',
       'Sud_Winddirection_min', 'Sud_Windspeed_min', 'Sud_peek_wind_speed_min',
       'West_Temp_min', 'West_RH_min', 'West_Winddirection_min',
       'West_Windspeed_min', 'West_peek_wind_speed_min','DonBosco_Temp_max', 'DonBosco_RH_max', 'Nord_Temp_max', 'Nord_RH_max',
       'Nord_Pressure_max', 'Nord_Precip_max', 'Nord_Radiation_max',
       'Nord_Winddirection_max', 'Nord_Windspeed_max',
       'Nord_peek_wind_speed_max', 'Ost_Temp_max', 'Ost_RH_max',
       'Ost_Pressure_max', 'Ost_Winddirection_max', 'Ost_Windspeed_max',
       'Ost_peek_wind_speed_max', 'S_PM10K_max', 'Sud_Temp_max', 'Sud_RH_max',
       'Sud_Winddirection_max', 'Sud_Windspeed_max', 'Sud_peek_wind_speed_max',
       'West_Temp_max', 'West_RH_max', 'West_Winddirection_max',
       'West_Windspeed_max', 'West_peek_wind_speed_max']

pollutants = ['D_PM10K_mean','N_PM10K_mean','O_PM10K_mean','S_PM10K_mean', 'W_PM10K_mean']

temporal = ['year', 'dayofyear', 'month_Apr', 'month_Aug', 'month_Dec', 'month_Feb',
       'month_Jan', 'month_Jul', 'month_Jun', 'month_Mar', 'month_May',
       'month_Nov', 'month_Oct', 'month_Sep', 'weekday_Friday',
       'weekday_Monday', 'weekday_Saturday', 'weekday_Sunday',
       'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday',
       'season_fall', 'season_spring', 'season_summer', 'season_winter',
       'holiday', 'holiday_school']

sites = ['DonBosco','Nord','Ost','Sud','West']

satelit = ['Temperature_Air_2m_Mean_24h', 'Cloud_Cover_Mean',
       'Wind_Speed_10m_Mean', 'Dew_Point_Temperature_2m_Mean',
       'Vapour_Pressure_Mean']


# sklearn support
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


# visualization
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

# metrics import
#from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, r2_score
#from eli5.sklearn import PermutationImportance
#from eli5 import explain_weights_df