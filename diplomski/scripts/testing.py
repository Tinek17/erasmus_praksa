import sys
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
sys.path.append('..')
from src import *
from src import *
from src.model_support import parameter, ret_params, features_for_models
from src.preprocessing import prep_data, split_data, prep_data_satelite, traffic_for_station,split_data_nn_of_rf
import warnings
from src.preprocessing import prep_data,split_data,prep_data_satelite,traffic_for_station
from src.model_support import best_parameters
from sklearn.preprocessing import StandardScaler

# load data
graz_data = pd.read_csv('../data/data_train_reg.csv',index_col=0)
# prepare data
PM = graz_data[pollutants]
rest_data = graz_data.drop(columns=pollutants,axis=1)

random_states = [1,
                2,
                3,4,5
                ]

models = {'nn':MLPRegressor,
          'rf':RandomForestRegressor}

traffic_data = pd.read_csv('../data/promet.csv',index_col=0)

satelit_data = pd.read_csv('../data/satelit_data.csv',index_col=0)

file = ['local_parameters_D_PM10K_mean','local_parameters_N_PM10K_mean','local_parameters_O_PM10K_mean','local_parameters_S_PM10K_mean','local_parameters_W_PM10K_mean']

for i in file:
    infile = open(f'../results/{i}.pickle','rb')
    new_dict = pickle.load(infile)

all_results = pd.DataFrame()

for varname,site in zip(pollutants,sites):
    print(f'Start {varname}')
    for algoritam  in models.keys():
        print(f'Start {algoritam}')
        for random in random_states:
            print(f'Start random state {random}')
            # function for getting best parameters
            model_params = parameter(varname,algoritam,new_dict)
            # function for best features
            features = features_for_models(varname,algoritam,new_dict)
            # prepairing temporal and local data
            X, Y = prep_data(varname, site, PM, rest_data)
            # split on traing and test
            X_train, X_test, Y_train, Y_test = split_data(X, Y)
            X_train,X_test = split_data_nn_of_rf(X_train,X_test,algoritam)
            # fitting parameters
            clean_params = ret_params(model_params, algoritam,random)
            # fitting model with parameters
            model = models[algoritam](**clean_params)
            # fitting model with data
            model.fit(X_train[features], Y_train)
            # predictions
            predictions = model.predict(X_test[features])
            # saving predctions
            y_predicted = pd.Series(model.predict(X_test[features]),
                                index=Y_test.index,
                                name=varname + '_' + algoritam +'_PRED'+'_random_state_'+str(random)).sort_index()
            if random == 1:
                all_results = pd.concat([all_results, y_predicted, Y_test], axis=1)
            else:
                all_results = pd.concat([all_results, y_predicted], axis=1)
            print(f'Done random state {random}')
        print(f'Done {algoritam}')
    print(f'Done {varname}')
# save results
all_results.to_csv('../results/rezultati_local.csv')