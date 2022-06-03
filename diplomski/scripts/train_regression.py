import sys
import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('..')
from src import *
from src.preprocessing import prep_data,split_data,prep_data_satelite,traffic_for_station
from src.model_support import best_parameters


# load data
graz_data = pd.read_csv('../data/data_train_reg.csv',index_col=0)
# prepare data
PM = graz_data[pollutants]
rest_data = graz_data.drop(columns=pollutants,axis=1)

models = {'nn':MLPRegressor,
          'rf':RandomForestRegressor}
print('local data')
# local data i temporal
results_collector_dict = {}
for varname,site in zip(pollutants,sites):
    dictionar_par = {}
    print(f'Start {varname} {site}')
    for key in models.keys():
        print(varname)
        #selecting data for specific site
        X,Y = prep_data(varname,site,PM,rest_data)
        #spliting data into train and test
        X_train,X_test,Y_train,Y_test = split_data(X,Y)
        #best parameters for models
        values = best_parameters(key,X_train,Y_train,10)
        dictionar_par.update({key:values})
        print(f'Done {key}')
    results_collector_dict.update({varname: dictionar_par})
    with open(f'../results/local_parameters_{varname}.pickle', 'wb') as handle:
        pickle.dump(results_collector_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Done {varname}')
# satelite data i temporal

satelit_data = pd.read_csv('../data/satelit_data.csv',index_col=0)
print('satelite data')
results_collector_dict = {}
for varname,site in zip(pollutants,sites):
    dictionar_par = {}
    print(f'Start {varname} {site}')
    for key in models.keys():
        print(varname)
        #selecting data for specific site
        X,Y = prep_data(varname,site,PM,rest_data)
        #adding satelite data
        X = prep_data_satelite(X, site, satelit_data)
        #spliting data into train and test
        X_train,X_test,Y_train,Y_test = split_data(X,Y)
        #best parameters for models
        values = best_parameters(key,X_train,Y_train,10)
        dictionar_par.update({key:values})
        print(f'Done {key}')
    results_collector_dict.update({varname: dictionar_par})
    with open(
            f'../results/satelite_parameters_{varname}.pickle'
            , 'wb') as handle:
        pickle.dump(results_collector_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Done {varname}')
# all data temporal,local,satelite
print('all data')
results_collector_dict = {}
for varname,site in zip(pollutants,sites):
    dictionar_par = {}
    print(f'Start {varname} {site}')
    for key in models.keys():
        print(varname)
        #selecting data for specific site
        X,Y = prep_data(varname,site,PM,rest_data)
        #adding satelite data
        X = pd.concat([X,satelit_data],axis=1)
        #spliting data into train and test
        X_train,X_test,Y_train,Y_test = split_data(X,Y)
        #best parameters for models
        values = best_parameters(key,X_train,Y_train,10)
        dictionar_par.update({key:values})
        print(f'Done {key}')
    results_collector_dict.update({varname: dictionar_par})
    with open(
            f'../results/all_parameters_{varname}.pickle'
            , 'wb') as handle:
        pickle.dump(results_collector_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Done {varname}')
# traffic, local and temporal

traffic_data = pd.read_csv('../data/promet.csv',index_col=0)

results_collector_dict = {}
print('local and traffic data')
for varname,site in zip(pollutants,sites):
    dictionar_par = {}
    print(f'Start {varname} {site}')
    for key in models.keys():
        print(varname)
        #selecting data for specific site
        X,Y = prep_data(varname,site,PM,rest_data)
        #adding traffic data
        traffic_site = traffic_for_station(site,traffic_data)
        X = pd.concat([X,traffic_site],axis=1)
        #spliting data into train and test
        X_train,X_test,Y_train,Y_test = split_data(X,Y)
        #best parameters for models
        values = best_parameters(key,X_train,Y_train,10)
        dictionar_par.update({key:values})
        print(f'Done {key}')
    results_collector_dict.update({varname: dictionar_par})
    with open(
            f'../results/local_traffic_parameters_{varname}.pickle'
            , 'wb') as handle:
        pickle.dump(results_collector_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Done {varname}')
#satelite, temporal and traffic

print('satelite and traffic data')
results_collector_dict = {}
for varname,site in zip(pollutants,sites):
    dictionar_par = {}
    print(f'Start {varname} {site}')
    for key in models.keys():
        print(varname)
        #selecting data for specific site
        X,Y = prep_data(varname,site,PM,rest_data)
        #adding satelite data
        X = prep_data_satelite(X, site, satelit_data)
        # adding traffic data
        traffic_site = traffic_for_station(site, traffic_data)
        X = pd.concat([X, traffic_site], axis=1)
        #spliting data into train and test
        X_train,X_test,Y_train,Y_test = split_data(X,Y)
        #best parameters for models
        values = best_parameters(key,X_train,Y_train,10)
        dictionar_par.update({key:values})
        print(f'Done {key}')
    results_collector_dict.update({varname: dictionar_par})
    with open(
            f'../results/satelite_traffic_parameters_{varname}.pickle'
            , 'wb') as handle:
        pickle.dump(results_collector_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Done {varname}')
# traffic,satelit,local and temporal
print('all traffic')
results_collector_dict = {}
for varname,site in zip(pollutants,sites):
    dictionar_par = {}
    print(f'Start {varname} {site}')
    for key in models.keys():
        print(varname)
        #selecting data for specific site
        X,Y = prep_data(varname,site,PM,rest_data)

        # adding traffic data
        traffic_site = traffic_for_station(site, traffic_data)
        X = pd.concat([X, traffic_site,satelit_data], axis=1)
        #spliting data into train and test
        X_train,X_test,Y_train,Y_test = split_data(X,Y)
        #best parameters for models
        values = best_parameters(key,X_train,Y_train,10)
        dictionar_par.update({key:values})
        print(f'Done {key}')
    results_collector_dict.update({varname: dictionar_par})
    with open(
            f'../results/all_traffic_parameters_{varname}.pickle'
            , 'wb') as handle:
        pickle.dump(results_collector_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Done {varname}')
































