from src import *
from src.model_support import parameter, ret_params, features_for_models
from src.preprocessing import prep_data, split_data, prep_data_satelite, traffic_for_station,split_data_nn_of_rf
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=FutureWarning)

random_states = [1,
                2,
                3,4,5
                ]

models = {'nn': MLPRegressor,
          'rf': RandomForestRegressor}
sc = StandardScaler()
# load data
graz_data = pd.read_csv('../data/data_train_reg.csv',index_col=0)

# prepare data
PM = graz_data[pollutants]
rest_data = graz_data.drop(columns=pollutants,axis=1)

# temporal and local weather data

# opening pickle and using features and hyperparameters from there
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

# satelit data and temporal

# opening pickle and using features and hyperparameters from there
file = ['satelite_parameters_D_PM10K_mean','satelite_parameters_N_PM10K_mean','satelite_parameters_O_PM10K_mean','satelite_parameters_S_PM10K_mean','satelite_parameters_W_PM10K_mean']

for i in file:
    infile = open(f'../results/{i}.pickle','rb')
    new_dict = pickle.load(infile)

all_results = pd.DataFrame()


satelit_data = pd.read_csv('../data/satelit_data.csv',index_col=0)

for varname,site in zip(pollutants,sites):
    print(f'Start {varname}')
    for algoritam in models.keys():
        print(f'Start {algoritam}')
        for random in random_states:
            print(f'Start random state {random}')
            model_params = parameter(varname,algoritam,new_dict)
            features = features_for_models(varname, algoritam, new_dict)
            X, Y = prep_data(varname, site, PM, rest_data)
            X = prep_data_satelite(X, site, satelit_data)
            X_train, X_test, Y_train, Y_test = split_data(X, Y)
            X_train, X_test = split_data_nn_of_rf(X_train, X_test, algoritam)
            clean_params = ret_params(model_params, algoritam,random)
            model = models[algoritam](**clean_params)
            model.fit(X_train[features], Y_train)
            predictions = model.predict(X_test[features])

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

all_results.to_csv('../results/rezultati_satelit.csv')

# local, temporal and satelite data

# opening pickle and using features and hyperparameters from there
file = ['all_parameters_D_PM10K_mean','all_parameters_N_PM10K_mean','all_parameters_O_PM10K_mean','all_parameters_S_PM10K_mean','all_parameters_W_PM10K_mean']

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
            model_params = parameter(varname,algoritam,new_dict)
            features = features_for_models(varname, algoritam, new_dict)
            X, Y = prep_data(varname, site, PM, rest_data)
            X = pd.concat([X,satelit_data],axis=1)
            X_train, X_test, Y_train, Y_test = split_data(X, Y)
            X_train, X_test = split_data_nn_of_rf(X_train, X_test, algoritam)
            clean_params = ret_params(model_params, algoritam,random)
            model = models[algoritam](**clean_params)
            model.fit(X_train[features], Y_train)
            predictions = model.predict(X_test[features])

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

all_results.to_csv('../results/rezultati_all_meteo.csv')

# local,temporal and traffic
traffic_data = pd.read_csv('../data/promet.csv',index_col=0)
file = ['local_traffic_parameters_D_PM10K_mean','local_traffic_parameters_N_PM10K_mean','local_traffic_parameters_O_PM10K_mean','local_traffic_parameters_S_PM10K_mean','local_traffic_parameters_W_PM10K_mean']


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
            model_params = parameter(varname,algoritam,new_dict)
            features = features_for_models(varname, algoritam, new_dict)
            X, Y = prep_data(varname, site, PM, rest_data)
            traffic_site = traffic_for_station(site, traffic_data)
            X = pd.concat([X, traffic_site], axis=1)
            X_train, X_test, Y_train, Y_test = split_data(X, Y)
            X_train, X_test = split_data_nn_of_rf(X_train, X_test, algoritam)
            clean_params = ret_params(model_params, algoritam,random)
            model = models[algoritam](**clean_params)
            model.fit(X_train[features], Y_train)
            predictions = model.predict(X_test[features])

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

all_results.to_csv('../results/rezultati_local_traffic.csv')

# satelite,temporal and traffic

file = ['satelite_traffic_parameters_D_PM10K_mean','satelite_traffic_parameters_N_PM10K_mean','satelite_traffic_parameters_O_PM10K_mean','satelite_traffic_parameters_S_PM10K_mean','satelite_traffic_parameters_W_PM10K_mean']


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
            model_params = parameter(varname,algoritam,new_dict)
            features = features_for_models(varname, algoritam, new_dict)
            X, Y = prep_data(varname, site, PM, rest_data)
            X = prep_data_satelite(X, site, satelit_data)
            traffic_site = traffic_for_station(site, traffic_data)
            X = pd.concat([X, traffic_site], axis=1)
            X_train, X_test, Y_train, Y_test = split_data(X, Y)
            X_train, X_test = split_data_nn_of_rf(X_train, X_test, algoritam)
            clean_params = ret_params(model_params, algoritam,random)
            model = models[algoritam](**clean_params)
            model.fit(X_train[features], Y_train)
            predictions = model.predict(X_test[features])

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

all_results.to_csv('../results/rezultati_satelite_traffic.csv')

#temporal,local,traffic and satelite

file = ['all_traffic_parameters_D_PM10K_mean','all_traffic_parameters_N_PM10K_mean','all_traffic_parameters_O_PM10K_mean','all_traffic_parameters_S_PM10K_mean','all_traffic_parameters_W_PM10K_mean']


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
            model_params = parameter(varname,algoritam,new_dict)
            features = features_for_models(varname, algoritam, new_dict)
            X, Y = prep_data(varname, site, PM, rest_data)

            traffic_site = traffic_for_station(site, traffic_data)
            X = pd.concat([X, traffic_site,satelit_data], axis=1)
            X_train, X_test, Y_train, Y_test = split_data(X, Y)
            X_train, X_test = split_data_nn_of_rf(X_train, X_test, algoritam)
            clean_params = ret_params(model_params, algoritam,random)
            model = models[algoritam](**clean_params)
            model.fit(X_train[features], Y_train)
            predictions = model.predict(X_test[features])

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

all_results.to_csv('../results/rezultati_all_traffic.csv')