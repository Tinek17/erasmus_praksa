import numpy
import pandas
import pandas as pd
from src import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.inspection import permutation_importance


def parameter(pollutant, algo,new_dict):
    # extracting best parameters for every model
    value_1 = new_dict[pollutant]
    value_2 = value_1[algo]
    value_3 = value_2['best_params']
    return value_3

def features_for_models(pollutant,algo,new_dict):
    # extracting best features for every model
    value_1 = new_dict[pollutant]
    value_2 = value_1[algo]
    value_3 = value_2['feature']
    return value_3


def ret_params(best_params,rg,random_temp):
    # uses best parameters for every model
    if rg == 'rf':
        p = {'n_estimators': int(best_params['n_estimators']),
             'max_depth': int(best_params['max_depth']),
             'n_jobs': int(best_params['n_jobs']),
             'random_state': random_temp,
             'max_samples': float(best_params['max_samples']),
             'min_samples_leaf': int(best_params['min_samples_leaf'])
             }
        return p
    else:
        if type(best_params['hidden_layer_sizes']) == int:
            p = {'hidden_layer_sizes': int(best_params['hidden_layer_sizes']),
                 'learning_rate_init': float(best_params['learning_rate_init']),
                 'solver':str(best_params['solver']),
                 'max_iter': int(best_params['max_iter']),
                 'early_stopping': bool(best_params['early_stopping']),
                 'random_state': random_temp,
                 'alpha': float(best_params['alpha']),

                 }
            return p
        else:
            p = {'hidden_layer_sizes': list(tuple(best_params['hidden_layer_sizes'])),
                'learning_rate_init': float(best_params['learning_rate_init']),
                'solver': str(best_params['solver']),
                 'max_iter': int(best_params['max_iter']),
                'early_stopping': bool(best_params['early_stopping']),
                 'random_state': random_temp,
               'alpha': float(best_params['alpha']),
                 }
            return p

# picking best features and parameters for every model
def best_parameters(model_key,train_features,train_labels,runs):

    temp_dict = {}
    print('hyperparameter search')
    if model_key == 'nn':
        for i in range(runs):
            print(f'--------run {i}--------')
            model = MLPRegressor()
            sc = StandardScaler()
            train_features_scaled = sc.fit_transform(train_features)
            train_features_df = pd.DataFrame(train_features_scaled, columns=train_features.columns)
            params = {
                'hidden_layer_sizes': [(100, 20, 5),
                                        (100,20,10),
                                        (100,100,5),
                                        (100, 100, 10),
                                        (500,100,10),
                                        (500,100,5),
                                        (500,20,10),
                                        (500,20,5),
                                       ],
                 'solver': ['adam'],
                 'learning_rate_init':[0.0001],
                 'max_iter':[300],
                 'early_stopping':[True],
                'random_state': [42],
                 'alpha': [0.0001, 0.05]
                    }
            if i == 0:
                feature = train_features.columns
                mlp_regressor_grid = GridSearchCV(model, param_grid=params, n_jobs=-1, cv=5, verbose=3,return_train_score=True)
                mlp_regressor_grid.fit(train_features_df[feature], train_labels)
                best_params_temp = mlp_regressor_grid.best_params_
                model=MLPRegressor(**best_params_temp).fit(train_features_df[feature],train_labels)
                feature = feature_importance(model,train_features_df[feature],train_labels)
                temp_dict.update({'best_params': mlp_regressor_grid.best_params_})
                temp_dict.update({'feature': feature})

            else:

                mlp_regressor_grid = GridSearchCV(model, param_grid=params, n_jobs=-1, cv=5, verbose=3)
                mlp_regressor_grid.fit(train_features_df[feature], train_labels)
                best_params_temp = mlp_regressor_grid.best_params_
                model = MLPRegressor(**best_params_temp).fit(train_features_df[feature], train_labels)
                feature = feature_importance(model, train_features_df[feature], train_labels)
                temp_dict.update({'best_params': mlp_regressor_grid.best_params_})
                temp_dict.update({'feature': feature})

        return temp_dict
    else:
        for i in range(runs):
            print(f'--------run {i}--------')
            model = RandomForestRegressor()
            param_grid = {
            'n_estimators': [100,200,300],
            'max_depth': [4,6,8,10],
            'n_jobs': [-1],
             'random_state': [42],
            'max_samples' : [0.4,0.45,0.5],
            'min_samples_leaf': [2,5,10]
            }
            if i == 0:
                feature = train_features.columns
                rf_random = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
                rf_random.fit(train_features[feature], train_labels)
                best_params_temp = rf_random.best_params_
                model=RandomForestRegressor(**best_params_temp).fit(train_features[feature],train_labels)
                feature = feature_importance(model,train_features[feature],train_labels)
                temp_dict.update({'best_params': rf_random.best_params_})
                temp_dict.update({'feature': feature})
            else:
                rf_random = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
                rf_random.fit(train_features[feature], train_labels)
                best_params_temp = rf_random.best_params_
                model = RandomForestRegressor(**best_params_temp).fit(train_features[feature], train_labels)
                feature = feature_importance(model, train_features[feature], train_labels)
                temp_dict.update({'best_params': rf_random.best_params_})
                temp_dict.update({'feature': feature})

        return temp_dict

# only pickes best parameters and uses all features
def best_parames_bez_feature(model_key,train_features,train_labels):
    temp_dict = {}
    if model_key == 'nn':
        model = MLPRegressor()
        sc = StandardScaler()
        train_features_scaled = sc.fit_transform(train_features)
        train_features_df = pd.DataFrame(train_features_scaled, columns=train_features.columns)
        params = {
            'hidden_layer_sizes': [(100, 20, 5),
                                    (100,20,10),
                                    (100,100,5),
                                    (100, 100, 10),
                                    (500,100,10),
                                    (500,100,5),
                                    (500,20,10),
                                    (500,20,5),
                                   ],
             'solver': ['adam'],
             'learning_rate_init':[0.0001],
             'max_iter':[300],
             'early_stopping':[True],
            'random_state': [42],
             'alpha': [0.0001, 0.05]
        }
        mlp_regressor_grid = GridSearchCV(model, param_grid=params, n_jobs=-1, cv=3, verbose=3)
        mlp_regressor_grid.fit(train_features_df, train_labels)
        temp_dict.update({'best_params': mlp_regressor_grid.best_params_})
        return temp_dict
    else:
        model = RandomForestRegressor()
        param_grid = {
            'n_estimators': [100, 200, 300],
             'max_depth': [4,6,8,10],
             'n_jobs': [-1],
            'random_state': [42],
             'max_samples' : [0.4,0.45,0.5],
             'min_samples_leaf': [2,5,10]
        }
        rf_random = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        rf_random.fit(train_features, train_labels)
        temp_dict.update({'best_params': rf_random.best_params_})
        return temp_dict


# features importace
def feature_importance(model,train_features_temp,train_labels_temp):
    result = permutation_importance(model, train_features_temp, train_labels_temp, n_repeats=3,
                                    random_state=42, n_jobs=24)
    weights: pd.Series = pd.Series(result['importances_mean'], index=train_features_temp.columns).\
        sort_values(ascending=False)
    one_third_of_feature_count = int(len(train_features_temp) / 3)
    selected_features_list = weights[weights > 0.001].index.tolist()
    if len(selected_features_list) == 0:
        # if there are no pos.weights
        print('error in selection', len(train_features_temp.columns.tolist()), ', or all features selected')
        return train_features_temp.columns.tolist()

    elif len(selected_features_list) > one_third_of_feature_count:
        # reduce to one third
        return selected_features_list[0:one_third_of_feature_count]

    else:
        return selected_features_list

