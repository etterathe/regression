# This file is just a snipet of code from jupyter notebook presenting just a modelling stage
# Because of that there are 2 missing arrays: x_train and y_train - to use this snipet user need to intialize those 2 manually
# But its highly recommended to use notebook instead, where the full code is presented

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError

pipelines = {
    'ridge':      make_pipeline(Ridge(random_state = 1234)),
    'lasso':      make_pipeline(Lasso(random_state = 1234)),
    'elasticnet': make_pipeline(ElasticNet(random_state = 1234)),
    'rf':         make_pipeline(RandomForestRegressor(random_state = 1234)),
    'gb':         make_pipeline(GradientBoostingRegressor(random_state = 1234))
}

hypergrid = {
    'ridge':{
        'ridge__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.99]
    },
    'lasso':{
        'lasso__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.99]
    },
    'elasticnet':{
        'elasticnet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.99]
    },
    'rf':{
        'randomforestregressor__min_samples_split':[2,4,6],
        'randomforestregressor__min_samples_leaf':[1,2,3]
    },  
    'gb':{
        'gradientboostingregressor__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.99]
    }
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hypergrid[algo], cv=10, n_jobs=-1)
    try:
        print(f'Training model: {algo} ')
        model.fit(x_train, y_train)
        fit_models[algo] = model
        print(f'{algo} model training finished')
    except NotFittedError as e:
        print(repr(e))