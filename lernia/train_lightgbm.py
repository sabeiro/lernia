"""
train_model:
iterate and tune over different models for prediction and regression
"""

import random, json, datetime, re, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import lernia.train_reshape as t_r
import lernia.train_modelList as modL
from lernia.train_model import trainMod
import lernia.train_score as t_s
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
import sklearn as sk

class trainGbm(trainMod):
    """
    implementation on lightgbm
    Args:
        X: Data set for prediction
        y: reference data

    """
    def __init__(self,X,y):
        self.__init__(X,y)
        self.param = {"early_stopping_rounds":20,"eval_metric":'auc','eval_names':['valid'],'verbose':100,'categorical_feature':'auto'}
        self.param_tune = {'max_depth': sp_randint(10,50),'num_leaves': sp_randint(6, 50), 'learning_rate ': [0.1,0.01,0.001],'min_child_samples': sp_randint(100, 500),'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],'subsample': sp_uniform(loc=0.2, scale=0.8), 'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}


    def tuneParam(self):
        """tune params"""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=0)
        classifier = lgb.LGBMClassifier(random_state=300, silent=True, metric='None', n_jobs=4, n_estimators=5000)
        find_param = RandomizedSearchCV(estimator=classifier, param_distributions=self.param_tune, n_iter=100, scoring='roc_auc', cv=5,refit=True,random_state=300, verbose=False)
        find_param.fit(X_train, y_train, **param)
        print('Best score : {} with parameters: {} '.format(find_param.best_score_, find_param.best_params_))
        best_param = find_param.best_params_
        return best_param

    def fit(param):
        """fit the model"""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=0)
        param['eval_set'] = [(X_test,y_test)]
        model = lgb.LGBMClassifier(**param)
        model.set_params(**best_param)
        model.fit(X_train, y_train, group=query_train,
                eval_set=[(X_test, y_test)], eval_group=[query_val],
                eval_at=[5, 10, 20], early_stopping_rounds=50)
        self.model = model
        return model

    def predict(X_test):
        """run a prediction"""
        y_pred = self.model.predict(X_test)
        
