import os, sys, gzip, random, json, datetime, re, io
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import xgboost
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose

dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = os.environ['LAV_DIR']
import albio.forecast as a_f
import albio.series_forecast as s_f
import albio.series_stat as s_s
import lernia.train_reshape as t_r
import lernia.train_score as tsc
import lernia.train_viz as t_v
import lernia.train_model as t_m
import deep_lernia.train_keras as t_k
import deep_lernia.train_longShort as tls
import importlib

fName = baseDir + "rem/raw/telemetry_1sec.csv.gz"
fName = baseDir + "rem/raw/spike_list.csv.gz"
fName = baseDir + "rem/raw/spike_deci.csv.gz"
featL = pd.read_csv(fName,compression="gzip")

xL = ['object_distance','brake_pressure','force_lon','wheel_speed','vehicle_ping','rtp_lost','rtp_late','modem_rtt','modem_tx','camera_jitter','room_ram','room_cpu','vehicle_ram','vehicle_cpu']
xL = ['object_distance','force_lat','steering_wheel','wheel_speed','vehicle_ping','rtp_lost','rtp_late','modem_tx','vehicle_ram','vehicle_cpu']
#xL = ['force_lon','wheel_speed','camera_jitter','bytes-Cellular 1','bytes-Cellular 2','bytes-Cellular 3','bytes-Cellular 4']
yL = ['camera_latency','joystick_latency']
tL = xL + yL
featL.loc[:,"from_peak"] = featL['from_peak'].apply(lambda x:  int( x*10 )/10. )
def transform(X):
    setL = X['from_peak'].abs() < 160
    for t in tL:
        X.loc[:,t] = s_s.interpMissing(X[t])
        X.loc[:,t] = t_r.normPercentile(X[t],perc=[1,99])
        X.loc[:,t] = s_s.interpMissing(X[t])
    X = X.loc[setL,:]
    return X
    
yL = ['joystick_latency','camera_latency']
tL = yL + xL
feat = transform(featL.copy())
serAv = feat.groupby('from_peak').agg(np.nanmean).reset_index()
t_v.plotTimeSeries(serAv[tL],t=serAv['from_peak'])
plt.show()

if False:
    print('-----------------------xgboost-----------------------')
    from xgboost import XGBRegressor
    import xgboost as xgb
    tL = yL + xL
    n_train = 60
    X, y = serAv[tL], serAv['from_peak']
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X)
    plt.scatter(y_pred,y)
    y_pred = model.predict(X_test)
    print(tsc.calcMetrics(y_pred,y_test))
    plt.scatter(y_pred,y_test)
    plt.show()

    importlib.reload(t_m)
    params = {'alpha': 0.15000000000000002,'eta': 0.02,'eval_metric': 'cox-nloglik','gamma': 0.26,'lambda': 1.4000000000000001,'max_depth': 9,'min_child_weight': 4,'objective': 'survival:cox','subsample': 0.8}
    serL = np.unique(featL['series'])
    def xgb_fit(X_train,y_train,X_test,y_test,params):
        """prepare the data sets and train"""
        dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)
        dtest = xgb.DMatrix(X_test, label=y_test, nthread=-1)
        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(params,dtrain,num_boost_round=100,evals=watchlist,early_stopping_rounds=5,verbose_eval=0)
        y_pred = model.predict(dtest)
        return y_pred, model
    
    def xgb_train(featL,serL,params):
        """train on one series and predict on another"""
        s1 = np.random.choice(serL)
        s2 = np.random.choice(serL)
        g1 = featL[featL['series'] == s1].groupby('from_peak').agg(np.mean)
        g2 = featL[featL['series'] == s2].groupby('from_peak').agg(np.mean)
        X_train, y_train = g1[tL], g1.index
        X_test, y_test = g2[tL], g2.index
        y_pred, model = xgb_fit(X_train,y_train,X_test,y_test,params)
        return y_pred, model
        
    predL = []
    for i in range(1000):
        y_pred, model = xgb_train(featL,serL,params)
        predL.append(pd.DataFrame({i:y_pred}))
    predL = pd.concat(predL,axis=1)

    y_pred = predL.mean(axis=1)
    t = np.unique(featL['from_peak'])
    plt.title("risk prediction on cross validation")
    plt.plot(t,y_pred,label="cross validation")
    n_train = 90
    X, y = serAv[tL], serAv['from_peak']
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    dall = xgb.DMatrix(X, label=y, nthread=-1)
    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)
    dtest = xgb.DMatrix(X_test, label=y_test, nthread=-1)
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(params,dtrain,num_boost_round=100,evals=watchlist,early_stopping_rounds=5,verbose_eval=0)
    y_pred = model.predict(dall)
    plt.plot(serAv['from_peak'],y_pred,label="average")
    plt.xlabel("seconds from spike")
    plt.ylabel("risk value")
    plt.legend()
    plt.show()
        
    plt.scatter(y_test,y_pred)
    plt.show()
    print(tsc.calcMetrics(y_pred,y_test))
    y_pred = model.predict(dall)
    plt.scatter(y,y_pred)
    plt.xlabel("seconds from peak")
    plt.ylabel("risk estimation")
    plt.show()

    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv, check_y_survival
    from sksurv.linear_model.coxph import BreslowEstimator
    from sksurv.nonparametric import CensoringDistributionEstimator, SurvivalFunctionEstimator
    from sklearn.utils import check_consistent_length, check_array
    def target_to_time(x):
        return abs(x) - (x >= 0).astype(int)

    def target_to_event(x):
        return (x >= 0)
    
    def sksurv_format(y):
        return Surv.from_arrays(time=target_to_time(y),event=target_to_event(y))

    y_pred = predL.mean(axis=1)
    y_pred = model.predict(dall)
    y_test = y
    t = target_to_time(y_test)
    va_times = np.arange(min(t), max(t), 1)
    va_auc, va_mean_auc = cumulative_dynamic_auc(
        sksurv_format(y_train),
        sksurv_format(y_test),
        y_pred, va_times
    )
    plt.plot(va_times, va_auc, marker="o")
    plt.axhline(va_mean_auc, linestyle="--")
    plt.xlabel("seconds from spike")
    plt.ylabel("time-dependent AUC")
    plt.grid(True)
    plt.show()

    
