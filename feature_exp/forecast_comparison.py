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
featL = pd.read_csv(fName,compression="gzip")
xL = ['object_distance','brake_pressure','force_lon','wheel_speed','vehicle_ping','rtp_lost','rtp_late','modem_rtt','modem_tx','camera_jitter','room_ram','room_cpu','vehicle_ram','vehicle_cpu']
xL = ['object_distance','force_lat','steering_wheel','wheel_speed','vehicle_ping','rtp_lost','rtp_late','modem_tx','vehicle_ram','vehicle_cpu']
yL = ['camera_latency','joystick_latency','e2e_latency']
tL = xL + yL
featL.loc[:,"from_peak"] = featL['from_peak'].apply(lambda x:  int( x*10 )/10. )
def transform(X):
    setL = X['from_peak'].abs() < 16
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

if False: # forecast
    print('----------------------forecast-model-comparison----------------------------')
    conf = a_f.standardConf()
    n = 16
    tL = xL + ['joystick_latency']
    y = serAv['camera_latency']
    X = serAv[tL]
    t = serAv['from_peak']
    yh, Xh, th = y[:-n], X[:-n], t[:-n]
    importlib.reload(a_f)
    sF = a_f.forecast(t,X,y)
    sF.mlp_regressor()
    sF.prophet()
    sF.arima()
    ser = serAv.copy()#[serAv['from_peak'].abs() < 16]
    foreL = [];  peakL = []
    for x in ser['from_peak']:
        print(x)
        g1 = ser[ser['from_peak'] > x].index
        if len(g1) == 0: break
        g1 = g1[:6]
        yf, Xf, tf = ser.loc[g1,'camera_latency'], ser.loc[g1,tL], ser.loc[g1,'from_peak']    
        y_mlp = sF.mlp_forecast(tf,Xf)
        y_pro = sF.prophet_forecast(tf,Xf)
        y_arm = sF.arima_forecast(tf,Xf)
        #y_lst = sF.lstm_forecast(tf,Xf)
        peak = {"series":"average","from_peak":x,"spike_mlp":max(y_mlp),"spike_pro":max(y_pro['yhat']),"spike_arm":max(y_arm)}
        peakL.append(peak)
    peakS = pd.DataFrame(peakL)

    plt.plot(ser['from_peak'],ser['camera_latency'],label="average",linewidth=3)
    plt.plot(peakS['from_peak'],peakS['spike_mlp'],"--",label="mlp")
    plt.plot(peakS['from_peak'],peakS['spike_pro'],"--",label="prophet")
    plt.plot(peakS['from_peak'],peakS['spike_arm'],"--",label="arima")
    plt.legend()
    plt.xlabel("seconds from peak")
    plt.ylabel("latency normalized")
    plt.show()
    

    plt.plot(ser['from_peak'],ser['camera_latency'],label="real",linewidth=2)
    sL = list(range(-12,2,2))
    for i, j in enumerate(sL):
        c = t_v.colorL[i]
        g1 = ser[ser['from_peak'] > j].index
        if len(g1) == 0: break
        g = g1[:6]
        yf, Xf, tf = ser.loc[g1,'camera_latency'], ser.loc[g1,tL], ser.loc[g1,'from_peak']    
        y_mlp = sF.mlp_forecast(tf,Xf)
        y_pro = sF.prophet_forecast(tf,Xf)
        y_arm = sF.arima_forecast(tf,Xf)
        plt.plot(tf,y_mlp,"-.",label="mlp " + str(j),color=c)
        plt.plot(tf,y_pro["yhat"],"--",label="prophet " + str(j),color=c)
        plt.plot(tf,y_arm,".-",label="arima " + str(j),color=c)
    plt.xlabel("seconds from peak")
    plt.ylabel("normalized latency")
    plt.legend()
    plt.show()


    sF.mlp_regressor()
    sF.prophet()
    sF.arima()
    yh, Xh, th = y[:-n], X[:-n], t[:-n]
    yf, Xf, tf = y[-n:], X[-n:], t[-n:]
    y_mlp = sF.mlp_forecast(tf,Xf)
    y_pro = sF.prophet_forecast(tf,Xf)
    print(y_pro['yhat'])
    y_arm = sF.arima_forecast(tf,Xf)
    plt.plot(th,yh,marker='o',linewidth=2,color="red",label="historical data")
    plt.plot(t,y,linewidth=1,color="red",label="real data")
    #plt.fill_between(tf,y_pro['yhat_upper'],y_pro['yhat_lower'],color="#00004040",linestyle="--",label="confidence")
    plt.plot(tf,y_pro['yhat'],label="prophet")
    plt.plot(tf,y_mlp,label="MLP regressor")
    plt.plot(tf,y_arm,label="arima")
    plt.legend()
    plt.xticks(rotation=15)
    plt.show()

    


if False:
    print('-------------------------------distinguish-components------------------------')
    import statsmodels.api as sm
    dta = sm.datasets.co2.load_pandas().data
    dta.co2.interpolate(inplace=True)
    res = sm.tsa.seasonal_decompose(dta.co2)
    res.plot()
    plt.show()
    
    today = datetime.datetime.today().replace(microsecond=0)
    serAv.index = [today + datetime.timedelta(seconds=x) for x in serAv['from_peak']]
    serAv.index.freq = pd.infer_freq(serAv.index)
    res = seasonal_decompose(serAv["camera_latency"],model="add")
    res.plot()
    plt.show()

    
