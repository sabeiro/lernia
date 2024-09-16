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
import lernia.train_score as t_s
import deep_lernia.train_keras as t_k
import deep_lernia.train_longShort as tls
import importlib


folder = "august"
folder = "modem"
featL = pd.read_csv(baseDir + "/rem/raw/spike_"+folder+".csv.gz",compression="gzip")
featL.loc[:,"from_peak"] = featL['from_peak'].apply(lambda x:  int( x*10 )/10. )
featL = featL.sort_values(['series','from_peak'])
xL = ['object_distance','brake_pressure','force_lon','wheel_speed','vehicle_ping','rtp_lost','rtp_late','modem_rtt','modem_tx','camera_jitter','room_ram','room_cpu','vehicle_ram','vehicle_cpu']
xL = ['object_distance','force_lat','steering_wheel','wheel_speed','vehicle_ping','rtp_lost','rtp_late','modem_tx','vehicle_ram','vehicle_cpu']
xL = [x for x in featL.columns if re.search("_si",x)]# + [x for x in featL.columns if re.search("_rss",x)]
xL = xL + ['object_distance','brake_pressure','force_lon','wheel_speed','vehicle_ping','camera_jitter','room_ram','room_cpu','vehicle_ram','vehicle_cpu']
yL = ['camera_latency','joystick_latency']
tL = yL + xL
print('-----------------------norm----------------------------')
importlib.reload(t_r)
setL = featL['from_peak'].abs() < 160
feat = featL.loc[setL]
#feat = t_r.diffDf(feat,tL=tL)
feat, normD = t_r.normDf(feat,perc=[2,98],lim=[-1,1],tL=tL)
normD.to_csv(baseDir + "/rem/train/norm.csv")
serAv = feat.groupby('from_peak').agg(np.nanmean).reset_index()
n_in = 1
t_v.plotTimeSeries(feat[tL],t=feat['from_peak'],mode="binned")
plt.show()


if False:
    importlib.reload(t_k)
    importlib.reload(tls)
    tK = tls.timeSeries(feat[tL])
    #tK.scale(feature_range=(-1,1))
    if True:
        #tK.lstmBatch(batch_size=1,neurons=6)
        tK.longShortDeep(batch_size=1,neurons=6,n_in=n_in)
    else:
        tK.loadModel(baseDir + "rem/train/lstm_camera_tmp")
        tK.model.compile(loss='mean_squared_error', optimizer='adam')
    kpi = tK.train(batch_size=1,nb_epoch=10,portion=.8,shuffle=True,n_in=n_in)
    tK.setX(serAv[tL]); #tK.scale()
    for i in range(20):
        kpi = tK.train(batch_size=1,nb_epoch=20,portion=.8,shuffle=True,n_in=n_in)

    serL = np.unique(feat['series'])
    for i in range(500):
        s = np.random.choice(serL)
        g = feat[feat['series'] == s]
        tK.setX(g[tL]); #tK.scale()
        kpi = tK.train(batch_size=1,nb_epoch=20,portion=.8,shuffle=True,n_in=n_in)

    for i in range(500):
        s1 = np.random.choice(serL)
        s2 = np.random.choice(serL)
        g1 = feat[feat['series'] == s1]
        g2 = feat[feat['series'] == s2]
        X_train, X_test = g1[tL].values, g2[tL].values
        tK.setX(X_train); #tK.scale()
        kpi = tK.trainCross(X_test,batch_size=1,nb_epoch=20,n_in=n_in)
                                
    tK.setX(serAv[tL]); #tK.scale()
    for i in range(20):
        kpi = tK.train(batch_size=1,nb_epoch=20,portion=.8,shuffle=True,n_in=n_in)

    # importlib.reload(t_k)
    # importlib.reload(tls)
    # tK = tls.timeSeries(serAv[tL])
    # tK.loadModel(baseDir + "rem/train/lstm_camera_tmp")
    # tK.model.compile(loss='mean_squared_error', optimizer='adam')
    g = serAv
    X = g[tL].values
    y_fore = tK.forecast(X,n_in=n_in)
    y_pred = tK.predict(X,n_in=n_in)
    plt.plot(g['from_peak'],X[:,0],label="latency camera",linewidth=2)
    plt.plot(g['from_peak'],y_pred[:,0],label="prediction camera")
    plt.plot(g['from_peak'],y_fore[:,0],label="forecast camera")
    plt.legend()
    plt.xlabel("seconds to peak")
    plt.ylabel("normalized latency")
    plt.show()
    tK.plotHistory()
    plt.show()
    
    if False:
        tK.saveModel(baseDir + "rem/train/lstm_camera_fore")
        tK.saveModel(baseDir + "rem/train/lstm_camera_sec")
        tK.saveModel(baseDir + "rem/train/lstm_camera_single")
        tK.saveModel(baseDir + "rem/train/lstm_camera_deep")
        tK.saveModel(baseDir + "rem/train/lstm_camera_modem")
        # tK.saveModel(baseDir + "rem/train/lstm_camera_diff")
        # tK.saveModel(baseDir + "rem/train/lstm_camera_in2")
        tK.saveModel(baseDir + "rem/train/lstm_camera_tmp")

if False:
    print('------------------------------forecast-from-peak----------------------------')
    if False:
        importlib.reload(t_k)
        importlib.reload(tls)
        tK = tls.timeSeries(feat[tL])
        tK.loadModel(baseDir + "rem/train/lstm_camera_tmp")
        tK.model.compile(loss='mean_squared_error', optimizer='adam')

    def forePeak(g1):
        g = g1[:6]
        X = g[tL].values
        y_fore = tK.forecast(X,n_in=n_in)
        t = g['from_peak'].values
        max_camera = t[np.argmax(y_fore[:,0])]
        max_joystick = t[np.argmax(y_fore[:,1])]
        peak = {"series":i,"from_peak":x
                ,"spike_camera":max(y_fore[:,0]),"spike_joystick":max(y_fore[:,1]),"max_camera":max_camera,"max_joystick":max_joystick}
        fore = pd.DataFrame({"start":x,"joystick":X[:,0],"camera":X[:,0],"fore_camera":y_fore[:,0],"from_peak":g['from_peak'],"series":i})
        return peak, fore
    
    importlib.reload(t_k)
    importlib.reload(tls)
    tK = tls.timeSeries(feat[tL])
    tK.loadModel(baseDir + "rem/train/lstm_camera_tmp")
    tK.model.compile(loss='mean_squared_error', optimizer='adam')
    foreL = [];  peakL = []
    ser = serAv.copy()#[serAv['from_peak'].abs() < 16]
    for x in serAv['from_peak']:
        print(x)
        g = serAv[serAv['from_peak'] > x]
        if len(g) == (n_in-1): break
        peak, fore = forePeak(g)
        peakL.append(peak)
        foreL.append(fore)
    foreS = pd.concat(foreL)
    peakS = pd.DataFrame(peakL)

    X = serAv[tL]
    plt.plot(ser['from_peak'],X['camera_latency'],label="real",linewidth=3)
    plt.plot(peakS['from_peak'],peakS['spike_camera'],"--",label="peak forecast")
    plt.plot(peakS['max_camera'],peakS['spike_camera'],"--",label="peak forecast time")
    plt.legend()
    plt.xlabel("seconds from peak")
    plt.ylabel("latency normalized")
    plt.show()

    g = serAv
    X = g[tL].values
    y_fore = tK.forecast(X,n_in=n_in)
    y_pred = tK.predict(X,n_in=n_in)
    plt.plot(g['from_peak'],g['camera_latency'],label="real",linewidth=2)
    plt.plot(g['from_peak'],y_pred[:,0],".-",label="prediction",color="green")
    sL = [-2,-1,-0.8,-0.5,-0.1,0]
    sL = [-6,-5,-4,-3,-2,-1,0,1]
    sL = list(range(-30,10,4))
    for i, j in enumerate(sL):
        c = t_v.colorL[i]
        g1 = ser[ser['from_peak'] > j]
        g1 = g1[:12]
        X = g1[tL]
        y_fore = tK.forecast(X)
        plt.plot(g1['from_peak'],y_fore[:,0],"--",label="forecast " + str(j),color=c)
    plt.xlabel("seconds from peak")
    plt.ylabel("normalized latency")
    plt.legend()
    plt.show()

if False:
    print('-----------------------------show-prediction-------------------------')
    fig, ax = plt.subplots(2,3)
    ax = ax.flatten()
    for a in ax:
        s = np.random.choice(serL)
        g = feat[feat['series'] == s]
        n = 0#int(len(g)*np.random.random())
        X = g[tL][n:]
        y = g['camera_latency'][n:]
        t = g['from_peak'][n:]
        y_pred = tK.predict(X.values,n_in=n_in)
        a.set_title(s)
        a.plot(g['from_peak'],y,label="real")
        a.plot(t,y_pred[:,0],label="forecast")
        a.legend()
    plt.show()

    spikeL = []
    for i,g in featL.groupby('series'):
        y = s_s.interpMissing(g['camera_latency'])
        X = s_s.interpMissingMatrix(g[xL + ['camera_latency']])
        tK.setX(X), tK.setY(y)
        tK.scale()
        y_pred = tK.predict(tK.reshape(tK.X.values))
        sp1 =  1.*(tK.y[:,0] > 0.5)
        sp2 =  1.*(y_pred[:,0] > 0.5)
        spi = pd.DataFrame({"real":sp1,"pred":sp2,"series":s})
        spikeL.append(spi)

    spikeL = pd.concat(spikeL)
    cm = t_v.plotConfMat(spikeL['real'],spikeL['pred'])
    t_v.plotHeatmap(cm/cm.sum())
    plt.title("confusion matrix")
    plt.show()

if False:
    print('---------------------------feature-knock-out-------------------------------')
    importlib.reload(t_k)
    importlib.reload(tls)
    tK = tls.timeSeries(serAv[tL])
    tK.loadModel(baseDir + "rem/train/lstm_camera_tmp")
    tK.model.compile(loss='mean_squared_error', optimizer='adam')
    perfL = tK.featKnockOut(portion=0.9,mode="forecast",shuffle=True)
    tK.plotFeatImportance(perfL)
    plt.show()
    
#{name=Cellular 1, message=Connected to BleibGesund, ip=100.89.14.68, status_led=green, uptime=107, carrier_name=BleibGesund, carrier_country=Germany, band=LTE Band 7 (2600 MHz), rssi=-75, sinr=13.2, rsrp=-103, rsrq=-9, level=0, cell_id=55559, tunnel_state=ACTIVE, tunnel_active=true, tunnel_uptime_sec=224, tunnel_uptime_ns=463051382, tunnel_rtt=58, tunnel_rx=785989, tunnel_tx=823246, tunnel_loss=0, tunnel_delta_valid=true, tunnel_delta_interval_sec=1, tunnel_delta_interval_ns=31814877, tunnel_delta_loss=0, tunnel_delta_tx=97089, tunnel_delta_rx=93134}
