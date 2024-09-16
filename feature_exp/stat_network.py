import os, sys, gzip, random, json, datetime, re, io
import pandas as pd
import numpy as np
from scipy import stats as st
import joypy
import matplotlib.pyplot as plt
import seaborn as sns

dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = os.environ['LAV_DIR']

import lernia.train_reshape as t_r
import lernia.train_modelList as t_l
import lernia.train_model as tlib
import lernia.train_viz as t_v
import deep_lernia.train_keras as t_k
import deep_lernia.train_longShort as tls
import albio.series_stat as s_s
import importlib
importlib.reload(s_s)

print('-----------------------load-data------------------------------')
fL = os.listdir(baseDir + "/rem/raw/network")
fL1 = os.listdir(baseDir + "/rem/raw/telemetry/august/")

featL = []
f = fL[4]
f1 = re.sub("network_","telemetry_",f)
#for f in fL:
net = pd.read_csv(baseDir + "/rem/raw/network/" + f)
featL.append(net)
featL = pd.concat(featL).reset_index()
featL.replace(0,float('nan'),inplace=True)
featL = featL[featL['timestamp_ms'] == featL['timestamp_ms']]
featL.loc[:,"second"] = featL['timestamp_ms'].apply(lambda x: x.split(".")[0])
featL.loc[:,'ts'] = [datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S.%f") for x in featL['timestamp_ms']]
featL.loc[:,"ms"] = [float(x[20:23]) for x in featL['timestamp_ms']]
featL = featL[featL['name'] == featL['name']]
tL = [{'packets_since_start':'packets_start'},{'bytes_since_start':'bytes_start'},{'min_inter_packet_arrival_time':'min_arrival'},{'max_inter_packet_arrival_time':'max_arrival'},{'avg_inter_packet_arrival_time':'avg_arrival'},{'median_inter_packet_arrival_time':'median_arrival'}]
for t in tL:
    featL.rename(columns=t,inplace=True)

tL = ['packets_start','bytes_start', 'packets', 'bytes', 'min_ttl', 'max_ttl', 'min_arrival','max_arrival', 'avg_arrival', 'median_arrival', 'interval_duration']
dt = len(np.unique(featL['second']))

if False:
    print('------------------------join-telemetry------------------------')
    fL = os.listdir(baseDir + "/rem/raw/network")
    netL = []
    for f in fL:
        print(f)
        net = pd.read_csv(baseDir + "/rem/raw/network/" + f)
        f1 = re.sub("network_","telemetry_",f)
        try:
            tel = pd.read_csv(baseDir + "/rem/raw/telemetry/august/" + f1)
        except:
            continue
        net.loc[:,"ts1"] = net['timestamp_ms'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S.%f"))
        net.loc[:,"timebucket"] = net['ts1'].apply(lambda x: int(x.timestamp()*10.)/10)
        net.drop(columns={'min_inter_packet_arrival_time','packets_since_start','bytes_since_start','max_inter_packet_arrival_time','avg_inter_packet_arrival_time','meta','year','month','day','hour','dt','ts','resampled','interval_duration','ts1'},inplace=True)
        tel.drop(columns={'object_distance','brake_pressure','force_lat','force_lon','yaw_rate','steering_wheel','steering_angle','wheel_speed',"session_id"},inplace=True)
        netl = tel.merge(net,on=["timebucket","vehicle_id"],how="left")
        f2 = re.sub("network_","telnet_",f)
        #netl.to_csv(baseDir + '/rem/raw/telnet/' + f2,index=False)
        netL.append(netl)
    netL = pd.concat(netL)
    netL.to_csv(baseDir + "/rem/raw/telnet.csv.gz",index=False)

    
if False:
    print('-----------------------telnet-stats---------------------------')
    featL = pd.read_csv(baseDir + "/rem/raw/telnet.csv.gz")
    xL = ['vehicle_ping','rtp_lost','rtp_late','room_ram','room_cpu','vehicle_ram','vehicle_cpu','packets','bytes','min_ttl','max_ttl','median_inter_packet_arrival_time']
    yL = ['camera_latency','joystick_latency','e2e_latency']
    tL = xL + yL

    t_v.plotCorr(netL[tL].dropna(),labV=tL)
    plt.yticks(rotation=15)
    plt.xticks(rotation=35)
    plt.show()

    
    importlib.reload(s_s)
    cML = {}
    for i,g in featL.groupby('name'):
        X = s_s.interpMissingMatrix(g[tL])
        cM = s_s.delayM(X)
        cM = cM/len(y1)*2
        cML[i] = cM

    t_v.plotTimeSeries(featL[tL])
    plt.show()

    
    t_v.plotHeatmap(cM,tL,vmin=-0.5,vmax=0.5)
    plt.title("delay between features")
    plt.yticks(rotation=15)
    plt.xticks(rotation=35)
    plt.show()
    

    
if False:
    print('------------------------print-time-series-------------------------')
    t_v.plotTimeSeries(featL.loc[:2500,tL])
    plt.show()

if False:
    print('--------------------histograms-and-outliers------------------')
    feat = featL.copy()
    for t in tL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    #sns.factorplot(data=feat.loc[:500],x='packets',y='interval_duration',hue='name',col='interval_duration',kind='box',legend=True)
    plt.clf()
    fig, ax = plt.subplots(2,3)
    ax = ax.flatten()
    j = 0
    for i,g in feat.groupby('name'):
        ax[j].set_title(i)
        bx = g[tL].boxplot(ax=ax[j],color=t_v.colorL[j],return_type="dict")
        [[item.set_color(t_v.colorL[j]) for item in bx[key]] for key in bx.keys()]
        j = j + 1
    for a in ax:
        for tick in a.get_xticklabels():
            tick.set_rotation(15)
    plt.show()
    
    sL = ['room_cpu','room_ram','vehicle_ram','vehicle_cpu']
    sL = ['wheel_speed','brake_pressure','object_distance','modem_rtt','modem_tx','modem_rx','rtp_lost','rtp_late']
    t_v.plotCorr(featL[tL].dropna(),labV=tL)
    plt.yticks(rotation=15)
    plt.xticks(rotation=35)
    plt.show()

if False:
    print('------------------------------produce-joyplots---------------------------')
    feat = featL.copy()
    for t in tL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    fig, axes = joypy.joyplot(feat,column=tL,xlim='own',ylim='own',figsize=(12,6),alpha=.5)#,colormap=plt.cm.Blues)
    plt.show()



