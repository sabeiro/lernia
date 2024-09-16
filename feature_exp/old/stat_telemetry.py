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

import importlib
import lernia.train_viz as t_v
import lernia.train_reshape as t_r
import albio.series_stat as s_s

print('-----------------------load-data------------------------------')
folder = "august"
folder = "modem"
featL = pd.read_csv(baseDir + "rem/raw/telemetry_"+folder+".csv.gz",compression="gzip")
print(featL.shape)
#featL = pd.read_csv(baseDir + "rem/raw/spike_list.csv.gz",compression="gzip")

xL = [x for x in featL.columns if re.search('modem',x)]
yL = ['camera_latency','joystick_latency']
tL = yL + xL

if False:
    print('------------------------print-time-series-------------------------')
    #t_v.plotTimeSeries(featL[:100000][tL])
    t_v.plotTimeSeries(featL[tL])
    plt.show()

    t_v.plotTimeSeries(feat[tL])
    plt.show()

    
if False:
    print('--------------------histograms-and-outliers------------------')
    feat = featL.copy()
    for t in tL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    feat[tL].boxplot()
    plt.xticks(rotation=35)
    plt.show()
    
    sL = ['object_distance','brake_pressure','force_lat','force_lon','yaw_rate','steering_wheel','steering_angle','wheel_speed']
    sL = ['vehicle_ping','rtp_lost','rtp_late','modem_rtt','modem_rx','modem_tx']
    sL = ['camera_jitter','room_ram','room_cpu','vehicle_ram','vehicle_cpu']
    sL = ['modem0_rssi', 'modem0_sinr','modem0_rsrp','modem0_rsrq','modem1_rssi', 'modem1_sinr','modem1_rsrp','modem1_rsrq','modem2_rssi', 'modem2_sinr','modem2_rsrp','modem2_rsrq','modem3_rssi', 'modem3_sinr','modem3_rsrp','modem3_rsrq']

    t_v.plotCorr(featL.loc[:,sL],labV=sL)
    plt.yticks(rotation=15)
    plt.xticks(rotation=15)
    plt.show()

    fig, ax = plt.subplots(2,2)
    ax = ax.flatten()
    for i in range(4):
        sL = [x for x in tL if re.search("modem"+str(i),x)]
        t_v.plotCorr(featL[sL],labV=sL,ax=ax[i])
        plt.yticks(rotation=15)
        plt.xticks(rotation=35)
    plt.show()

    
if False:
    print('------------------------------produce-joyplots---------------------------')
    xL = ['object_distance','force_lat','force_lon','steering_wheel','wheel_speed','vehicle_ping','rtp_lost','modem_rtt','modem_tx','camera_jitter','vehicle_ram','vehicle_cpu']
    feat = featL.copy()
    tL = xL + yL
    for t in tL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    fig, axes = joypy.joyplot(feat,column=tL,xlim='own',ylim='own',figsize=(12,6),alpha=.5)#,colormap=plt.cm.Blues)
    plt.show()

if False:
    print('--------------------------------pair-plots------------------------------')
    sL = ['object_distance','brake_pressure','force_lat','force_lon','yaw_rate','steering_wheel','steering_angle','wheel_speed']
    sL = ['camera_jitter','room_ram','room_cpu','vehicle_ram','vehicle_cpu']
    sL = ['rtp_lost','rtp_late']#,'modem_rx','modem_tx','modem_rtt']
    sL = ['vehicle_ping','rtp_lost','modem_tx','modem_rtt']
    featL.loc[:,'b_latency'], _ = t_r.binOutlier(featL['camera_latency'],nBin=3)
    feat = featL.copy()
    feat = featL[:10000]
    for t in tL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])

    i = 3
    sL = [x for x in tL if re.search("modem"+str(i),x)]
    sns.pairplot(feat[sL+['b_latency']],hue='b_latency',kind="reg",diag_kind="kde",markers="+"
                 #,plot_kws={"s":50,"edgecolor":"b","linewidth":1}
                  ,plot_kws={'scatter_kws':{'alpha':0.1}}
                 ,diag_kws={"shade":True})
    plt.show()

    g = sns.PairGrid(featL[sL+['b_latency']],hue="b_latency")
    g = g.map_diag(sns.kdeplot, lw=3)
    g = g.map_offdiag(sns.kdeplot, lw=1)
    plt.show()
    
    plt.plot(feat['camera_latency'],feat['e2e_latency'],'o')
    plt.show()



if False:
    print('---------------------------------phase-shift-----------------------------')
    y1 = s_s.interpMissing(featL['force_lat'])
    y2 = s_s.interpMissing(featL['room_cpu'])
    importlib.reload(s_s)
    # s_s.maxLag(y1,y2,isPlot=True)
    print(s_s.timeLag(y1,y2,isPlot=False))
    X1 = s_s.interpMissingMatrix(featL[:10000][tL])
    lagM = s_s.lagMatrix(X1)
    importlib.reload(t_v)
    lagM[np.abs(lagM) < 0.1] = 0
    t_v.plotHeatmap(lagM,X1.columns,vmin=-np.pi,vmax=np.pi)
    plt.title("phase lag between features")
    plt.yticks(rotation=15)
    plt.xticks(rotation=35)
    plt.show()

if False:
    print('------------------------------camera-index--------------------------------')
    xL = ['wheel_speed','vehicle_ping','rtp_lost','rtp_late','modem0_rtt','modem1_rtt','modem2_rtt','modem3_rtt','modem0_rx','modem1_rx','modem2_rx','modem3_rx','modem0_tx','modem1_tx','modem2_tx','modem3_tx','camera_jitter','room_ram','room_cpu','vehicle_ram','vehicle_cpu']
    tL = yL + xL
    feat = featL.copy()
    for t in tL:
        feat.loc[:,t] = s_s.interpMissing(feat[t])
        feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[1,99])
        feat.loc[:,t] = s_s.interpMissing(feat[t])
        
    fig, ax = plt.subplots(3,2)
    ax = ax.flatten()
    for i,g in feat.groupby('camera_index'):
        g[tL].boxplot(ax=ax[int(i)])
    plt.show()
