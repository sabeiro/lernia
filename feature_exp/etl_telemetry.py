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

folder = 'august'
folder = 'modem'
print('-----------------------load-data-'+folder+'-----------------------------')
fL = os.listdir(baseDir + "/rem/raw/telemetry/"+folder+"/")
featL = []
f = fL[4]
for f in fL:
    feat = pd.read_csv(baseDir + "/rem/raw/telemetry/"+folder+"/" + f)
    #feat.loc[:,"second"] = [int(x*10.)/10 for x in feat['timebucket']]
    feat.loc[:,"second"] = [int(x) for x in feat['timebucket']]
    feat.loc[:,"joystick_latency"] = feat["joystick_latency"]/1000000.
    for i in [x for x in feat.columns if re.search("_cell",x)]: # handover
        y = feat[i].bfill().ffill()
        y = y - y.shift(1)
        y[0] = 0
        y = 1*(y != 0)
        feat.loc[:,i] = y
    feat.drop(columns={"timebucket","vehicle_id"},inplace=True)
    feat = feat.groupby(['second','session_id']).agg(np.nanmean).reset_index()
    featL.append(feat)
featL = pd.concat(featL)
featL = featL.sort_values('second')
print(featL.shape)
#featL.loc[:,'ts'] = [datetime.datetime.fromtimestamp(x) for x in featL['second']]

if False:
    print('-----------------------------interpolate-missing------------------------------')
    tL = featL.columns[featL.dtypes == float]
    for t in tL:
        featL.loc[:,t] = s_s.interpMissing(featL[t])
        
if False:
    print('-----------------------------missing----------------------------')
    missing = featL[tL].isna().sum(axis=0)/len(featL)*100.
    missing = missing.sort_values(ascending=False)
    plt.bar(missing.index,missing)
    plt.title("missing values")
    plt.ylabel("% missing")
    plt.xticks(rotation=15)
    plt.show()
    
if False:
    print('-------------------------------mean-modem---------------------------------')
    featL.replace(float('inf'),float('nan'),inplace=True)
    featL.loc[:,'modem_tx'] = featL.apply(lambda x: np.nanmean([x['modem0_tx'],x['modem1_tx'],x['modem2_tx'],x['modem3_tx']]),axis=1)
    featL.loc[:,'modem_rx'] = featL.apply(lambda x: np.nanmean([x['modem0_rx'],x['modem1_rx'],x['modem2_rx'],x['modem3_rx']]),axis=1)
    featL.loc[:,'modem_rtt'] = featL.apply(lambda x: np.nanmean([x['modem0_rtt'],x['modem1_rtt'],x['modem2_rtt'],x['modem3_rtt']]),axis=1)
if False:
    print('-------------------------------log-transform---------------------------------')
    sL = ['force_lat','force_lon','yaw_rate','steering_wheel','steering_angle','vehicle_ping','brake_pressure','object_distance','modem_tx','modem_rx','modem_rtt','rtp_lost','rtp_late']
    feat = featL.copy()
    for t in sL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    fig, ax = plt.subplots(1,2)
    feat[sL].boxplot(ax=ax[0])
    for i in sL:
        threshold = abs(np.nanmean(featL[i])*.05)
        featL.loc[:,i] = np.log(threshold + np.abs(featL[i]))
        # featL.loc[:,i] = t_r.normPercentile(featL[i],perc=[5,95])

    feat = featL.copy()
    for t in sL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    feat[sL].boxplot(ax=ax[1])
    for a in ax.flatten():
        for tick in a.get_xticklabels():
            tick.set_rotation(15)
    plt.show()

xL = ['object_distance','brake_pressure','force_lat','force_lon','yaw_rate','steering_wheel','steering_angle','wheel_speed','vehicle_ping','rtp_lost','rtp_late','modem_rtt','modem_rx','modem_tx','camera_jitter','room_ram','room_cpu','vehicle_ram','vehicle_cpu']
yL = ['camera_latency','joystick_latency','e2e_latency']
tL = xL + yL
sL = ['second','session_id','session_time','longitude','latitude'] + tL
sL = featL.columns
featL[sL].to_csv(baseDir + "/rem/raw/telemetry_"+folder+".csv.gz",compression="gzip",index=False)
