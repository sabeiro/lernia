import os, sys, gzip, random, json, datetime, re, io
import pandas as pd
import numpy as np
from scipy import stats as st
import joypy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = os.environ['LAV_DIR']

import importlib
import lernia.train_viz as t_v
import lernia.train_reshape as t_r
import albio.series_stat as s_s
from sawmill import aws_utils as a_u

print('-----------------------load-data------------------------------')
fL = os.listdir(baseDir + "/rem/raw/telemetry/august/")
featL = []
f = fL[4]
for f in fL:
    feat = pd.read_csv(baseDir + "/rem/raw/telemetry/august/" + f)
    feat.loc[:,"joystick_latency"] = feat["joystick_latency"]/1000000.
    featL.append(feat)
featL = pd.concat(featL)
featL = featL.sort_values('timebucket')
featL.loc[:,'ts'] = [datetime.datetime.fromtimestamp(x) for x in featL['timebucket']]
featL.loc[:,"second"] = featL['timebucket'].apply(lambda x: int(x*10)/10.)
sL = [x for x in featL.columns if bool(re.search("modem",x))]
featL.drop(columns=sL,inplace=True)

if True:
    print('----------------------------upsample-to-a-deci-second-----------------------------')
    featL = featL.groupby(['second','session_id','vehicle_id']).agg(np.nanmean).reset_index()
    featL.loc[:,'ts'] = [datetime.datetime.fromtimestamp(x) for x in featL['second']]
    featL.loc[:,'b_latency'], _ = t_r.binOutlier(featL['camera_latency'],nBin=4)
    for i,g in featL.groupby('session_id'):
        featL.loc[g.index,"session_time"] = g['second'] - g.iloc[0]['second']

if False:
    print('-----------------------------missing----------------------------')
    missing = featL[tL].isna().sum(axis=0)/len(featL)*100.
    missing = missing.sort_values(ascending=False)
    plt.bar(missing.index,missing)
    plt.title("missing values")
    plt.ylabel("% missing")
    plt.xticks(rotation=15)
    plt.show()

if True:
    print('-------------------------------input-transformation---------------------------------')
    sL = ['force_lat','force_lon','yaw_rate','steering_wheel','steering_angle','vehicle_ping','brake_pressure','object_distance','rtp_lost','rtp_late']
    feat = featL.copy()
    for t in sL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    # fig, ax = plt.subplots(1,2)
    # feat[sL].boxplot(ax=ax[0])
    for i in sL:
        threshold = abs(np.nanmean(featL[i])*.05)
        featL.loc[:,i] = np.log(threshold + np.abs(featL[i]))
        # featL.loc[:,i] = t_r.normPercentile(featL[i],perc=[5,95])

    feat = featL.copy()
    for t in sL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    # feat[sL].boxplot(ax=ax[1])
    # for a in ax.flatten():
    #     for tick in a.get_xticklabels():
    #         tick.set_rotation(15)
    # plt.show()

xL = ['object_distance','brake_pressure','force_lat','force_lon','yaw_rate','steering_wheel','steering_angle','wheel_speed','vehicle_ping','rtp_lost','rtp_late','camera_jitter','room_ram','room_cpu','vehicle_ram','vehicle_cpu']
yL = ['camera_latency','joystick_latency','e2e_latency']
tL = xL + yL
sL = ['second','session_id','session_time','longitude','latitude']

threshold = 300
sec_post = 15
featL = featL[featL['second'] == featL['second']]
featL = featL.sort_values(["session_id","second"]).reset_index()
idxL = featL[featL['camera_latency'] >  threshold].index
secL = featL.loc[idxL,'second'].values
keep = [True] + list(secL[1:] - secL[:-1] > sec_post)
idxL = idxL[keep]
print("number of peaks",len(idxL))

featD = []
for i,j  in enumerate(idxL):
    peakT = featL.loc[j]['second']
    setL = (featL['second'] - peakT).abs() < sec_post
    g = featL[setL]
    g = g[g['session_id'] == featL.loc[j]['session_id']]
    # g = g[g['vehicle_id'] == featL.loc[j]['vehicle_id']]
    g.loc[:,"from_peak"] = g['second'].apply(lambda x:  int( (x-peakT)*10 )/10. )
    g.index = [datetime.datetime.fromtimestamp(x) for x in g['second']]
    day = str(g.index.month[0]) + "_" + str(g.index.day[0])
    dt = g.index[-1] - g.index[0]
    if dt.seconds < sec_post*1.1: continue
    plt.plot(g['second'],g['camera_latency'],label=i)
    g.loc[:,"series"] = day+ "_s_"+ str(i)
    featD.append(g)
    
featD = pd.concat(featD)
print(len(set(featD['series'])))
for i in yL:
    featD.loc[featD[i] > threshold,i] = threshold

featD.drop(columns={"index"},inplace=True)
featD.to_csv(baseDir + "/rem/raw/spike_list_deci.csv.gz",compression="gzip",index=False)
plt.show()

if False:
    print('-------------------------------------aggregate-files--------------------------------')
    featL = []
    fL = os.listdir(baseDir + "/rem/raw/incident/incident_deci/")
    for f in fL:
        feat = pd.read_csv(baseDir + "/rem/raw/incident/incident_deci/" + f)
        
        featL.append(feat)
    featL = pd.concat(featL)
    if True:
        print('-------------------------------input-transformation---------------------------------')
        sL = ['arrival_time-Cellular 1','arrival_time-Cellular 2','arrival_time-Cellular 3','arrival_time-Cellular 4']
        feat = featL.copy()
        for t in sL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
        for i in sL:
            threshold = abs(np.nanmean(featL[i])*.05)
            featL.loc[:,i] = np.log(threshold + np.abs(featL[i]))

    xL = ['force_lat','force_lon','wheel_speed','camera_jitter','arrival_time-Cellular 1','arrival_time-Cellular 2','arrival_time-Cellular 3','arrival_time-Cellular 4','bytes-Cellular 1','bytes-Cellular 2','bytes-Cellular 3','bytes-Cellular 4']
    yL = ['camera_latency','joystick_latency','e2e_latency']
    sL = ['second','series','longitude','latitude','from_peak']
    tL = sL + xL + yL
    featL[tL].to_csv(baseDir + "/rem/raw/spike_deci.csv.gz",compression="gzip",index=False)
