import os, sys, gzip, random, json, datetime, re, io
import pandas as pd
import numpy as np
from scipy import stats as st
import scipy as sp
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

featL = pd.read_csv(baseDir + "rem/raw/spike_list.csv.gz",compression="gzip")
incL = pd.read_csv(baseDir + "/rem/raw/incident_list.csv")
projDir = baseDir + "/rem/raw/incident/incident_sec/"
fL = os.listdir(projDir)
f = fL[-1]
featD = []
for f in fL:
    feat = pd.read_csv(projDir + f)
    if len(feat) == 0: continue
    feat = feat.iloc[1:]
    feat.replace(float('inf'),float('nan'),inplace=True)
    feat.replace('NaN',float('nan'),inplace=True)
    feat['joystick_latency'].replace(0,float('nan'),inplace=True)
    # feat = feat.ffill()
    # feat = feat.bfill()
    feat.index = feat['timebucket'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)))
    mL = [x for x in feat.columns if bool(re.search("modem",x))]
    feat.loc[:,"modem_rtt"] = feat.apply(lambda x: np.nanmean(x[['modem0_rtt','modem1_rtt','modem2_rtt','modem3_rtt']]),axis=1)
    feat.loc[:,"modem_tx"] = feat.apply(lambda x: np.nanmean(x[['modem0_tx','modem1_tx','modem2_tx','modem3_tx']]),axis=1)
    feat.loc[:,"modem_rx"] = feat.apply(lambda x: np.nanmean(x[['modem0_rx','modem1_rx','modem2_rx','modem3_rx']]),axis=1)
    if feat['modem_tx'].sum() <= 0.: continue
    # feat.drop(columns=mL,inplace=True)
    feat.loc[:,"series"] = f.split("_")[1].split(".")[0]
    featD.append(feat)
featL = pd.concat(featD)

xL = ['object_distance','brake_pressure','force_lat','force_lon','yaw_rate','steering_wheel','steering_angle','wheel_speed','vehicle_ping','rtp_lost','rtp_late','modem_rtt', 'modem_rx','modem_tx','camera_jitter','room_ram','room_cpu','vehicle_ram','vehicle_cpu']
yL = ['camera_latency','joystick_latency', 'e2e_latency']
sL = ['timebucket','series']
mL = [x for x in xL if bool(re.search("modem",x))]
tL = xL + yL

if True:
    for i in tL:
        threshold = np.nanmean(featL[i])*7.
        featL.loc[featL[i] > threshold,i] = threshold
    featL.loc[:,'b_latency'], _ = t_r.binOutlier(featL['camera_latency'],nBin=4)

if True:
    for i in ['force_lat','force_lon','yaw_rate','steering_wheel','steering_angle','vehicle_ping','rtp_lost','rtp_late','brake_pressure','object_distance'] + yL:
        threshold = abs(np.nanmean(featL[i])*.05)
        featL.loc[:,i] = np.log(threshold + np.abs(featL[i]))
        # featL.loc[:,i] = t_r.normPercentile(featL[i],perc=[5,95])

if False:
    for t in tL: featL.loc[:,t] = t_r.normPercentile(featL[t],perc=[5,95])

y = s_s.interpMissing(featL['camera_latency'])
X = s_s.interpMissingMatrix(featL[xL])
t = featL.index
l = list(range(len(y)))

sL = ['timebucket','series']
featL[sL + xL + yL].to_csv(baseDir + "rem/raw/incident_series.csv.gz",index=False,compression="gzip")
