import os, sys, gzip, random, json, datetime, re, io
import pandas as pd
from scipy import stats as st
import matplotlib.pyplot as plt
import joypy

dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = os.environ['LAV_DIR']
from sawmill import aws_utils as a_u
import importlib

threshold = 300
sec_tot = 2*60
sec_post = 60
folder = 'august'
folder = 'modem'
featL = pd.read_csv(baseDir + "rem/raw/telemetry_"+folder+".csv.gz")
featL = featL[featL['second'] == featL['second']]
featL = featL.sort_values(["session_id","second"]).reset_index()
idxL = featL[featL['camera_latency'] >  threshold].index
secL = featL.loc[idxL,'second'].values
keep = [True] + list(secL[1:] - secL[:-1] > sec_post)
idxL = idxL[keep]
print("number of peaks",len(idxL))

yL = ['e2e_latency', 'camera_latency', 'joystick_latency']
featD = []
for i,j  in enumerate(idxL):
    peakT = featL.loc[j]['second']
    setL = (featL['second'] - peakT).abs() < sec_post
    if sum(setL) < int(1.5*sec_post): continue
    g = featL[setL]
    g = g[g['session_id'] == featL.loc[j]['session_id']]
    # g = g[g['vehicle_id'] == featL.loc[j]['vehicle_id']]
    g.index = [datetime.datetime.fromtimestamp(x) for x in g['second']]
    sec = [(x-g.index[0]).seconds for x in g.index]
    g.loc[:,"from_peak"] = g['second'] - peakT
    day = "%02d" % g.index.day[0]
    hour = "%02d" % g.index.hour[0]
    dt = g.index[-1] - g.index[0]
    if dt.seconds < sec_post*1.1: continue
    plt.plot(g['camera_latency'],label=i)
    g.loc[:,"series"] = day+"_"+hour+"_s_"+ str(i)
    featD.append(g)
featL = pd.concat(featD)
for i in yL:
    featL.loc[featL[i] > threshold,i] = threshold

featL.to_csv(baseDir + "rem/raw/spike_"+folder+".csv.gz",compression="gzip",index=False)
print(len(set(featL['series'])))
plt.show()
    
if False:
    print('--------------------analyze-cases---------------------------')
    serL = []
    for i,g in featL.groupby('series'):
        if not (re.search("17_11",i) ):# or re.search("24_18",i) ):
            continue
        serL.append(g)
        #plt.plot(g.index,g['camera_latency'],label=g['session_id'][0])
    #plt.legend()
    #plt.show()
    serL = pd.concat(serL)
    serL = serL.groupby('from_peak').agg(np.nanmean).reset_index()
    X = serL[tL]
    for t in tL:
        X.loc[:,t] = X[t].shift(1) - X[t]
    X = X.iloc[1:,]
    t_v.plotTimeSeries(serL[tL])
    plt.xlabel("seconds")
    plt.show()




#{name=Cellular 1, message=Connected to BleibGesund, ip=100.89.14.68, status_led=green, uptime=107, carrier_name=BleibGesund, carrier_country=Germany, band=LTE Band 7 (2600 MHz), rssi=-75, sinr=13.2, rsrp=-103, rsrq=-9, level=0, cell_id=55559, tunnel_state=ACTIVE, tunnel_active=true, tunnel_uptime_sec=224, tunnel_uptime_ns=463051382, tunnel_rtt=58, tunnel_rx=785989, tunnel_tx=823246, tunnel_loss=0, tunnel_delta_valid=true, tunnel_delta_interval_sec=1, tunnel_delta_interval_ns=31814877, tunnel_delta_loss=0, tunnel_delta_tx=97089, tunnel_delta_rx=93134}
