import os, sys, gzip, random, json, datetime, re, io
import pandas as pd
from scipy import stats as st
import joypy

import lernia.train_reshape as t_r
import lernia.train_viz as t_v
import lernia.train_shape as t_s
import lernia.train_feature as t_f
import lernia.train_model as tlib
import lernia.train_modelList as t_l

dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = os.environ['LAV_DIR']
from sawmill import aws_utils as a_u
import importlib

feat = pd.read_csv(baseDir + "rem/raw/telemetry_1sec.csv.gz")
feat = feat[feat['camera_latency'] < 1000.]
print(len(feat[feat['camera_latency'] >  400.]))
idxL = feat[feat['camera_latency'] >  400.].index
n_peak = idxL[1] + 12
n1, n2 = max(0,n_peak - 500), n_peak
feat = feat[n1:n2]
feat.index = [datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S.%f") for x in feat['resampled_dt']]
y = s_s.interpMissing(feat['camera_latency'])
y = s_s.interpMissing(feat['camera_latency'])
tL = ['mean_km_per_hour', 'lateral_force_m_per_sec_squared','longitudinal_force_m_per_sec_squared', 'v_cpu_usage_percent', 'v_ram_usage_percent']
yL = ['e2e_latency', 'camera_latency', 'joystick_latency']
t = feat.index
l = list(range(len(y)))
X = s_s.interpMissingMatrix(feat[tL])
X = pd.DataFrame(X,columns=tL)
if False:
    t_v.plotTimeSeries(feat[tL+yL])
    plt.show()

if False:
    print('--------------------cross-correlation------------------')
    plt.plot(feat['camera_latency'],feat['e2e_latency'])
    plt.plot(feat['camera_latency'],feat['e2e_latency'])
    plt.show()
    

if False:
    print('--------------------histograms-and-outliers------------------')
    featL = feat[tL + yL]
    for t in featL.columns: featL.loc[:,t] = t_r.normPercentile(featL[t],perc=[1,99])
    
    fig, ax = plt.subplots(1,2)
    featL.boxplot(ax=ax[0])    
    featL = feat[tL + yL]
    for t in ['lateral_force_m_per_sec_squared', 'longitudinal_force_m_per_sec_squared', 'e2e_latency','camera_latency','joystick_latency']:
        print(t)
        featL.loc[:,t] = np.log(np.abs(featL.loc[:,t]))
    for t in featL.columns: featL.loc[:,t] = t_r.normPercentile(featL[t],perc=[1,99])
    featL.boxplot(ax=ax[1])
    for a in ax.flatten():
        for tick in a.get_xticklabels():
            tick.set_rotation(15)
    plt.show()

    t_v.plotCorr(featL[tL + yL].dropna(),labV=tL+yL)
    plt.yticks(rotation=15)
    plt.xticks(rotation=35)
    plt.show()

    fig, axes = joypy.joyplot(featL,column=tL+yL,xlim='own',ylim='own',figsize=(12,6),alpha=.5)#,colormap=plt.cm.Blues)
    plt.show()
    

if False:
    print('------------------pca-reduction-----------------------------')
    X = featL[tL].dropna()
    x, xv = t_s.calcPCA(X)
    y = pd.DataFrame(featL.loc[X.index,'camera_latency'].values,columns=["camera_latency"])
    pcaL = pd.DataFrame(x,columns=range(len(tL)))
    pcaL = pd.concat([pcaL,y],axis=1)
    t_v.plotCorr(pcaL,labV=list(pcaL.columns))
    plt.yticks(rotation=15)
    plt.xticks(rotation=15)
    plt.show()

if False:
    print('----------------------predictability------------------------')
    m = yL[1]
    X = t_s.interpMissing(featL[tL])
    y = s_s.interpMissing(featL[m])
    #y, _ = t_r.binVector(y,nBin=7,threshold=0.5)
    mod = t_l.modelList(paramF=baseDir+"rem/train/weath_"+m+".json")
    mod.get_params()
    importlib.reload(t_l)
    importlib.reload(tlib)
    tMod = tlib.trainMod(X,y)
    mod, trainR = tMod.loopMod(paramF=baseDir+"rem/train/weath_"+m+".json",test_size=.4)
    tMod.plotRoc()

    
if False:
    print('----------------------feature-knock-out---------------------------')
    m = yL[1]
    X = t_s.interpMissing(featL[tL])
    y = s_s.interpMissing(featL[m])
    y = t_r.normPercentile(y,perc=[1,99])
    y = s_s.interpMissing(y)
    from lernia import train_longShort as tls
    importlib.reload(lernia)
    importlib.reload(t_k)
    importlib.reload(tls)
    tK = lernia.train_longShort.timeSeries(X.values)
    perfL = tK.featKnockOut(X,y)
    #model, kpi = tK.train(y,epochs=20)
    tK.plotFeatImportance(perfL)
    plt.show()




    




net = pd.read_csv(baseDir + "rem/raw/network_small.csv")
inc = pd.read_csv(baseDir + "rem/raw/incident_small.csv")
tel = pd.read_csv(baseDir + "rem/raw/telemetry_small.csv")
ses = pd.read_csv(baseDir + "rem/raw/session_small.csv")
