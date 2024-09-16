import os, sys, gzip, random, json, datetime, re, io
import pandas as pd
import numpy as np
from scipy import stats as st
import scipy as sp
import joypy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.collections as mcoll
import matplotlib.path as mpath

dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = os.environ['LAV_DIR']

import lernia.train_reshape as t_r
import lernia.train_modelList as t_l
import lernia.train_model as tlib
import lernia.train_viz as t_v
import albio.series_stat as s_s
import importlib
import lernia.train_feature as t_f
importlib.reload(s_s)

#fName = baseDir + "rem/raw/spike_deci.csv.gz"
folder = "august"
folder = "modem"
featL = pd.read_csv(baseDir + "rem/raw/spike_"+folder+".csv.gz",compression="gzip")
featL = pd.read_csv(baseDir + "rem/raw/telemetry_"+folder+".csv.gz",compression="gzip")
print(featL.shape)

xL = ['object_distance','brake_pressure','force_lon','wheel_speed','vehicle_ping','rtp_lost','rtp_late','modem_rtt','modem_tx','camera_jitter','room_ram','room_cpu','vehicle_ram','vehicle_cpu']
xL = ['object_distance','force_lat','steering_wheel','wheel_speed','vehicle_ping','rtp_lost','rtp_late','modem_tx','vehicle_ram','vehicle_cpu']
xL = [x for x in featL.columns if re.search("modem",x)]
#xL = ['force_lon','wheel_speed','camera_jitter','arrival_time-Cellular 1','arrival_time-Cellular 2','arrival_time-Cellular 3','arrival_time-Cellular 4','bytes-Cellular 1','bytes-Cellular 2','bytes-Cellular 3','bytes-Cellular 4']
yL = ['camera_latency','joystick_latency','e2e_latency']
tL = xL + yL
featL.loc[:,"from_peak"] = featL['from_peak'].apply(lambda x:  int( x*10 )/10. )
serAv = featL.groupby('from_peak').agg(np.nanmean).reset_index()
serVar = featL.groupby('from_peak').agg(np.std).reset_index()

if False:
    print('--------------------histograms-and-outliers------------------')
    feat = featL.copy()
    for t in tL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    feat[tL].boxplot()
    plt.xticks(rotation=35)
    plt.show()
    
    sL = [x for x in tL if re.search("_s",x)] + [x for x in tL if re.search("_rs",x)]
    sL = [x for x in featL.columns if re.search("_cell",x)]
    t_v.plotCorr(featL[sL],labV=sL)
    plt.yticks(rotation=15)
    plt.xticks(rotation=35)
    plt.show()

    sL = [x for x in featL.columns if re.search("_cell",x)]
    t_v.plotTimeSeries(featL[sL])
    plt.show()

    fig, ax = plt.subplots(2,2)
    ax = ax.flatten()
    for i in sL:
        h, t = np.unique(featL[i],return_counts=True)
        t = np.array(t)/sum(t)
        ax[i].bar(h,t,label=i,alpha=0.3)
    plt.legend()
    plt.show()
        
    
    
if False:
    print('------------------------plot-time-series-------------------------')
    xL = [x for x in featL.columns if re.search("_sinr",x)] + [x for x in featL.columns if re.search("_rs",x)]
    tL = xL + yL
    importlib.reload(t_v)
    t_v.plotTimeSeries(featL[tL],t=featL['from_peak'],mode="binned")
    plt.show()

    setL = featL['from_peak'].abs() < 4
    t_v.plotTimeSeries(featL.loc[setL,tL],t=featL.loc[setL,'from_peak'],mode="binned")
    plt.show()
   
    t_v.plotBinned(featL['from_peak'],featL['camera_latency'],isScatter=True)
    plt.show()
    
    t_v.plotBinned(featL['from_peak'],featL['wheel_speed'],isScatter=True)
    plt.show()
    
    serAv = featL.groupby('from_peak').agg(np.nanmean).reset_index()
    t_v.plotTimeSeries(serAv[tL],t=serAv['from_peak'])
    plt.xlabel("seconds")
    plt.show()

    t_v.plotTimeSeries(serVar[tL],t=serVar['from_peak'])
    plt.xlabel("seconds")
    plt.show()

    setL = serAv['from_peak'].abs() < 8
    t_v.plotTimeSeries(serAv.loc[setL,tL],t=serAv['from_peak'][setL])
    plt.xlabel("seconds")
    plt.show()
    
    importlib.reload(s_s)
    setL = featL['from_peak'].abs() < 12
    X = featL.copy()
    for t in tL:
        X.loc[:,t] = s_s.interpMissing(X[t])
        X.loc[:,t] = s_s.serSmooth(X[t],width=5,steps=12)
        #X.loc[:,t] = X[t].shift(1) - X[t]
    X = X.iloc[1:,]
    t_v.plotTimeSeries(X.loc[setL,tL],t=X.loc[setL,'from_peak'],mode="binned")
    plt.xlabel("seconds")
    plt.show()

    
    t_v.plotTimeSeries(X.loc[setL,tL],t=X.loc[setL,'from_peak'],mode="binned")
    plt.xlabel("seconds")
    plt.show()

    t_v.plotTimeSeries(featL[tL],t=featL['from_peak'],mode="scatter")
    plt.xlabel("seconds")
    plt.show()

    
if False:
    print('--------------------------------pair-plots------------------------------')
    sL = ['bytes-Cellular 1', 'bytes-Cellular 2', 'bytes-Cellular 3', 'bytes-Cellular 4']
    sL = [x for x in featL.columns if re.search("_rsrp",x)]
    featL.loc[:,'b_latency'], _ = t_r.binOutlier(featL['camera_latency'],nBin=3)

    feat = featL[:10000]
    for t in sL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])

    sns.pairplot(feat[sL+['b_latency']],hue='b_latency',kind="reg",diag_kind="kde",markers="+"
                 #,plot_kws={"s":50,"edgecolor":"b","linewidth":1}
                  ,plot_kws={'scatter_kws':{'alpha':0.1}}
                 ,diag_kws={"shade":True})
    plt.show()

    g = sns.PairGrid(feat[sL+['b_latency']],hue="b_latency")
    g = g.map_diag(sns.kdeplot, lw=3)
    g = g.map_offdiag(sns.kdeplot, lw=1)
    plt.show()
    
    plt.plot(feat['camera_latency'],feat['e2e_latency'],'o')
    plt.show()

if False:
    print('------------------------------produce-joyplots---------------------------')
    feat = featL.copy()
    for t in tL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    fig, axes = joypy.joyplot(feat,column=tL,xlim='own',ylim='own',figsize=(12,6),alpha=.5)#,colormap=plt.cm.Blues)
    plt.show()    
    
if False:
    print('----------------------------synchrony------------------------------')
    featL = pd.read_csv(fName,compression="gzip")
    y = s_s.interpMissing(featL['camera_latency'])

    feat = featL.copy()
    for t in tL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    y = s_s.interpMissing(feat['camera_latency'])
    y2 = s_s.interpMissing(feat['vehicle_ping'])
    sync = s_s.synchrony(y,y2,period=16,isPlot=True,labV=["camera","ping"])
    plt.show()
    
    importlib.reload(s_s)
    syncL = pd.DataFrame()
    for t in tL:
        y2 = s_s.interpMissing(featL[t])#[:30000]
        sync = s_s.synchrony(y,y2,period=16)
        syncL.loc[:,t] = sync

    med = syncL.median()
    med = med.reindex(med.abs().sort_values(ascending=False).index)
    plt.bar(med.index,med)
    plt.ylabel("correlation")
    plt.xticks(rotation=15)
    plt.show()

    sL = ['force_lat','wheel_speed','vehicle_ping','rtp_lost','modem_rtt','modem_tx','camera_jitter','room_ram','room_cpu','e2e_latency','joystick_latency']
    melt = syncL[sL].melt()
    melt = melt.replace(float('-inf'),float('nan')).replace(float('inf'),float('nan')).dropna()
    ax = sns.violinplot(x="variable", y="value", data=melt)
    plt.show()

if False:
    print('-----------------------cross-correlation----------------------------')
    feat = featL.copy()
    for t in tL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    importlib.reload(s_s)
    y1 = s_s.interpMissing(feat['camera_latency'])
    y2 = s_s.interpMissing(feat['vehicle_ping'])
    xmax, xcor = s_s.xcorMax(y1,y2,period=64,isPlot=True)
    xfft, xcor = s_s.xcorFft(y1,y2)
    plt.show()
    

    importlib.reload(s_s)
    syncL = {}
    for t in tL:
        y2 = s_s.interpMissing(featL[t])
        res, sync = s_s.xcorDecay(y,y2,period=64)
        syncL[t] = sum(sync)/len(sync)
        plt.plot(sync,label=t)
    plt.legend()
    plt.show()

    syncL = pd.Series(syncL)
    syncL = syncL.reindex(syncL.abs().sort_values(ascending=False).index)
    plt.bar(syncL.index,syncL)
    plt.ylabel('persistance')
    plt.xticks(rotation=15)
    plt.show()

    importlib.reload(s_s)
    X = s_s.interpMissingMatrix(featL[tL])
    cM = s_s.decayM(X)
    cM[abs(cM) < 0.05] = 0.
    t_v.plotHeatmap(cM,tL,vmin=-0.5,vmax=0.5)
    plt.title("persistance of features")
    plt.yticks(rotation=15)
    plt.xticks(rotation=35)
    plt.show()

if False:
    print('-----------------------spectrum-autocorrelation------------------')
    feat1 = pd.read_csv(baseDir + "rem/raw/telemetry_1sec.csv.gz",compression="gzip")
    feat2 = pd.read_csv(baseDir + "rem/raw/spike_list.csv.gz",compression="gzip")
    y1 = s_s.interpMissing(feat1['camera_latency'])
    y1 = np.array(y[:30000])
    y2 = s_s.interpMissing(feat2['e2e_latency'])
    y2 = np.array(y2[:30000])
    s_s.spectrum(y1,isPlot=True,lab="latency")
    s_s.spectrum(y2,isPlot=True,lab="spike")
    plt.show()


if False:
    print('---------------------------------phase-shift-----------------------------')
    xL = ['wheel_speed','vehicle_ping','rtp_lost','rtp_late','modem_rx','modem_tx','camera_jitter','vehicle_ram','vehicle_cpu']
    tL = xL + yL

    feat = serAv.copy()
    setL = feat['from_peak'].abs() < 12
    feat = feat.loc[setL,:]
    for t in tL:
        feat.loc[:,t] = s_s.serSmooth(feat[t],width=4,steps=10)
        feat.loc[:,t] = t_r.norm(feat[t])
    t_v.plotTimeSeries(feat[tL],t=feat['from_peak'])
    plt.xlabel("seconds")
    plt.show()
    t = 'joystick_latency'
    t = 'rtp_lost'
    importlib.reload(s_s)
    for t in tL:
        y1 = s_s.interpMissing(feat['camera_latency'])
        y2 = s_s.interpMissing(feat[t])
        fig, ax = plt.subplots(6,1)
        ax[0].plot(feat['from_peak'],y1,label="camera_latency")
        ax[0].plot(feat['from_peak'],y2,label=t)
        ax[0].legend()
        delay = s_s.maxLag(y1,y2,isPlot=True,ax=ax[1])
        delay = s_s.groupDelay(y1,y2,isPlot=True,ax=ax[2])
        delay = s_s.phaseLag(y1,y2,isPlot=True,ax=ax[3])
        delay = s_s.xcorMax(y1,y2,isPlot=True,ax=ax[4])
        delay = s_s.xcorFft(y1,y2,isPlot=True,ax=ax[5])
        plt.show()

    importlib.reload(s_s)
    lagM = s_s.delayM(feat[tL])
    #lagM[np.abs(lagM) < 0.5] = 0.
    #lagM = lagM/1000.
    importlib.reload(t_v)
    t_v.plotHeatmap(lagM,tL,vmin=-4,vmax=4)
    plt.title("phase lag between features")
    plt.yticks(rotation=15)
    plt.xticks(rotation=35)
    plt.show()
    
    feat = featL.copy()
    setL = feat['from_peak'].abs() < 12
    for t in tL:
        feat.loc[:,t] = s_s.serSmooth(feat[t],width=5,steps=12)
        feat.loc[:,t] = t_r.norm(feat[t])
    feat = feat.loc[setL,:]
    offL = []
    for i,g in feat.groupby('series'):
        setL = g['from_peak'].abs() < 12
        X = s_s.interpMissingMatrix(g.loc[setL,tL])
        if len(X) == 0: continue
        y1 = X['camera_latency']
        off = {}
        for t in tL:
            y2 = X[t]
            offset, xcor = s_s.maxLag(y1,y2)
            off[t] = offset
        offL.append(off)
    offL = pd.DataFrame(offL)
    plt.title('phase lag for single series')
    offL.boxplot()
    plt.xticks(rotation=15)
    plt.show()
    
    melt = offL.melt()
    melt = melt.replace(float('-inf'),float('nan')).replace(float('inf'),float('nan')).dropna()
    ax = sns.violinplot(x="variable", y="value", data=melt)
    plt.show()
    
    corD = []
    for i,g in featL.groupby('series'):
        cor = {}
        for t in tL:
            y1 = s_s.interpMissing(g['camera_latency'])
            y2 = s_s.interpMissing(g[t])
            # y1 = s_s.serSmooth(y1)
            # y2 = s_s.serSmooth(y2)
            offset, xcor = s_s.xcorSer(y1,y2)
            cor[t] = offset
        corD.append(cor)
    corD = pd.DataFrame(corD)
    corD.boxplot()
    plt.title('cross correlation')
    plt.xticks(rotation=15)
    plt.show()
    med = corD.mean()
    med = med.reindex(med.abs().sort_values(ascending=False).index)
    plt.bar(med.index,med)
    plt.xticks(rotation=15)
    plt.show()

    cM = s_s.delayM(X,period=12)
    cML.append(cM)


    sL = ['force_lat','wheel_speed','vehicle_ping','rtp_lost','modem_rtt','modem_tx','camera_jitter','room_ram','room_cpu','e2e_latency','joystick_latency']
    melt = syncL[sL].melt()
    melt = melt.replace(float('-inf'),float('nan')).replace(float('inf'),float('nan')).dropna()
    ax = sns.violinplot(x="variable", y="value", data=melt)
    plt.show()


if False:
    print('-----------------------------convolution--------------------------------')
    y1 = s_s.interpMissing(featL['camera_latency'])
    y2 = s_s.interpMissing(featL['camera_latency'].shift(0))
    t_response = featL['second'].values
    m_full = np.convolve(y1,y2,mode='full')
    t_full = np.linspace(t_response[0]-min(t_response),t_response[-1]-min(t_response),len(m_full))
    m_full /= np.trapz(m_full,x=t_full)
    plt.scatter(t_full[::2], m_full[::2], label='Full')
    plt.show()

    
if False:
    print('------------------pca-reduction-----------------------------')
    x, xv = t_s.calcPCA(X)
    pcaL = pd.DataFrame(x,columns=range(len(xL)))
    pcaL = pd.concat([pcaL,pd.Series(y)],axis=1)
    t_v.plotCorr(pcaL,labV=list(pcaL.columns))
    plt.yticks(rotation=15)
    plt.xticks(rotation=15)
    plt.show()

if False:
    print('------------------session-id-influence-----------------------')
    sL = ['room_ram','room_cpu','vehicle_ram','vehicle_cpu'] + ['rtp_lost'] + ['session_time']
    t_v.plotCorr(featL[sL].dropna(),labV=sL)
    plt.yticks(rotation=15)
    plt.xticks(rotation=35)
    plt.show()

    plt.scatter(featL['session_time'],featL['camera_latency'],marker="+")
    plt.show()

    sL = ['room_ram','room_cpu','vehicle_ram','vehicle_cpu'] + ['rtp_lost'] + ['camera_latency']
    feat = featL[:10000]
    feat.loc[:,'b_session'], _ = t_r.binOutlier(feat['session_time'],nBin=2)
    for t in sL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    sns.pairplot(feat[sL+['b_session']],hue='b_session',kind="reg",diag_kind="kde",markers="+"
                 #,plot_kws={"s":50,"edgecolor":"b","linewidth":1}
                  ,plot_kws={'scatter_kws':{'alpha':0.1}}
                 ,diag_kws={"shade":True})
    plt.show()

    import matplotlib.cm as cmx
    import matplotlib 
    feat = featL.copy()
    cL = ['Oranges','Blues','Reds','Greens','Purples','RdPu','GnBu']
    sL = ['vehicle_ram','room_ram'] #+ ['camera_latency']
    #for t in sL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    cs = s_s.interpMissing(feat['session_time'])
    for i,t in enumerate(sL):
        cm = plt.get_cmap(cL[i])
        cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        plt.scatter(feat['session_time'],feat[t],c=scalarMap.to_rgba(cs),marker="+",label=t)
    # scalarMap.set_array(cs)
    # fig.colorbar(scalarMap)
    plt.xlabel("session_time")
    plt.ylabel("normalized feature")
    plt.legend()
    plt.show()

    feat.loc[:,"spike"] = 1.*(feat['camera_latency'] > 200)
    feat.loc[:,"session_time"] = s_s.interpMissing(feat['session_time'])
    feat.loc[:,"session_time"] = feat['session_time'].apply(lambda x: int(x))
    featG = feat.groupby('session_time').agg(np.mean).reset_index()
    
    plt.scatter(featG['session_time'],featG['spike'])
    plt.xlabel('session_time')
    plt.ylabel('spike portion')
    plt.show()
    
if False:
    print('----------------------predictability------------------------')
    #y, _ = t_r.binVector(y,nBin=7,threshold=0.5)
    xL = ['object_distance','force_lat','force_lon','yaw_rate','steering_wheel','steering_angle','wheel_speed','vehicle_ping','rtp_lost','modem_rx','camera_jitter','vehicle_ram','vehicle_cpu']
    feat = featL.copy()
    tL = xL + yL
    for t in tL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    b = 1.*(feat['camera_latency'] > 0.5)
    #b, _ = t_r.binOutlier(featL['camera_latency'],nBin=3)
    X = s_s.interpMissingMatrix(feat[xL])
    print(np.unique(b,return_counts=True))

    mod = t_l.modelList(paramF=baseDir+"rem/train/weath_"+"camera"+".json")
    mod.get_params()
    importlib.reload(t_l)
    importlib.reload(tlib)
    tMod = tlib.trainMod(X.values,b)
    mod, trainR = tMod.loopMod(paramF=baseDir+"rem/train/weath_"+"camera"+".json",test_size=.4)
    tMod.plotRoc()
    y_pred = mod.predict(X) 
    importlib.reload(t_v)
    cm = t_v.plotConfMat(b.values,y_pred)
    plt.show()
    t_v.plotHeatmap(cm/cm.sum())
    plt.title("confusion matrix")
    plt.show()

if False:
    print('--------------------------use-train-feature-------------------------------')
    feat = featL.copy()
    for t in xL: feat.loc[:,t] = t_r.normPercentile(feat[t],perc=[5,95])
    y = s_s.interpMissing(featL['camera_latency'])
    X = s_s.interpMissingMatrix(feat[xL])
    b, _ = t_r.binOutlier(y,nBin=3)
    xL = ['object_distance','brake_pressure','force_lat','force_lon','yaw_rate','steering_wheel','steering_angle','wheel_speed','vehicle_ping','rtp_lost','rtp_late','modem_rtt','modem_rx','modem_tx','camera_jitter','room_ram','room_cpu','vehicle_ram','vehicle_cpu']
    importlib.reload(t_f)
    t_f.plotFeatImp(X[xL],b,xL)
    plt.show()

    importlib.reload(t_f)
    dist = t_f.kmeans(X,n_clust=5,isPlot=True)
    t_f.variance(X)
    t_f.chi2(X.values,y.values)
    reg = t_f.regression(X,y,isPlot=True)
    reg = t_f.featureRegularisation(X,b,isPlot=True,method="lasso")

    
