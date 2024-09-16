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

import lernia.train_viz as t_v
import albio.series_stat as s_l
import deep_lernia.train_keras as t_k
import deep_lernia.train_deep as t_d
import albio.series_stat as s_s
import importlib

projDir = baseDir + "rem/raw/spike/"
fL = os.listdir(projDir)
camL = pd.DataFrame()
joyL = pd.DataFrame()
for i,f in enumerate(fL):
    feat = pd.read_csv(projDir + f)
    #feat.index = [datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S.%f") for x in feat['resampled_dt']]
    spike = f.split("_")[-1].split(".")[0]
    day = f.split("_")[2]
    cam = feat[['camera_latency']]
    cam.loc[cam['camera_latency'] > 250,'camera_latency'] = 250
    joy = feat[['joystick_latency']]
    cam.columns = ["spike_" + day + "_" + spike]
    joy.columns = ["spike_" + day + "_" + spike]
    camL = pd.concat([camL,cam],axis=1)
    joyL = pd.concat([joyL,joy],axis=1)
    
if False:
    print('---------------------show-spike-stack---------------------------')
    plt.axvline(x=4,linewidth=10,alpha=0.2,color="red")
    cL = list(camL.columns)
    jL = list(joyL.columns)
    for i in range(camL.shape[1]):
        col = t_v.colorL[i]
        plt.plot(camL.index/60.,camL[cL[i]],color=col,label="series_"+ str(i),linewidth=2)
        #plt.plot(joyL.index/60.,joyL[jL[i]]/1000.,color=col,label="joystick_"+ spike)

    plt.xlabel("minutes")
    plt.ylabel("latency (ms)")
    plt.yscale("log")
    plt.legend()
    plt.show()

if False:
    print('-------------calc-power-spectrum-------------')
    cL = list(camL.columns)
    jL = list(joyL.columns)
    psN = 400
    psS = []
    psL = {}
    for i in cL:
        ser = camL[i]
        ser = ser[ser==ser]
        serD = s_l.serDecompose(ser,period=5)
        ps = np.abs(np.fft.fft(serD['run_av']))**2
        psL[i] = np.log(ps[:psN])
    psS = pd.DataFrame(psL)
    plt.imshow(psS.values.T,aspect=12)
    plt.xlabel("modes")
    plt.ylabel("series")
    plt.show()
    
    importlib.reload(s_l)
    serD = s_l.serDecompose(ser,period=5,isPlot=True)

    
if False:
    print('---------------unsupervised-segment-detection------------')
    l = random.choice(camL.columns)
    y = camL[l].values
    serD = s_s.serDecompose(y,period=14)
    segment_len = 16
    slide_len = 2
    segments = []
    for l in camL.columns:
        y = s_s.interpMissing(camL[l].values)
        serD = s_s.serDecompose(y,period=3)
        y = serD['smooth']
        for start_pos in range(0, len(y), slide_len):
            end_pos = start_pos + segment_len
            segment = np.copy(y[start_pos:end_pos])
            if len(segment) != segment_len:
                continue
            segments.append(segment)
    print(len(segments))
    window_rads = np.linspace(0, np.pi, segment_len)
    window = np.sin(window_rads)**2
    windowed_segments = []
    for segment in segments:
        windowed_segment = np.copy(segment) * window
        windowed_segments.append(windowed_segment)

    from sklearn.cluster import KMeans
    clusterer = KMeans(copy_x=True,init='k-means++',max_iter=300,n_clusters=18,n_init=10,precompute_distances='auto',random_state=None,tol=0.0001,verbose=2)
    # clusterer.fit(windowed_segments)
    clusterer.fit(segments)
    centroids = clusterer.cluster_centers_
    nearest_centroid_idx = clusterer.predict(segments)[0]
    nearest_centroid = np.copy(centroids[nearest_centroid_idx])
    
    if False:
        corr = pd.DataFrame(np.corrcoef(np.array(centroids)))
        ax = sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='RdYlGn')
        plt.show()

    if False:
        y1 = segments[0]
        y2 = nearest_centroid
        plt.figure()
        plt.title("corr %.2f rmse %.2f" % (sp.stats.pearsonr(y1,y2)[0],np.sqrt(np.mean((y1-y2)**2))) )
        plt.plot(y1, label="Original segment")
        plt.plot(y2, label="Nearest centroid")
        plt.legend()
        plt.show()
        
    if False:
        clusL = 'abcdefghijklmnopqrstuvwxyz'
        disp = segments
        disp = windowed_segments
        disp = clusterer.cluster_centers_
        plt.figure()
        lL = random.sample(range(len(disp)),len(disp))
        lL = range(len(disp))
        for i in range(3):
            for j in range(6):
                axes = plt.subplot(3,6,6*i+j+1)
                l = lL[min(len(disp)-1,6*i+j)]
                plt.plot(disp[l],label="c " + str(l))
                plt.title(clusL[l])
        #plt.legend()
        plt.tight_layout()
        plt.show()


    

if False:
    print('-------------------sensor-relation-graph------------------')
    import networkx as nx
    cor = corR
    G = nx.Graph()
    for i in cor.index:
        G.add_node(i)
    for i in cor.index:
        for j in cor.index:
            if abs(cor.loc[i,j]) > 0.8:
                if i !=j :
                    G.add_edge(i,j)

    nodS = cor.sum(axis=1)*10
    pos = nx.circular_layout(G)
    pos = nx.spectral_layout(G)
    pos = nx.spring_layout(G)
    labels = {}
    for i in cor.index:
        labels[i] = i
    nx.draw_networkx_edges(G,pos,width=.6,alpha=0.3)
    nx.draw_networkx_nodes(G,pos,node_color='g',node_size=nodS,alpha=0.3,with_labels=True)
    nx.draw_networkx_labels(G,pos,labels,font_size=18)
    plt.show()
    
