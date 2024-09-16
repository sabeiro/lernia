import os, sys, gzip, random, json, datetime, re, io
import pandas as pd
import numpy as np
from scipy import stats as st
import joypy
import matplotlib.pyplot as plt

dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = os.environ['LAV_DIR']

import lernia.train_viz as t_v
import albio.series_stat as s_s
import deep_lernia.train_keras as t_k
import deep_lernia.train_deep as t_d
import lernia.train_reshape as t_r
import importlib

featL = pd.read_csv(baseDir + "/rem/raw/incident_series.csv.gz")

if False:
    print('-------------------------visualize-series-----------------------')
    for i,g in featL.groupby('series'):
        plt.plot(g['camera_latency'],label=i)
    plt.ylabel("camera latency")
    plt.show()
    
if True:
    print('---------------------prediction-of-latency----------------------')
    xL = ['force_lat','force_lon','yaw_rate','steering_wheel','steering_angle','wheel_speed','vehicle_ping','rtp_lost','modem_rx','camera_jitter','vehicle_ram','vehicle_cpu']
    yL = ['camera_latency','joystick_latency', 'e2e_latency']
    importlib.reload(t_k)
    importlib.reload(t_d)
    y = s_s.interpMissing(featL['camera_latency'])
    #y = s_s.serSmooth(y,width=2,steps=3)
    X = s_s.interpMissingMatrix(featL[xL])
    mod = t_d.regressor(X.values,y)
    mod.setRegressor("three_layer",layer=[10,6,3])
    mod.scale()
    mod.printModel()
    kpi = mod.train(epochs=40)
    fig, ax = plt.subplots(2,2)
    ax = ax.flatten()
    j = 0
    for i,g in featL.groupby('series'):
        k = j % 3
        y = s_s.interpMissing(g['camera_latency'])
        X = s_s.interpMissingMatrix(g[xL])
        X.dropna(inplace=True)
        if len(X) == 0: continue
        mod.setX(X.values), mod.setY(y)
        mod.scale()
        kpi = mod.train(epochs=50)
        ax[k].set_title(i)
        mod.plotPrediction(X=X,ax=ax[k])
        j = j + 1
        if j > 12: break

    ax[3].set_title("history")
    mod.plotHistory(ax[3])
    plt.show()
    
    
if False:
    from keras.utils import to_categorical
    y_binary = to_categorical(y)
    model = Sequential()
    model.add(Dense(10, input_shape=(20,), activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y_binary, epochs=500, batch_size=10)
    y_pred = np.array(model.predict(X) > 0.5)[:,1]
    print(sum( abs(y*1-y_pred*1) ) )
    
    import geomadi.train_modelList as t_m
    import geomadi.train_lib as t_l
    from sklearn.metrics import confusion_matrix
    tM = t_l.trainMod(X,y)
    mL = t_l.modelList()
    clf = t_m.binCla[1]
    mod = clf.fit(X,y)
    y_pred = mod.predict(X)
    print(sum( abs(y*1-y_pred*1) ) )
    cm = confusion_matrix(y,y_pred)
    cm = np.array(cm)
    print(cm)
    from sklearn.externals import joblib
    joblib.dump(clf,'out/svc_model.pkl', compress=9)
    
    if False:
        print("on diagonal %.2f" % (sum(cm.diagonal())/sum(sum(cm))) )
        plt.xlabel("prediction")
        plt.ylabel("score")
        plt.imshow(cm)
        plt.show()

