import os, sys, gzip, random, json, datetime, re, io
import pandas as pd
import numpy as np
from scipy import stats as st
import joypy
import matplotlib.pyplot as plt

dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = os.environ['LAV_DIR']

import importlib
import lernia.train_viz as t_v
import albio.series_stream

if True:
    print('-----------------------load-data------------------------------')
    fL = os.listdir(baseDir + "rem/raw/telemetry")
    telL = []
    f = fL[4]
    for f in fL:
        tel = pd.read_csv(baseDir + "rem/raw/telemetry/" + f)
        telL.append(tel)
    tel = pd.concat(telL).reset_index()
    tel = tel[tel['timestamp_ms'] == tel['timestamp_ms']]
    tel.loc[:,"second"] = tel['timestamp_ms'].apply(lambda x: x.split(".")[0])
    tel.loc[:,'ts'] = [datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S.%f") for x in tel['timestamp_ms']]
    tel.loc[:,"ms"] = [float(x[20:23]) for x in tel['timestamp_ms']]
    tL = control + board + asset
    sensorL = np.array(tL)
    tel.loc[:,'sensor'] = tel[tL].apply(lambda x: sensorL[~x.isna()][0],axis=1)
    tel.loc[:,"bin"] = tel['ms'].apply(lambda x: int(x/5.))
    tel.loc[:,"freq"] = 1.
    sens = tel.pivot_table(index="sensor",columns="bin",values="freq",aggfunc=np.sum)
    sens.replace(float('nan'),0.,inplace=True)
    dt = len(np.unique(tel['second']))
    order = sens.sum(axis=1).sort_values(ascending=False)/dt
    sens = sens.loc[order.index]
    xL = list(sens.columns*5)[::10]
    sens = sens/dt*200.
    print(sens.mean(axis=1))

    # frequency matrix plot
    plt.imshow(sens,aspect=3.)
    plt.yticks(ticks=range(sens.shape[0]),labels=sens.index)
    plt.xticks(ticks=[x/5 for x in xL],labels=xL)
    plt.xlabel("milliseconds")
    plt.show()

    # frequency bar plot
    plt.bar(order.index[:15],order[:15])
    plt.ylabel("frequency Hz")
    plt.xticks(rotation=15)
    plt.show()
    

    fig, ax = plt.subplots(2,2)
    for i,g in tel.groupby("sensor"):
        a = ax[0][0]
        if bool(re.search("vehicle",i)): a = ax[0][1]
        if bool(re.search("control",i)): a = ax[1][0]
        if bool(re.search("modem",i)): a = ax[1][1]
        if bool(re.search("frame",i)): a = ax[1][1]
        if bool(re.search("steering",i)): a = ax[1][1]
        a.hist(g['ms'],bins=100,label=i,alpha=.3)
    for i in range(2):
        for j in range(2):
            ax[i][j].legend()
            # ax[i][j].xlabel('events')
            # ax[i][j].xlabel('milliseconds')
    plt.show()

    sens = tel.groupby('sensor').agg(len).reset_index().sort_values('date',ascending=False)
    sens.loc[:,"frequency"] = sens['ts']/dt
    plt.bar(sens['sensor'],sens['frequency'])
    plt.xticks(rotation=15)
    plt.ylabel("frequency Hz")
    plt.show()
    
    tel.loc[:,"processing"] = tel['meta'].apply(lambda x: x.split("=")[2].split(",")[0])
    tel.loc[:,"processing"] = [datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S.%f") for x in tel['processing']]
    tel.loc[:,"streaming"] = tel['meta'].apply(lambda x: x.split("=")[1].split(",")[0])
    tel.loc[:,"streaming"] = [datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S.%f") for x in tel['streaming']]
    tel.loc[:,"delta_proc"] = tel['processing'] - tel['ts']
    tel.loc[:,"delta_stream"] = tel['streaming'] - tel['ts']
    tel.loc[:,"delta_proc"] = [x.seconds + x.microseconds/1000000. for x in tel['delta_proc']]
    tel.loc[:,"delta_stream"] = [x.seconds + x.microseconds/1000000. for x in tel['delta_stream']]
    plt.hist(tel['delta_proc'],bins=40,label="delay processing",alpha=0.3)
    plt.hist(tel['delta_stream'],bins=40,label="delay streaming",alpha=0.3)
    plt.legend()
    plt.xlabel("seconds delay")
    plt.show()
