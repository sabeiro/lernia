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
from sawmill import aws_utils as a_u
import lernia.train_reshape as t_r
import lernia.train_viz as t_v
import lernia.train_shape as t_s
import lernia.train_feature as t_f
import lernia.train_model as tlib
import lernia.train_modelList as t_l
import deep_lernia.train_keras as t_k
import deep_lernia.train_longShort as tls
import albio.series_stat as s_s
import importlib
importlib.reload(s_s)

feat = pd.read_csv(baseDir + "rem/raw/prediction/telemetry.csv.gz").ffill()
feat.index = feat['deci'].apply(lambda x: datetime.datetime.fromtimestamp(x/10))
xL = ['room_jitter', 'vehicle_jitter', 'camera_jitter','vehicle_ping', 'joystick_freq', 'wheel_speed', 'radar_speed','force_lateral', 'force_longitudinal', 'ram_usage','cpu_usage']
yL = ['camera_latency', 'joystick_latency', 'e2e_latency']
tL = xL + yL
for i in ['room_jitter','vehicle_jitter','vehicle_ping','joystick_freq'] + yL:
    feat.loc[:,i] = np.log(1e-6 + np.abs(feat[i]))

feat.loc[:,'class'], _ = t_r.binOutlier(feat['camera_latency'],nBin=4)
y = s_s.interpMissing(feat['camera_latency'])
X = s_s.interpMissingMatrix(feat[xL])
t = feat.index
l = list(range(len(y)))
    
if False:
    t_v.plotTimeSeries(feat[tL])
    plt.show()

if False:
    print('--------------------histograms-and-outliers------------------')
    tL1 = ['wheel_speed', 'radar_speed', 'force_lateral', 'force_longitudinal', 'ram_usage','room_jitter','vehicle_jitter', 'joystick_freq','vehicle_ping']
    feat = pd.read_csv(baseDir + "rem/raw/prediction/telemetry.csv.gz").ffill()
    featL = feat[tL1]
    for t in featL.columns: featL.loc[:,t] = t_r.normPercentile(featL[t],perc=[1,99])    
    fig, ax = plt.subplots(1,2)
    featL.boxplot(ax=ax[0])
    featL = feat[tL1]
    for i in ['room_jitter','vehicle_jitter','vehicle_ping','joystick_freq'] :
        featL.loc[:,i] = np.log(np.abs(featL[i]))
    for t in featL.columns: featL.loc[:,t] = t_r.normPercentile(featL[t],perc=[1,99])    
    featL.boxplot(ax=ax[1])
    for a in ax.flatten():
        for tick in a.get_xticklabels():
            tick.set_rotation(15)
    plt.xticks(rotation=15)
    plt.show()

    t_v.plotCorr(featL[tL].dropna(),labV=tL)
    plt.yticks(rotation=15)
    plt.xticks(rotation=35)
    plt.show()

    fig, axes = joypy.joyplot(featL,column=tL,xlim='own',ylim='own',figsize=(12,6),alpha=.5)#,colormap=plt.cm.Blues)
    plt.show()

    tL1 = ['vehicle_ping', 'joystick_freq', 'wheel_speed', 'radar_speed', 'force_lateral', 'force_longitudinal', 'ram_usage']
    sns.pairplot(feat[tL1+['class']],hue='class',diag_kind="kde",markers="+",plot_kws=dict(s=50,edgecolor="b", linewidth=1),diag_kws=dict(shade=True))
    plt.show()

    g = sns.PairGrid(feat[tL1+['class']],hue="class")
    g = g.map_diag(sns.kdeplot, lw=3)
    g = g.map_offdiag(sns.kdeplot, lw=1)
    plt.show()
    
    plt.plot(feat['camera_latency'],feat['e2e_latency'],'o')
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
    print('----------------------predictability------------------------')
    #y, _ = t_r.binVector(y,nBin=7,threshold=0.5)
    b = feat['class']
    b, _ = t_r.binOutlier(feat['camera_latency'],nBin=2)
    mod = t_l.modelList(paramF=baseDir+"rem/train/weath_"+"camera"+".json")
    mod.get_params()
    importlib.reload(t_l)
    importlib.reload(tlib)
    tMod = tlib.trainMod(X.values,b)
    mod, trainR = tMod.loopMod(paramF=baseDir+"rem/train/weath_"+"camera"+".json",test_size=.4)
    tMod.plotRoc()

if False:
    importlib.reload(t_k)
    importlib.reload(tls)
    y1 = y[1:] - y[:-1]
    y1 = y
    #y1 = y1[:200]
    y1 = X#[:200]
    X1 = y1
    tK = tls.timeSeries(X1,y1)
    tK.scale(feature_range=(-1,1))    
    tK.lstmBatch(1, 4)
    kpi = tK.train(batch_size=1,nb_epoch=20)
    tK.plotPrediction()
    plt.show()
    tK.plotHistory()
    plt.show()

    #perfL = tK.featKnockOut()
    tK.plotFeatImportance(perfL)
    plt.show()


    X_train, X_test, y_train, y_test = tK.splitSet(tK.X,tK.y,shuffle=False)
    tK.model.predict(X_train, batch_size=batch_size)
    y_pred = tK.predict(X_test,batch_size=batch_size)

    
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)
 
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]
 
    
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df
 
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled
 
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]
 
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]
 
diff_values = difference(y, 1)
#diff_values = y
df = pd.DataFrame(diff_values)
lag = 1
columns = [df.shift(i) for i in range(1, lag+1)]
columns.append(df)
df = pd.concat(columns, axis=1)
df.fillna(0, inplace=True)
supervised = df
supervised_values = supervised.values
n_pred = 100
train, test = supervised_values[0:-n_pred], supervised_values[-n_pred:]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train)
train = train.reshape(train.shape[0], train.shape[1])
train_scaled = scaler.transform(train)
test = test.reshape(test.shape[0], test.shape[1])
test_scaled = scaler.transform(test)
neurons = 4
batch_size = 1
nb_epoch = 2
X, y = train_scaled[:, 0:-1], train_scaled[:, -1]
X = X.reshape(X.shape[0], 1, X.shape[1])
model = Sequential()
model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(nb_epoch):
    model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
    model.reset_states()

train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
model.predict(train_reshaped, batch_size=1)
predictions = list()
for i in range(len(test_scaled)):
    print(i)
    X1, y1 = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(model, 1, X1)
    yhat = invert_scale(scaler, X1, yhat)
    yhat = inverse_difference(y, yhat, len(test_scaled)+1-i)
    predictions.append(yhat)
    expected = y[len(test_scaled) + i + 1]
    print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
 
plt.plot(y[-n_pred:])
plt.plot(predictions)
plt.show()


