import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
import shapely as sh
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = os.environ['LAV_DIR']
import geomadi.geo_octree as g_o
import geomadi.geo_motion as g_m
import geomadi.geo_enrich as g_e
import albio.series_stat as s_s
import sklearn as sk
import lernia
import lernia.train_viz as t_v
import lernia.train_shape as shl
import lernia.train_reshape as t_r
import lernia.train_shape as t_s
import lernia.train_feature as t_f
import lernia.train_model as tlib
import lernia.train_modelList as t_l
import importlib
import joypy

print('--------------------load-datasets------------------')

hourL = pd.read_csv(baseDir+"traffic/raw/"+"dep"+"/weather_h.csv.gz",compression="gzip")
poi = pd.read_csv(baseDir+"traffic/raw/dep/poi_10.csv.gz",compression="gzip")

if False:
    print('--------------------histograms-and-outliers------------------')
    hourL.loc[:,"rain"] = (hourL['precipIntensity']>0.)*1.
    vis = np.mean(hourL['visibility'])
    hourL.loc[:,"visible"] = (hourL['visibility']>vis+vis*.01) | (hourL['visibility']<vis-vis*.01)
    hourL.loc[:,"visible"] = hourL.loc[:,"visible"]*1.
    
    tL = ['precipIntensity','precipProbability','temperature','apparentTemperature','dewPoint','humidity','windSpeed','windGust','windBearing','cloudCover','uvIndex','visibility','precipAccumulation','pressure','ozone']
    tL = ['temperature','apparentTemperature','dewPoint','humidity','windSpeed','windGust','windBearing','cloudCover','uvIndex','precipAccumulation','pressure','ozone']
    timeL = hourL[tL]
    for t in tL: timeL.loc[:,t] = t_r.normPercentile(timeL[t],perc=[1,99])
    tL = tL + ['rain','visible']
    timeL = pd.concat([timeL,hourL[['rain','visible']]],axis=1)
    
    timeL.boxplot()
    plt.xticks(rotation=15)
    plt.show()

    tL = ['temperature','apparentTemperature','dewPoint','humidity','windSpeed','windGust','windBearing','cloudCover','precipAccumulation','pressure','ozone']
    t_v.plotCorr(hourL[tL].dropna(),labV=tL)
    plt.yticks(rotation=15)
    plt.xticks(rotation=25)
    plt.show()

    tL = ['temperature','apparentTemperature','humidity','ozone','pressure','windSpeed','windBearing','cloudCover','precipAccumulation','rain']
    fig, axes = joypy.joyplot(timeL,column=tL,xlim='own',ylim='own',figsize=(12,6),alpha=.5)#,colormap=plt.cm.Blues)
    plt.show()
    
    poi[dL].boxplot()
    plt.show()
    
    fig, ax = plt.subplots(1,2)
    y = spaceL['dis_historic']
    t_v.plotHist(y,ax=ax[0])
    ax[1].hist(y)
    plt.show()

if False:
    print('--------------------change-distributions------------------')

    hourL.loc[:,"rain"] = (hourL['precipIntensity']>0.)*1.
    vis = np.mean(hourL['visibility'])
    hourL.loc[:,"visible"] = (hourL['visibility']>vis+vis*.01) | (hourL['visibility']<vis-vis*.01)
    hourL.loc[:,"visible"] = hourL.loc[:,"visible"]*1.
    
    tL = ['precipIntensity','precipProbability','temperature','apparentTemperature','dewPoint','humidity','windSpeed','windGust','windBearing','cloudCover','uvIndex','visibility','precipAccumulation','pressure','ozone']
    tL = ['temperature','apparentTemperature','dewPoint','humidity','windSpeed','windGust','windBearing','cloudCover','uvIndex','precipAccumulation','pressure','ozone']
    timeL = hourL[tL]
    for t in tL: timeL.loc[:,t] = t_r.normPercentile(timeL[t],perc=[1,99])
    tL = tL + ['rain','visible']
    timeL = pd.concat([timeL,hourL[['rain','visible']]],axis=1)
    
    timeL.boxplot()
    plt.xticks(rotation=15)
    plt.show()

    tL = ['temperature','apparentTemperature','dewPoint','humidity','windSpeed','windGust','windBearing','cloudCover','precipAccumulation','pressure','ozone']
    t_v.plotCorr(hourL[tL].dropna(),labV=tL)
    plt.yticks(rotation=15)
    plt.xticks(rotation=25)
    plt.show()

    tL = ['temperature','apparentTemperature','humidity','ozone','pressure','windSpeed','windBearing','cloudCover','precipAccumulation','rain']
    fig, axes = joypy.joyplot(timeL,column=tL,xlim='own',ylim='own',figsize=(12,6),alpha=.5)#,colormap=plt.cm.Blues)
    plt.show()
    
    poi[dL].boxplot()
    plt.show()
    
    fig, ax = plt.subplots(1,2)
    y = spaceL['dis_historic']
    t_v.plotHist(y,ax=ax[0])
    ax[1].hist(y)
    plt.show()

if False:
    print('------------------pca-reduction-----------------------------')
    tL = ['temperature','humidity','ozone','pressure','windSpeed','windBearing','cloudCover']
    tL = ['temperature','humidity','windSpeed','windBearing','cloudCover']
    X = hourL[tL].dropna()
    x, xv = t_s.calcPCA(X)
    y = pd.DataFrame(hourL.loc[X.index,'apparentTemperature'].values,columns=["apparentTemperature"])
    pcaL = pd.DataFrame(x,columns=range(len(tL)))
    pcaL = pd.concat([pcaL,y],axis=1)
    t_v.plotCorr(pcaL,labV=list(pcaL.columns))
    plt.yticks(rotation=15)
    plt.xticks(rotation=15)
    plt.show()

    
if False:
    print('----------------------predictability------------------------')
    tL = ['temperature','apparentTemperature','dewPoint','humidity','windSpeed','windGust','windBearing','cloudCover','uvIndex','visibility','precipAccumulation','pressure','ozone']
    
    tL = ['temperature','apparentTemperature','humidity','ozone','pressure','windSpeed','windBearing','cloudCover','precipAccumulation']
    m = "rain"
    X = t_s.interpMissing(timeL[tL])
    y = s_s.interpMissing(hourL[m])
    #y, _ = t_r.binVector(y,nBin=7,threshold=0.5)
    
    mod = t_l.modelList(paramF=baseDir+"train/weath_"+m+".json")
    mod.get_params()
    
    importlib.reload(t_l)
    importlib.reload(tlib)
    tMod = tlib.trainMod(X,y)
    mod, trainR = tMod.loopMod(paramF=baseDir+"train/weath_"+m+".json",test_size=.4)
    tMod.plotRoc()

if False:
    print('----------------------feature-importance------------------------')
    tL = ['temperature','apparentTemperature','humidity','ozone','pressure','windSpeed','windBearing','cloudCover','precipAccumulation']
    m = "rain"
    X = t_s.interpMissing(timeL[tL])
    X[X<0.] = 0.
    y = s_s.interpMissing(hourL[m])
    t_f.plotFeatImp(X,y,tL)
    plt.show()

if False:
    print('----------------------feature-knock-out---------------------------')
    tL = ['temperature','apparentTemperature','ozone','pressure','windSpeed','windBearing','cloudCover','precipAccumulation',"humidity"]
    m = 'precipIntensity'
    X = timeL[tL].replace(float('nan'),0)
    y = hourL[m]
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

    t_v.plotTimeSeries(X)
    plt.show()

if False:
    print('----------------------boosting----------------------------')
    tL = ['pop_dens','women','foreign','land_use','elder','dis_subway','dis_station','dis_tourism','dis_historic','dis_crossing','dis_tourist','dis_resident']
    dL = [x for x in tL if bool(re.search("dis_",x))]
    spaceL = poi[tL]
    for t in tL: spaceL.loc[:,t] = t_r.normPercentile(spaceL[t],perc=[1,99])
    X = spaceL.replace(float('nan'),0.)
    y = poi['pot']
    y = t_r.normPercentile(y,perc=[1,99])
    y = s_s.interpMissing(y)
    
    reg = lernia.train_model.regressorSingle
    fit_q, corrL = reg(X,y)
    y_pred = fit_q.predict(X)
    y_red = y - y_pred
    fit_q, corrL = reg(X,y_red)
    y_pred2 = fit_q.predict(X)
    y_red2 = y_red - y_pred2

    fig, ax = plt.subplots(1,3)
    t_v.plotHist(y,nBin=20,ax=ax[0],threshold=5.,lab="y")
    t_v.plotHist(y_red,nBin=20,ax=ax[1],threshold=5.,lab="resid")
    t_v.plotHist(y_red2,nBin=20,ax=ax[2],threshold=5.,lab="resid pred")
    plt.show()
    
    resid = pd.DataFrame({"y":y,"y_resid":y_red,"y_resid_pred":y_red2})
    resid.boxplot()
    plt.show()

if False:
    print('----------------------mixed-metrics------------------------')
    tL = ['temperature','humidity','windSpeed','windBearing','cloudCover']
    importlib.reload(t_v)
    X = timeL[tL].fillna(0)
    
    t_v.plotCrossMatrix(X)
    plt.show()

    
if False:
    print('----------------------distribution------------------------')
    tL = ['pop_dens','women','foreign','land_use','elder','dis_subway','dis_station','dis_tourism','dis_historic','dis_crossing','dis_tourist','dis_resident']
    dL = [x for x in tL if bool(re.search("dis_",x))]
    yL = ['activation','pot']
    spaceL = poi[tL]
    for t in tL: spaceL.loc[:,t] = t_r.normPercentile(spaceL[t],perc=[1,99])
    
    fig, axes = joypy.joyplot(spaceL,column=tL,xlim='own',ylim='own',figsize=(12,6),alpha=.5)
    plt.show()

    t_v.plotCorr(poi[tL+yL].dropna(),labV=tL+yL)
    plt.yticks(rotation=15)
    plt.xticks(rotation=25)
    plt.show()


if False:
    print('----------------------predictability------------------------')
    tL = ['pop_dens','women','foreign','land_use','elder','dis_subway','dis_station','dis_tourism','dis_historic','dis_crossing','dis_tourist','dis_resident']
    m = "pot"
    X = t_s.interpMissing(poi[tL])
    y = s_s.interpMissing(poi[m])
    y, _ = t_r.binVector(y,nBin=7,threshold=0.05)
    mod = t_l.modelList(paramF=baseDir+"train/dens_"+m+".json")
    mod.get_params()
    importlib.reload(t_l)

    importlib.reload(tlib)
    tMod = tlib.trainMod(X,y)
    mod, trainR = tMod.loopMod(paramF=baseDir+"train/dens_"+m+".json",test_size=.4)

    tMod.plotRoc()

if False:
    print('---------------------------feature-regularisation----------------------')
    tL = ['zip5', 'pop_dens', 'women', 'foreign','land_use', 'elder', 'dis_subway', 'dis_station', 'dis_tourism','dis_historic', 'dis_crossing', 'dis_tourist', 'dis_resident']
    X = poi[tL].fillna(0)
    X = t_f.selectNumeric(X)
    tL = X.columns
    y = poi['pot'].fillna(0).apply(int)
    importlib.reload(t_f)
    featLa = t_f.featureRegularisation(X,y,method="lasso")
    featRi = t_f.featureRegularisation(X,y,method="ridge")
    featRe = t_f.regression(X,y)
    reg = sk.linear_model.ElasticNet()
    featEl = reg.fit(X.values,y.values).coef_
    reg = sk.linear_model.Lasso()
    featLs = reg.fit(X.values,y.values).coef_

    featD = pd.DataFrame({"reg_lasso":featLa,"reg_rigde":featRi,"lin_ridge":featRe,"lin_elastic":featEl,"lin_lasso":featLs})
    
    for c in featD.columns:
        plt.bar(tL,featD[c],alpha=0.3,label=c)
    plt.legend()
    plt.xticks(rotation=15)
    plt.show()
    
    

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')

