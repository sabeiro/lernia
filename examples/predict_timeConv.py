import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
import shapely as sh
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import geomadi.lib_graph as gra
import geomadi.geo_octree as g_o
import geomadi.geo_motion as g_m
import geomadi.geo_enrich as g_e
import lernia.train_viz as t_v
import albio.series_stat as s_s
import lernia.train_shape as shl
import lernia.train_keras as t_k
import lernia.train_reshape as t_r
import importlib
import joypy
refi = pd.read_csv(baseDir + "raw/"+"bast"+"/ref_iso_h.csv.gz",compression="gzip")

import importlib
importlib.reload(t_r)
dlY = t_r.isocalInWeek(refi,isBackfold=True)
YL = t_r.loadMnist()
XL = np.array([x['values'] for x in dlX])
YL = np.array([x['values'] for x in dlY])
ZL = np.load(baseDir+"raw/"+custD+"/dictionary.npy")
ZL = np.array([t_r.applyBackfold(x) for x in ZL])

X, idL, den ,norm = t_k.splitInWeek(sact,idField=idField,isEven=True)
X = t_k.loadMnist()
tK.plotImg()
tK.plotTimeSeries(nline=8)
tK.simpleEncoder(epoch=25)
tK.deepEncoder(epoch=25,isEven=True)
tK.convNet(epoch=50,isEven=True)
encoder, decoder = tK.getEncoder(), tK.getDecoder()

