import os, sys, gzip, random, json, datetime, re, io
import pandas as pd
import numpy as np
from scipy import stats as st
import joypy
import matplotlib.pyplot as plt
import geopandas as gpd
import shapely as sh

dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = os.environ['LAV_DIR']

import importlib
import lernia.train_viz as t_v
import geomadi.geo_octree as g_o
gO = g_o.h3tree()

projDir = baseDir + "rem/raw/telemetry/"
#camL = pd.read_csv(baseDir + "rem/raw/spatial_delay.csv.gz")
#camL = pd.read_csv(baseDir + "rem/raw/telemetry_1sec.csv.gz")
#camL = pd.read_csv(baseDir + "rem/raw/spike_deci.csv.gz")
#camL = pd.read_csv(baseDir + "rem/raw/spike_list.csv.gz")
folder = "modem"
camL = pd.read_csv(baseDir + "/rem/raw/telemetry_"+folder+".csv.gz",compression="gzip")

    
if True:
    print('-----------------------spatial-latency--------------------------------------')
    for i in [10,9,8]:
        camL.loc[:,"geohash"] = camL.apply(lambda x: gO.encode(x['longitude'],x['latitude'],precision=i),axis=1)
        camG = camL.groupby('geohash').agg(np.mean).reset_index()
        polyL = camG['geohash'].apply(lambda x: sh.geometry.Polygon(gO.decodePoly(x)))
        camG = gpd.GeoDataFrame(camG,geometry=polyL)
        #camG.to_file(baseDir + "rem/geo/latency_incident=_"+str(i)+".shp")
        camG.to_file(baseDir + "rem/geo/latency_"+str(i)+".shp")

if False:
    camG = gpd.read_file(baseDir + "rem/geo/latency_8.shp")
    xL = [x for x in camG.columns if re.search("_cel",x)]
    # xL = ['object_dis','brake_pres', 'force_lat', 'force_lon', 'yaw_rate', 'steering_w',
    #    'steering_a', 'wheel_spee', 'vehicle_pi', 'rtp_lost', 'rtp_late',
    #    'modem_rtt', 'modem_rx', 'modem_tx', 'camera_jit', 'room_ram',
    #    'room_cpu', 'vehicle_ra', 'vehicle_cp']
    yL = ['camera_lat','joystick_l','e2e_latenc']
    tL = xL + yL
    t_v.plotCorr(camG[tL],labV=tL)
    plt.yticks(rotation=15)
    plt.xticks(rotation=35)
    plt.show()
