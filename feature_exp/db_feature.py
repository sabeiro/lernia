#--------------------------------------import-------------------------------------
import os, sys, gzip, random, json, datetime, re, io
import pandas as pd
from pathlib import Path
from scipy import stats as st
from dotenv import load_dotenv
import boto3
from multiprocessing.dummy import Pool as ThreadPool

#-----------------------------------environment------------------------------------
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = os.environ['LAV_DIR']
projDir = baseDir + "/rem/src/feature_exp/"
cred = json.load(open(projDir + "conf/db_table.json"))['latency']
ENV = os.getenv('ENV', None)
if not ENV:
    load_dotenv(baseDir + "rem/credenza/.env")
#------------------------------------local-import----------------------------------
from sawmill import aws_utils as a_u
import importlib
#---------------------------------session-------------------------------------------
session = boto3.session.Session()#profile_name='athena')
athena = boto3.client('athena',region_name='eu-west-1')
s3 = boto3.client('s3')
tableL = ['telemetry','network_log','session','incident']
#---------------------------------definition--------------------------------------
def download_query(query,fileN):
    location = a_u.exec_athena(athena,query,params)
    if location != False:
        print(location)
        bucN, bucK = location.split("/")[2], location.split("/")[-1]
        try:
            s3.download_file(bucN,bucK,fileN + ".csv")
            down = pd.read_csv(fileN + ".csv")
            os.remove(fileN + ".csv")
            s3.delete_object(Bucket = bucN, Key = bucK)
            if len(down) == 0:
                print('empty data frame')
                return
            down.to_csv(fileN + ".csv.gz",compression="gzip",index=False)
        except:
            print('bucket not found')
            return
    
#---------------------------------extract----------------------------------------------
importlib.reload(a_u)
params = {"db":{'Database':'ree-cloud-data-prod'}
          ,"group":'AmazonAthenaPreviewFunctionality'
          ,"result":{'OutputLocation': 's3://' + "ree-cloud-athena" + '/' },"iteration":180}
if False:
    print('----------------------execute-query-folder---------------------')
    query = open(projDir + "queries/" + 'resample_1sec.sql', 'r').read()
    query = open(projDir + "queries/" + 'spatial_delay.sql', 'r').read()
    query = open(projDir + "queries/incident/" + 'incident_list.sql', 'r').read()
    print(query.format(environment='prod',time_horizon_sec=60))
    fileN = baseDir + "rem/raw/incident_list"
    download_query(query,fileN)
    #query = "SELECT * FROM ree_cloud_data_prod." + table + " where year='2020' and month='07' and day='07' LIMIT 100;"
    #query = "SELECT * FROM ree_cloud_data_prod." + table + " where year='2020' and month='07' and day='07' and latency_camera.latency_ms > 0.;"

if False:
    print('-------------------------execute-incident-list---------------------------')
    incL = pd.read_csv(baseDir + "rem/raw/incident_list.csv")
    for i, g in incL.iterrows():
        snapshot = g['snapshot']
        session_id = g['session_id']
        query = open(projDir + "queries/incident/" + 'incident_factors_sec.sql', 'r').read()
        query = query.format(session_id=session_id,environment='prod',snapshot=snapshot,time_horizon_sec=60)
        fileN = baseDir + "rem/raw/incident/incident_sec/incindent_" + str(snapshot)
        download_query(query,fileN)

if True:
    print('------------------------telemetry-features-----------------------------')
    #day = datetime.datetime(2020,9,1)
    day = datetime.datetime(2020,8,1)
    today = datetime.datetime.today()
    td = (today - day).days
    dL = [day + datetime.timedelta(days=x) for x in range(td)]
    dL = [x.strftime("%Y-%m-%d") for x in dL]
    hL = []
    for day in dL:
        hL = hL + [day + "T" + "%02d" % x for x in range(24)]
    h = hL[0]
    h = "2020-09-08T04"
    table = 'telemetry'
    def down_hour(h):
        print(h)
        query = open(projDir + "/queries/" + 'resample_deci.sql', 'r').read()
        #query = open(projDir + "/queries/" + 'camera_index_deci.sql', 'r').read()
        query = query.format(ts=h,environment="prod")
        fileN = baseDir + "/rem/raw/"+table+"/modem/"+table+"_"+h
        download_query(query,fileN)

    pool = ThreadPool(4)
    results = pool.map(down_hour, hL)
    pool.close()
    pool.join()

if False:
    print('------------------------network-incident-----------------------------')
    featL = pd.read_csv(baseDir + "/rem/raw/spike_list_deci.csv.gz",compression="gzip")
    featD = []
    for i,g in featL.groupby('series'):
        vehicle = g['vehicle_id'].values[0]
        snapshot = int(g['second'].values[0])
        query = open(projDir + "queries/" + 'network_log.sql', 'r').read()
        query = query.format(environment="prod",vehicle_id=vehicle,snapshot=snapshot,time_horizon_sec=8*60)
        fileN = baseDir + "/rem/raw/incident/incident_deci/incindent_" + str(snapshot)
        location = a_u.exec_athena(athena,query,params)
        if not location:
            continue
        print(location)
        try:
            bucN, bucK = location.split("/")[2], location.split("/")[-1]
            s3.download_file(bucN,bucK,fileN + ".csv")
        except:
            print('bucket not found')
            continue
        down = pd.read_csv(fileN + ".csv")
        if len(down) == 0: continue
        os.remove(fileN + ".csv")
        gname = down.pivot_table(index="timebucket",columns="name")#.reset_index()
        gname.rename(columns={"timebucket-":"timebucket"},inplace=True)
        gname.columns = [x[0] +"-"+ x[1] for x in gname.columns]
        feat = g.merge(gname,left_on="second",right_on="timebucket",how="left")
        feat.to_csv(fileN + ".csv.gz",compression="gzip",index=False)
        featD.append(feat)

    featL = pd.concat(featD)
    featL.to_csv(baseDir + "/rem/raw/spike_deci.csv.gz",compression="gzip",index=False)

if False:
    print('------------------------download-day-----------------------------')
    day = "2020-08-04"
    hL = [day + "T" + "%02d" % x for x in range(24)]
    table = 'telemetry'
    def down_hour(h):
        print(h)
        query = "SELECT * FROM ree_cloud_data_prod." + table + " where ts='"+h+"';"# LIMIT 1000;"
        fileN = baseDir + "/rem/raw/"+table+"/"+table+"_"+h
        download_query(query,fileN)

    pool = ThreadPool(4)
    results = pool.map(down_hour, hL)
    pool.close()
    pool.join()

        
if False:
    print('-------------------------------get-sample-parquets--------------------------------------')
    bucketL = s3.list_buckets()['Buckets']
    buckN = [x['Name'] for x in bucketL]
    contL = s3.list_objects(Bucket='ree-cloud-data-prod')['Contents']
    fL = [x['Key'] for x in contL]
    #buck = s3_resource.Bucket(cred['db'])

    for t in tableL:
        f = [x for x in fL if bool(re.search(t,x))][-1]
        d = "/".join(f.split("/")[:-1]) + "/"
        fName = "_".join(f.split("/"))
        s3.download_file(cred['db'], f, baseDir + 'rem/log/' + fName)
    
