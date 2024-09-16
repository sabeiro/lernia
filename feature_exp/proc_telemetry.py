#--------------------------------------import-------------------------------------
import os, sys, gzip, random, json, datetime, re, io
import pandas as pd
from pathlib import Path
from scipy import stats as st
#-----------------------------------environment------------------------------------
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = os.environ['LAV_DIR']
projDir = baseDir + "/rem/src/feature_exp/"
#-----------------------------pyspark-import----------------------
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-1.8.0-openjdk-amd64'
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.functions import to_utc_timestamp, from_utc_timestamp
from pyspark.sql.functions import date_format
from pyspark.sql import functions as func
from pyspark.sql.functions import col
from pyspark.sql.window import Window

conf = (SparkConf()
    .setMaster("yarn-client")
    .setAppName("proc library")
    .set("spark.deploy-mode", "cluster"))
conf.set("spark.executor.memory", "10g")
conf.set("spark.executor.cores", "10")
conf.set("spark.executor.instances", "2")
conf.set("spark.driver.maxResultSize", "10g")
conf.set("spark.driver.memory", "10g")
conf.set("spark.sql.crossJoin.enabled", "true")

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
sc.setLogLevel("ERROR")

print('--------------------------load-data----------------------------')
df = sqlContext.read.parquet(baseDir + "/rem/log/" + 'telemetry')
df = df.orderBy(['session_id','timestamp_ms'])
print('--------------------------bin-on-deci-seconds-------------------')
df = df.withColumn('deci',(func.col('timestamp_ms').cast('double')*10.).cast('long'))
ddf = df.groupBy(["session_id","deci"]).agg(
    func.mean('longitude').alias('longitude'),
    func.mean('latitude').alias('latitude'),
    func.mean('control_room_jitter.jitter_ms').alias('room_jitter'),
    func.mean('vehicle_jitter.jitter_ms').alias('vehicle_jitter'),
    func.mean('jitter.jitter_ms').alias('camera_jitter'),
    func.mean('vehicle_ping.ping_ms').alias('vehicle_ping'),
    func.mean('joystick_hz.joystick_hz').alias('joystick_freq'),
    func.mean('wheel_speed.mean_km_per_hour').alias('wheel_speed'),
    func.mean('radar_track.object_relative_speed_m_per_sec').alias('radar_speed'),
    func.mean('vehicle_physics.lateral_force_m_per_sec_squared').alias('force_lateral'),
    func.mean('vehicle_physics.longitudinal_force_m_per_sec_squared').alias('force_longitudinal'),
    func.mean('vehicle_physics.yaw_rate_deg_per_sec').alias('yaw_rate'),
    func.mean('vehicle_ram_usage.ram_usage_percent').alias('ram_usage'),
    func.mean('vehicle_cpu_usage.cpu_usage_percent').alias('cpu_usage'),
    func.mean('bitrate_adjusted.bitrate_bps').alias('bit_rate'),
    func.mean('vehicle_temperature_readings.b_cpu_temperature').alias('cpu_temperature'),
    func.mean('modem_report.modem_report_data[0].tunnel_rtt').alias('modem0_tunnel'),
    func.mean('modem_report.modem_report_data[1].tunnel_rtt').alias('modem1_tunnel'),
    func.mean('modem_report.modem_report_data[2].tunnel_rtt').alias('modem2_tunnel'),
    func.mean('modem_report.modem_report_data[3].tunnel_rtt').alias('modem3_tunnel'),
    func.mean('modem_report.modem_report_data[0].tunnel_delta_tx').alias('modem0_delay'),
    func.mean('modem_report.modem_report_data[1].tunnel_delta_tx').alias('modem1_delay'),
    func.mean('modem_report.modem_report_data[2].tunnel_delta_tx').alias('modem2_delay'),
    func.mean('modem_report.modem_report_data[3].tunnel_delta_tx').alias('modem3_delay'),
    func.mean('latency_camera.latency_ms').alias('camera_latency'),
    func.mean('joystick_latency.joystick_latency_ns').alias('joystick_latency'),
    func.mean('e2e_latency.latency_ms').alias('e2e_latency'),
    )
print('---------------------------running-average-------------------------------------')
window = Window.partitionBy('session_id').rowsBetween(-3, 3)
tL = ['room_jitter','camera_jitter','joystick_freq','wheel_speed','radar_speed','force_lateral','force_longitudinal',
      'yaw_rate','ram_usage','cpu_usage','bit_rate','cpu_temperature','camera_latency',
     'joystick_latency','e2e_latency']
for t in tL:
    ddf = ddf.withColumn(t,func.mean(ddf[t]).over(window))


ddf.write.parquet(baseDir + "/rem/log/prediction/telemetry.parquet" )
