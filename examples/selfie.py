import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
import cv2
import lernia.train_viz as t_v
import lernia.train_shape as shl
import lernia.train_keras as t_k
import lernia.train_reshape as t_r
import lernia.train_convNet as t_c
import albio.series_stat as s_s

baseDir = "/home/sabeiro/lav/tmp/gan/"
fL = os.listdir(baseDir + "/face/")
XL = []
for f in fL:
    img = mpimg.imread(baseDir + "/face/" + f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(img.shape) != 3:
        print('layer missing')
        continue
    XL.append(img)

X = np.array(XL)
import importlib
importlib.reload(t_r)
importlib.reload(t_c)
tK = t_c.weekImg(X,model_type="convNet",isBackfold=False)


tK.plotImg(nline=3)
tK.runAutoencoder()
tK.plotMorph(nline=3,n=1)
autoe = tK.model

f = "sab_Profilo.jpg"
img = mpimg.imread(baseDir + "/face/" + f)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
selL = tK.reshape(np.array([gray]) )
dec = autoe.predict(selL).reshape(dec.shape[1],dec.shape[2])

plt.imshow(dec[0])
plt.show()

tK.plotTimeSeries(nline=8)
tK.simpleEncoder(epoch=25)
tK.deepEncoder(epoch=25,isEven=True)
tK.convNet(epoch=50,isEven=True)
encoder, decoder = tK.getEncoder(), tK.getDecoder()



wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
