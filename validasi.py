
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# In[2]:

#initialize variabel
theta = np.array([0.4,0.5,0.6,0.7])
bias = 0.25
alpha = 0.1
errvalue = 0.00000
epoch = 60


# In[3]:

#baca dataset
dataX = pd.read_csv('iris.data.txt', sep=",", header = None, nrows=100)
dataX.columns = ("x1","x2","x3","x4","kelas")
dataX["kelas"]=dataX["kelas"].apply(lambda x:str(x).replace('Iris-setosa','0'))
dataX["kelas"]=dataX["kelas"].apply(lambda x:str(x).replace('Iris-versicolor','1'))
dataX["kelas"]=dataX["kelas"].astype('int64')
dataT0 = dataX.iloc[0:40]
dataV0 = dataX.iloc[40:50]
dataT1 = dataX.iloc[50:90]
dataV1 = dataX.iloc[90:100]
training = [dataT0,dataT1]
validate = [dataV0,dataV1]
dataT = pd.concat(training, ignore_index = True)
dataV = pd.concat(validate, ignore_index = True)


# In[4]:

#initialize fungsi
def fungsiH(x, theta, bias):
    sum = 0
    for i in range(0, len(x)):
        sum+= x[i]*theta[i]
    return sum+bias
        
def sigmoid(f):
    sigm = 1/(1+math.exp(-f))
    return sigm
    
def delta(sigm, kelas, x):
    delta = 2*(sigm-kelas)*(1-sigm)*sigm*x
    return delta

def predict(sigm):
    if sigm >= 0.5:
        return 1
    else:
        return 0

def error(kelas, sigm):
    return math.pow((sigm-kelas),2)



# In[5]:

#train
totalerrorT = 0.00000
totalerrorV = 0.00000
simpenTotalerrT = np.zeros(60)
simpenTotalerrV = np.zeros(60)
simpenEpoch = np.zeros(60)
for ep in range(0,epoch):
    print ("\nTRAIN\n==============================")
    totalerrorT=0
    for j in range(0,80):
        x = np.array(dataT.iloc[j,0:4])
        kelas = dataT.iloc[j,4]
        fH = fungsiH(x,theta,bias)
        sigm = sigmoid(fH)
        pred = predict(sigm)
        deltaB = delta(sigm,kelas,1)
        deltaT = delta(sigm, kelas, x)
        errvalue = error (kelas,sigm)
        print ("epoch\t: ",ep+1)
        print ("iterasi\t: ",j+1)
        print("teta\t: ", theta)
        print("x\t: ", x)
        print ("bias\t: ",bias)
        print("kelas\t: ", kelas)
        print("h\t: ", fH)
        print("sigmoid\t: ", sigm)
        print("predict\t: ", pred)
        print("deltaB\t: ", deltaB)
        print("deltaT\t: ", deltaT)
        print("error\t: ", errvalue)
        print("========================")
        totalerrorT+= errvalue
        if j<79:
            for k in range(0,4):
                theta[k] = theta[k]-(deltaT[k]*alpha)
            bias = bias-(deltaB*alpha)
    print ("total error value :",totalerrorT)
    simpenTotalerrT[ep]=totalerrorT                 #untuk menyimpan value total error (training) dalam array agar bisa diplot

    #validate
    print ("\n\nVALIDASI\n==============================")
    totalerrorV=0
    for k in range(0,20):
        x = np.array(dataV.iloc[k,0:4])
        kelas = dataV.iloc[k,4]
        fH = fungsiH(x,theta,bias)
        sigm = sigmoid(fH)
        pred = predict(sigm)
        errvalue = error (kelas,sigm)
        print ("epoch\t: ",ep+1)
        print ("iterasi\t: ",k+1)
        print("teta\t: ", theta)
        print("x\t: ", x)
        print ("bias\t: ",bias)
        print("kelas\t: ", kelas)
        print("h\t: ", fH)
        print("sigmoid\t: ", sigm)
        print("predict\t: ", pred)
        print("error\t: ", errvalue)
        print("========================")
        totalerrorV+= errvalue
    print ("total error value :",totalerrorV)
    simpenTotalerrV[ep]=totalerrorV                 #untuk menyimpan value total error (validasi) dalam array agar bisa diplot
    
    simpenEpoch[ep]=ep                              #untuk menyimpan value epoch ke-berapa ke dalam array agar bisa diplot


# In[7]:

xAxis = simpenEpoch
yTrain = simpenTotalerrT
yVali = simpenTotalerrV

plt.plot(xAxis,yTrain,'b' , xAxis,yVali,'r--')
plt.xlabel("Epoch")
plt.ylabel("Error Value")
plt.legend(['Train','Validasi'])
plt.show()


# In[ ]:



