
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
totalerror = 0.00000
simpenTotalerr = np.zeros(60)
simpenEpoch = np.zeros(60)
for ep in range(0,epoch):
    totalerror=0
    print ("==============================\nepoch : ",ep)
    for j in range(0,100):
        x = np.array(dataX.iloc[j,0:4])
        kelas = dataX.iloc[j,4]
        fH = fungsiH(x,theta,bias)
        sigm = sigmoid(fH)
        pred = predict(sigm)
        deltaB = delta(sigm,kelas,1)
        deltaT = delta(sigm, kelas, x)
        errvalue = error (kelas,sigm)
        print ("iterasi: ",j+1)
        print("teta: ", theta)
        print("x: ", x)
        print ("bias: ",bias)
        print("kelas: ", kelas)
        print("h: ", fH)
        print("sigmoid: ", sigm)
        print("predict: ", pred)
        print("deltaB: ", deltaB)
        print("deltaT: ", deltaT)
        print("error: ", errvalue)
        print("========================")
        totalerror+= errvalue
        for k in range(0,4):
            theta[k] = theta[k]-(deltaT[k]*alpha)
        bias = bias-(deltaB*alpha)
    print ("total error value :",totalerror)
    simpenTotalerr[ep]=totalerror                 #untuk menyimpan value total error dalam array agar bisa diplot
    simpenEpoch[ep]=ep                            #untuk menyimpan value epoch ke-berapa ke dalam array agar bisa diplot


# In[6]:

xAxis = simpenEpoch
yAxis = simpenTotalerr
plt.plot(xAxis,yAxis)
plt.show()


# In[ ]:



