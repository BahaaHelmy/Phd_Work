# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:46:20 2016

@author: Hossam Faris
"""

import numpy
import math
import numpy as np
import ctypes
import opfunu
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm


import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

np.random.get_state()
#2205*625
dim=2
SearchAgents_no=50

lb=[0.00001,.00001]
ub=[.7,.7]
Max_iter=100
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
dir = 'drive/MyDrive/train/train/'
classes = ['NORMAL','PNEUMONIA']
Data = []
Lables = []
for category in os.listdir(dir):
    #Newdir = 'C:\\Users\\Bahaa\\Desktop\\DataSets\\xray_dataset_covid19\\train\\NORMAL\\'
    #C:\\Users\\DOC\\Desktop\\DataSets\\xray_dataset_covid19\\train\\NORMAL
    newPath = os.path.join(dir,category)
    print(newPath)
    if 'Thumbs.db' not in newPath:
        for img in os.listdir(newPath):
            img_path = os.path.join(newPath,img)
            print(newPath)
            if 'Thumbs.db' not in img_path and img != 'IM-0640-0001-0002.jpeg' and img !='NORMAL2-IM-0553-0001.jpeg':
                print(img_path)
                x = ((ocv.resize(ocv.imread(img_path,0),(100,100))/255))
                x = (x.reshape(10000))
                Data.append(x)
                Lables.append(classes.index(category))
combined = list(zip(Data,Lables))
shuffle(combined)
Data[:],Lables[:] = zip(*combined)
DATA = np.array(Data)
print(np.shape(DATA))

LABELS = np.array(Lables)
LABELS = to_categorical(LABELS)
#Data Augmentation

X_train, X_test, y_train, y_test = train_test_split(DATA, LABELS, random_state=30, train_size = .7)
print(np.shape(X_train))

def F1(x):
    f32005 = opfunu.cec_based.F12020(ndim=10)
    return f32005.evaluate(x)

def F2(x):
    f32005 = opfunu.cec_based.F22020(ndim=10)
    return f32005.evaluate(x)
def F3(x):
    f32005 = opfunu.cec_based.F32020(ndim=10)
    return f32005.evaluate(x)
def F4(x):
    f32005 = opfunu.cec_based.F42020(ndim=10)
    return f32005.evaluate(x)

def F5(x):
    f32005 = opfunu.cec_based.F52020(ndim=10)
    return f32005.evaluate(x)
def F6(x):
    f32005 = opfunu.cec_based.F62020(ndim=10)
    return f32005.evaluate(x)
def F7(x):
    f32005 = opfunu.cec_based.F72020(ndim=10)
    return f32005.evaluate(x)
def F8(x):
    f32005 = opfunu.cec_based.F82020(ndim=10)
    return f32005.evaluate(x)

def F9(x):
    f32005 = opfunu.cec_based.F92020(ndim=10)
    return f32005.evaluate(x)
def F10(x):
    f32005 = opfunu.cec_based.F102020(ndim=10)
    return f32005.evaluate(x)

def F11(t):
  import math

  from xgboost import XGBClassifier
  boosters=['gbtree','gblinear', 'dart']

  # args = {'learning_rate': t[2], 'booster': boosters[np.int(np.round(t[0]))],'gamma':t[1]}
  args = {'learning_rate': t[1] ,'gamma':t[0]}

  model = XGBClassifier(**args)
  model.fit(X_train, y_train)
  y=model.predict(X_test)
  accuracy = accuracy_score(y_test, y)
  print ((accuracy * 100.0))
  return ((accuracy * 100.0))
  #correct = 0
  #total = y_train.shape[0]
  #for i in range(total):
  #    predicted = np.argmax(model.predict([X_train[i]]))
  #    test = np.argmax(y_train[i])
  #    correct = correct + (1 if predicted == test else 0)
  #return (correct/total)*100



  #boosters=['gbtree','gblinear', 'dart']
  #model = KNeighborsClassifier(n_neighbors=np.int(np.round(np.abs(t))))
  #model.fit(X_train, y_train)
  #correct = 0
  #total = y_train.shape[0]
  #for i in range(total):
    #   predicted = np.argmax(model.predict([X_train[i]]))
    #  test = np.argmax(y_train[i])
      # correct = correct + (1 if predicted == test else 0)
  #return (correct/total)*100


def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "F1": ["F1", -100, 100, 10],
        "F2": ["F2", -10, 10, 10],
        "F3": ["F3", -100, 100, 10],
        "F4": ["F4", -100, 100, 10],
        "F5": ["F5", -30, 30, 10],
        "F6": ["F6", -100, 100, 10],
        "F7": ["F7", -1.28, 1.28, 10],
        "F8": ["F8", -500, 500, 10],
        "F9": ["F9", -5.12, 5.12, 10],
        "F10": ["F10", -32, 32, 10],
        "F11": ["F11", -600, 600, 30],
        "F12": ["F12", -50, 50, 30],
        "F13": ["F13", -50, 50, 30],
        "F14": ["F14", -65.536, 65.536, 2],
        "F15": ["F15", -5, 5, 4],
        "F16": ["F16", -5, 5, 2],
        "F17": ["F17", -5, 15, 2],
        "F18": ["F18", -2, 2, 2],
        "F19": ["F19", 0, 1, 3],
        "F20": ["F20", 0, 1, 6],
        "F21": ["F21", 0, 10, 4],
        "F22": ["F22", 0, 10, 4],
        "F23": ["F23", 0, 10, 4],
        "Deep": ["Deep", 0, 10, 4],
        "F11": ["F11", 0, 10, 3],
    }
    return param.get(a, "nothing")

'''
# ESAs space mission design benchmarks https://www.esa.int/gsp/ACT/projects/gtop/
from fcmaes.astro import (
    MessFull,
    Messenger,
    Gtoc1,
    Cassini1,
    Cassini2,
    Rosetta,
    Tandem,
    Sagas,
)
def Ca1(x):
    return Cassini1().fun(x)
def Ca2(x):
    return Cassini2().fun(x)
def Ros(x):
    return Rosetta().fun(x)
def Tan(x):
    return Tandem(5).fun(x)
def Sag(x):
    return Sagas().fun(x)
def Mef(x):
    return MessFull().fun(x)
def Mes(x):
    return Messenger().fun(x)
def Gt1(x):
    return Gtoc1().fun(x)

def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "F1": ["F1", -100, 100, 30],
        "F2": ["F2", -10, 10, 30],
        "F3": ["F3", -100, 100, 30],
        "F4": ["F4", -100, 100, 30],
        "F5": ["F5", -30, 30, 30],
        "F6": ["F6", -100, 100, 30],
        "F7": ["F7", -1.28, 1.28, 30],
        "F8": ["F8", -500, 500, 30],
        "F9": ["F9", -5.12, 5.12, 30],
        "F10": ["F10", -32, 32, 30],
        "F11": ["F11", -600, 600, 30],
        "F12": ["F12", -50, 50, 30],
        "F13": ["F13", -50, 50, 30],
        "F14": ["F14", -65.536, 65.536, 2],
        "F15": ["F15", -5, 5, 4],
        "F16": ["F16", -5, 5, 2],
        "F17": ["F17", -5, 15, 2],
        "F18": ["F18", -2, 2, 2],
        "F19": ["F19", 0, 1, 3],
        "F20": ["F20", 0, 1, 6],
        "F21": ["F21", 0, 10, 4],
        "F22": ["F22", 0, 10, 4],
        "F23": ["F23", 0, 10, 4],
        "Ca1": [
            "Ca1",
            Cassini1().bounds.lb,
            Cassini1().bounds.ub,
            len(Cassini1().bounds.lb),
        ],
        "Ca2": [
            "Ca2",
            Cassini2().bounds.lb,
            Cassini2().bounds.ub,
            len(Cassini2().bounds.lb),
        ],
        "Gt1": ["Gt1", Gtoc1().bounds.lb, Gtoc1().bounds.ub, len(Gtoc1().bounds.lb)],
        "Mes": [
            "Mes",
            Messenger().bounds.lb,
            Messenger().bounds.ub,
            len(Messenger().bounds.lb),
        ],
        "Mef": [
            "Mef",
            MessFull().bounds.lb,
            MessFull().bounds.ub,
            len(MessFull().bounds.lb),
        ],
        "Sag": ["Sag", Sagas().bounds.lb, Sagas().bounds.ub, len(Sagas().bounds.lb)],
        "Tan": [
            "Tan",
            Tandem(5).bounds.lb,
            Tandem(5).bounds.ub,
            len(Tandem(5).bounds.lb),
        ],
        "Ros": [
            "Ros",
            Rosetta().bounds.lb,
            Rosetta().bounds.ub,
            len(Rosetta().bounds.lb),
        ],
    }
    return param.get(a, "nothing")
'''