import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def main():
    homework()


def homework() :
    data = pd.read_csv('mushrooms.csv')
    print(data.info)
    count=0

    print(data.columns[1])

    eps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 ]
    min_samples =[3, 5, 10, 15, 20, 30, 50, 100 ]
    preprocessing = ['none','LabelEncoding','LabelEncoding+MinMaxScaler','LabelEncoding+StandardScaler']
    Distance_measure = ['Euclidean','Hamming']


  #  for i in range ( len(data.columns) ) :
      #  encoder = LabelEncoder()
     #   encoder.fit(data[data.columns[i]].unique())
      #  data[data.columns[i]] = encoder.transform(data[data.columns[i]])

    #x = data.iloc[:,1:]
    #y = data.iloc[:,0]
    #print(x)
    #print(y)
    #scaler = MinMaxScaler()
    #x = scaler.fit_transform(x)

    print("Please Waiting for running algorithm")

    for i in range(len(eps)):
        for j in range(len(min_samples)):
            for l in range(len(Distance_measure)) :
                for k in range(len(preprocessing)):
                    if(k==0) :
                        x = data.iloc[:, 1:]
                        y = data.iloc[:, 0]
                    elif (k==1) :
                        for m in range(len(data.columns)):
                            encoder = LabelEncoder()
                            encoder.fit(data[data.columns[m]].unique())
                            data[data.columns[m]] = encoder.transform(data[data.columns[m]])
                        x = data.iloc[:, 1:]
                        y = data.iloc[:, 0]
                    elif (k==2) :
                        for m in range(len(data.columns)):
                            encoder = LabelEncoder()
                            encoder.fit(data[data.columns[m]].unique())
                            data[data.columns[m]] = encoder.transform(data[data.columns[m]])
                        x = data.iloc[:, 1:]
                        y = data.iloc[:, 0]
                        scaler = MinMaxScaler()
                        x = scaler.fit_transform(x)
                    else :
                        for m in range(len(data.columns)):
                            encoder = LabelEncoder()
                            encoder.fit(data[data.columns[m]].unique())
                            data[data.columns[m]] = encoder.transform(data[data.columns[m]])
                        x = data.iloc[:, 1:]
                        y = data.iloc[:, 0]
                        scaler = StandardScaler()
                        x = scaler.fit_transform(x)
                        

                    if(k==0) :
                        print(" in case of none preprocessing, you cant get a result that you want ")
                    else :
                        clustering = DBSCAN(eps=eps[i], min_samples=min_samples[j], p=l).fit_predict(x)
                        num = set(clustering)

                        if(count<10 and len(num)>0) :
                            count = count +0
                            getPurity(clustering,y)

def getPurity (clustering,classes) :


    Matrix = [[0 for col in range(2)] for row in range(len(clustering))]

    index = set(clustering)
    print(index)
    print(classes)

    for i in range(len(clustering)) :
        if(classes[i]==0) :
            Matrix[clustering[i]][0] += 1
        else :
            Matrix[clustering[i]][1] += 1

    print(Matrix)

    numerator = 0

    for i in range(len(clustering)) :
        x1=Matrix[i][0]
        x2=Matrix[i][1]
        if(x1>x2) :
            numerator+=x1
        else :
            numerator+=x2


    result = numerator/len(clustering)
    print("Purity is :" + str(result))





main()