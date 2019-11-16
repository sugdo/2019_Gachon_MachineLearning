import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt

def main():
    data = np.loadtxt("mouse.csv", delimiter="," )
    print("Case)DBSCAN")
    myDBSCAN(data)
    print("Case)k-Means")
    mykMeans(data)


def mykMeans(data) :
    x= []
    y= []

    n_clusters = [ 2, 3, 4, 5, 6  ]
    max_iter =[50, 100, 200, 300]

    for i in range (len(data)) :
        x.append(data[i][0])
        y.append(data[i][1])

    count=0;


    for i in range(len(n_clusters)) :
        for j in range (len(max_iter)) :

            result_col = []
            clustering=KMeans(n_clusters=n_clusters[i], max_iter=max_iter[j]).fit_predict(data)
            num = set(clustering)

            for k in range(len(clustering)) :
                if(clustering[k]==-1) :
                    result_col.append('black')
                elif(clustering[k]==0) :
                    result_col.append('gray')
                elif(clustering[k]==1) :
                    result_col.append('brown')
                elif (clustering[k] == 2):
                    result_col.append('coral')
                elif (clustering[k] == 3):
                    result_col.append('chocolate')
                elif (clustering[k] == 4):
                    result_col.append('blue')
                elif (clustering[k] == 5):
                    result_col.append('purple')
                elif (clustering[k] == 6):
                    result_col.append('cyan')
                elif (clustering[k] == 7):
                    result_col.append('crimson')
                elif (clustering[k] == 8):
                    result_col.append('lime')
                elif (clustering[k] == 9):
                    result_col.append('pink')
                else :
                    result_col.append('green')

            plt.title('k-Means')
            plt.scatter(x,y,c=result_col)
            if( count <100 and len(num) >4 ) :
                plt.show()
                count = count + 1




def myDBSCAN (data) :

    x= []
    y= []
    eps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 ]
    min_samples =[3, 5, 10, 15, 20, 30, 50, 100 ]

    for i in range (len(data)) :
        x.append(data[i][0])
        y.append(data[i][1])

    count=0;


    for i in range(len(eps)) :
        for j in range (len(min_samples)) :

            result_col = []
            clustering=DBSCAN(eps=eps[i], min_samples=min_samples[j]).fit_predict(data)
            num = set(clustering)

            for k in range(len(clustering)) :
                if(clustering[k]==-1) :
                    result_col.append('black')
                elif(clustering[k]==0) :
                    result_col.append('gray')
                elif(clustering[k]==1) :
                    result_col.append('brown')
                elif (clustering[k] == 2):
                    result_col.append('coral')
                elif (clustering[k] == 3):
                    result_col.append('chocolate')
                elif (clustering[k] == 4):
                    result_col.append('blue')
                elif (clustering[k] == 5):
                    result_col.append('purple')
                elif (clustering[k] == 6):
                    result_col.append('cyan')
                elif (clustering[k] == 7):
                    result_col.append('crimson')
                elif (clustering[k] == 8):
                    result_col.append('lime')
                elif (clustering[k] == 9):
                    result_col.append('pink')
                else :
                    result_col.append('green')

            plt.title('DBSCAN')
            plt.scatter(x,y,c=result_col)
            if( count <100 and len(num) >1 ) :
                plt.show()
                count = count + 1

main()