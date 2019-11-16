import pandas as pd
import random



def kModes(data,k) :
    print("-----Running Algorithm-----")
    index = list (data.columns)
    Matrix = [[0 for col in range(len(data.columns))] for row in range(k)]
    cluster_array = [0 for i in range(len(data))]

    # first select k objects as center
    array = k_random(data,k)

    # get cluster mode value
    Matrix = k_random_Matrix(index,array,Matrix,data,k)

    #assign the object to the cluster whose center has the shortest distance to the object
    cluster_array = get_cluster_array(cluster_array,k,index,Matrix,data )


    while(True) :
        count=0
        for i in range(k) :
            for j in range(len(index)) :
                lists=[]
                for o in range(len(cluster_array)) :
                    if(i==cluster_array[o]) :
                        lists.append(data[index[j]][o])
                mode = max(set(lists), key=lists.count)
                if (mode != Matrix[i][j]) :
                    Matrix[i][j] = mode
                    count = count+1
        if(count ==0 ) :
            break
        else :
            for i in range(len(cluster_array)):
                result_num = 100
                result_idx = -1
                for j in range(k):
                    num = 0
                    for o in range(len(index)):
                        if (Matrix[j][o] != data[index[o]][i]):
                            num = num + 1
                    if (num < result_num):
                        result_num = num
                        result_idx = j

                cluster_array[i] = result_idx

    print("-----Result-----")
    print("Cluster's center value")
    print(Matrix)
    print("Clustering")
    print(cluster_array)
    return cluster_array

def k_random(data,k) :
    array = [0 for i in range(k)]
    for i in range(k) :
        while(True) :
            n=random.randint(0,len(data)-1)
            if(n not in array) :
                array[i] = n
                break
    return array



def getPurity (data,cluster_array,target) :
    k = set(cluster_array)
    right =0
    for j in range(len(k)):
        lists=[]
        for i in range(len(cluster_array)) :
            if( j == cluster_array[i]) :
                lists.append(data[target][i])

        ok = max(set(lists), key=lists.count)
        ok = lists.count(ok)
        right = right + ok
    purity = right / len(cluster_array)
    print("purity is {}".format(purity))






def k_random_Matrix(index,array,Matrix,data,k) :
    for i in range(k) :
        for j in range(len(index)) :
            Matrix[i][j] = data[index[j]][array[i]]
    return Matrix


def get_cluster_array(cluster_array,k,index,Matrix,data ) :
    for i in range(len(cluster_array)) :
        result_num=100
        result_idx=-1
        for j in range(k) :
            num=0
            for o in range(len(index)) :
                if(Matrix[j][o] != data[index[o]][i]) :
                    num = num+1
            if(num < result_num) :
                result_num = num
                result_idx = j

        cluster_array[i] = result_idx
    return cluster_array



def main() :

    data = pd.read_csv('mushrooms.csv')
    print("case k=3 )")
    clustering_array = kModes(data,3)  # k is 3
    getPurity(data,clustering_array,"class")
    print("case k=4 )")
    clustering_array = kModes(data,4)  # k is 4
    getPurity(data,clustering_array,"class")



main()