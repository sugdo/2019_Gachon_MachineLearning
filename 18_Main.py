import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import  MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


score_decision = []
score_logistic = []
score_svm = []

"""
Function : Decision_score 
Parameter : data , data_target
objective : using decision tree for solving problem 
return : information for best classifier
"""

def Decision_score (X,y) :
    max = 0;
    text="";
    for i in ('gini','entropy') :
        clf = DecisionTreeClassifier(criterion=i)
        scores = cross_val_score(clf,X,y,cv=10)
        print("Decision Tree with "+ i)
        print(scores)
        print("mean:{}".format(np.mean(scores)))

        y_pred = cross_val_predict(clf, X, y, cv=10)
        conf_mat = confusion_matrix(y, y_pred)
        conf_mats = pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins = True)


        tn, fp, fn, tp = conf_mat.ravel()
        temp = tn + tp
        if( max < temp) :
            max = temp
            result = conf_mats
            text = "parameter values for classifier:"+i + " maximum score: " +str(np.max(scores))
        score_decision.append(np.mean(scores))

    sn.heatmap(result, annot=True)

    plt.title("Decision Tree's Best Confusion Matrix")
    plt.show()
    return text


"""
Function : Logistic_score 
Parameter : data , data_target
objective : using Logistic Regression for solving problem 
return : information for best classifier
"""

def Logistic_score (X,y) :
    max = 0;
    text=""
    for i in ('liblinear','lbfgs','sag') :
        for j in (50,100,200) :
            clf = LogisticRegression(solver=i,max_iter=j)
            scores = cross_val_score(clf,X,y,cv=10)
            print("LogisticRegression with "+ i +"," +str(j))
            print(scores)
            print("mean:{}".format(np.mean(scores)))
            y_pred = cross_val_predict(clf, X, y, cv=10)
            conf_mat = confusion_matrix(y, y_pred)
            conf_mats = pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins = True)

            tn, fp, fn, tp = conf_mat.ravel()
            temp = tn + tp
            if (max < temp):
                max = temp
                result = conf_mats
                text = "parameter values for classifier:" + i +","+str(j)+ " maximum score: " + str(np.max(scores))
            score_logistic.append(np.mean(scores))

    sn.heatmap(result, annot=True)
    plt.title("Logistic Regression's Best Confusion Matrix")
    plt.show()
    return text



"""
Function : SVM_score
Parameter : data , data_target
objective : using Support vector machine for solving problem 
return : information for best classifier
"""


def SVM_score (X,y) :
    max =0;
    text=""
    for i in ( 0.1, 1.0, 10.0) :
        for j in ('linear','poly','rbf','sigmoid') :
            for n in ('auto',10,100) :
                clf= SVC (C=i , kernel=j , gamma=n)
                scores = cross_val_score(clf,X,y,cv=10)
                print("SVM with "+ str(i) +"," +j+","+str(n))
                print(scores)
                print("mean:{}".format(np.mean(scores)))
                y_pred = cross_val_predict(clf, X, y, cv=10)
                conf_mat = confusion_matrix(y, y_pred)
                conf_mats = pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
                tn, fp, fn, tp = conf_mat.ravel()
                temp = tn + tp
                if (max < temp):
                    max = temp
                    result = conf_mats
                    text = "parameter values for classifier:" + str(i)+","+j+","+str(n) + " maximum score: " + str(np.max(scores))
                score_svm.append(np.mean(scores))

    sn.heatmap(result, annot=True)
    plt.title("SVM's Best Confusion Matrix")
    plt.show()
    return text



def main() :
    df = pd.read_csv('heart.csv')
    print(df)
    print(df.info())

    scaler = MinMaxScaler()
    df[:] = scaler.fit_transform(df[:])
    print(df)

    X = np.array(df.drop(['target'], 1).astype(float))
    y = np.array(df['target'])

    text1=Decision_score(X,y)
    text2=Logistic_score(X,y)
    text3=SVM_score(X,y)

    print("Find the maximum score and the corresponding parameter values for each classifier")
    print(text1)
    print(text2)
    print(text3)

    print(score_decision)
    print(score_logistic)
    print(score_svm)

    plt.bar(['gini', 'entropy'], score_decision, align='center')
    plt.ylabel('score')
    plt.title('Decision Tree Score')
    plt.show()

    #Logistic Regression



    # setup the figure and axes
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')

    x = [0,1,2 , 0,1,2 , 0,1,2]
    y = [0,0,0,1,1,1,2,2,2]
    top = score_logistic
    bottom = np.zeros_like(top)
    width = depth = 0.5

    plt.xticks([0, 1, 2], ['liblinear', 'lbfgs', 'sag'])
    plt.yticks([0, 1, 2], ['50', '100', '200'])


    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_title('Logistic Regression')

    ax1.set_xlabel('solver')
    ax1.set_ylabel('max_iter')
    ax1.set_zlabel('score')


    plt.show()

    top=[]
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')

    x = [0,1,2 , 0,1,2 , 0,1,2]
    y = [0,0,0,1,1,1,2,2,2]



    for i in range(3):
        for j in range(3):
            top.append(score_svm[(12*i) + j])


    bottom = np.zeros_like(top)
    width = depth = 0.5

    plt.xticks([0, 1, 2], ['0.1', '1.0', '10.0'])
    plt.yticks([0, 1, 2], ['0', '10', '100'])


    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_title('SVM - linear')

    ax1.set_xlabel('C')
    ax1.set_ylabel('gamma')
    ax1.set_zlabel('score')


    plt.show()


    top= []

    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')

    x = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    for i in range(3):
        for j in range(3):
            top.append(score_svm[(12 * i) + j +3])

    bottom = np.zeros_like(top)
    width = depth = 0.5

    plt.xticks([0, 1, 2], ['0.1', '1.0', '10.0'])
    plt.yticks([0, 1, 2], ['0', '10', '100'])

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_title('SVM - poly')

    ax1.set_xlabel('C')
    ax1.set_ylabel('gamma')
    ax1.set_zlabel('score')

    plt.show()


    top = []
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')

    x = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    for i in range(3):
        for j in range(3):
            top.append(score_svm[(12 * i) + j +6])

    bottom = np.zeros_like(top)
    width = depth = 0.5

    plt.xticks([0, 1, 2], ['0.1', '1.0', '10.0'])
    plt.yticks([0, 1, 2], ['0', '10', '100'])

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_title('SVM - rbf')

    ax1.set_xlabel('C')
    ax1.set_ylabel('gamma')
    ax1.set_zlabel('score')

    plt.show()


    top = []
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')

    x = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    for i in range(3):
        for j in range(3):
            top.append( score_svm[(12 * i) + j +9])
    bottom = np.zeros_like(top)
    width = depth = 0.5

    plt.xticks([0, 1, 2], ['0.1', '1.0', '10.0'])
    plt.yticks([0, 1, 2], ['0', '10', '100'])

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_title('SVM - sigmoid')

    ax1.set_xlabel('C')
    ax1.set_ylabel('gamma')
    ax1.set_zlabel('score')

    plt.show()

main()