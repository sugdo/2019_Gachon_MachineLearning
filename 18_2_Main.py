import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import  MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sn
from mpl_toolkits.mplot3d import Axes3D


score_decision = []
score_logistic = []
score_svm = []


def Decision_score (X,y) :

    for i in ('gini','entropy') :
        clf = DecisionTreeClassifier(criterion=i)
        scores = cross_val_score(clf,X,y,cv=10)
        print("Decision Tree with "+ i)
        print(scores)
        print("mean:{}".format(np.mean(scores)))

        y_pred = cross_val_predict(clf, X, y, cv=10)
        conf_mat = confusion_matrix(y, y_pred)
        print(conf_mat)

        score_decision.append(np.mean(scores))


def Logistic_score (X,y) :
    for i in ('liblinear','lbfgs','sag') :
        for j in (50,100,200) :
            clf = LogisticRegression(solver=i,max_iter=j)
            scores = cross_val_score(clf,X,y,cv=10)
            print("LogisticRegression with "+ i +"," +str(j))
            print(scores)
            print("mean:{}".format(np.mean(scores)))
            y_pred = cross_val_predict(clf, X, y, cv=10)
            conf_mat = confusion_matrix(y, y_pred)
            print(conf_mat)

            score_logistic.append(np.mean(scores))

def SVM_score (X,y) :
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
                print(conf_mat)

                score_svm.append(np.mean(scores))


def main() :
    df = pd.read_csv('heart.csv')
    print(df)
    print(df.info())

    scaler = MinMaxScaler()
    df[:] = scaler.fit_transform(df[:])
    print(df)

    X = np.array(df.drop(['target'], 1).astype(float))
    y = np.array(df['target'])

    Decision_score(X,y)
    Logistic_score(X,y)
    SVM_score(X,y)

    print("scores")
    print(score_decision)
    print(score_logistic)
    print(score_svm)







    plt.bar(['gini', 'entropy'], score_decision, align='center')
    plt.ylabel = 'score'
    plt.title = 'Decision Tree Score'
    plt.show()

    # set the figures and axes
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')

    # fake data
    _x = np.arange(4)
    _y = np.arange(5)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = x + y
    bottom = np.zeros_like(top)
    width = depth = 1

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_title('Shaded')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    plt.show()



"""
    clf = DecisionTreeClassifier(criterion='entropy')        # Decision Tree
    kfold = KFold(10, True, 1)
    for train, test in kfold.split(df):
        print(train)

"""

# SVM


   #SVM(X,y)

    #clf = SVC(gamma='auto')


main()