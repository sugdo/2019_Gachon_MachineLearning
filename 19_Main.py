from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

score_logistic = []
score_svm =[]


"""
Function : Logistic_score 
Parameter : data , data_target , test data , test target
objective : using Logistic Regression for solving problem (multinomial)
return : information for best classifier
"""

@ignore_warnings(category=ConvergenceWarning)
def Logistic_score (X,y , test_x , test_y) :
    max = 0;
    text=""
    for i in ('lbfgs','sag') :
        for j in (50,100,200) :
            clf = LogisticRegression(solver=i , max_iter=j,multi_class='multinomial')
            clf.fit(X,y)
            clf.predict(test_x)
            scores = clf.score(test_x,test_y)
            print("LogisticRegression with "+ i +"," +str(j)+"(multinomial)" )
            print(scores)
            print("mean:{}".format(np.mean(scores)))

            temp = np.mean(scores)

            if (max < temp):
                max = temp
                text = "parameter values for classifier:" + i +","+str(j)+ " maximum score: " + str(np.max(scores))
            score_logistic.append(np.mean(scores))

    return text


"""
Function : SVM_score 
Parameter : data , data_target , test data , test target
objective : using SVM_score for solving problem (multinomial)
return : information for best classifier
"""

def SVM_score (X,y,test_x,test_y) :
    max =0;
    text=""
    for i in ( 0.1, 1.0, 10.0) :
        for j in ('linear','rbf') :
            for n in ('auto',10,100) :
                clf= SVC(C=i , kernel=j , gamma=n)
                clf.fit(X,y)
                clf.predict(test_x)
                scores = clf.score(test_x, test_y)

                # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
                # The multiclass support is handled according to a one-vs-one scheme

                print("SVM with "+ str(i) +"," +j+","+str(n)+"(multinomial)")
                print(scores)
                print("mean:{}".format(np.mean(scores)))

                temp = np.mean(scores)

                if (max < temp):
                    max = temp
                    text = "parameter values for classifier:" + str(i)+","+j+","+str(n) + " maximum score: " + str(np.max(scores))
                score_svm.append(np.mean(scores))
    return text

"""
Function : ensemble
Parameter : data , data_target , test data , test target
objective : using ensemble for solving problem (multinomial)
"""

@ignore_warnings(category=ConvergenceWarning)
def ensemble (X,y,ensemble_x,ensemble_y) :
    clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state = 1,max_iter=50)
    clf2 = SVC(C=0.1 , kernel='linear' , gamma=10,probability=True)
    clf3 = SVC(C=0.1, kernel='rbf', gamma=10 ,probability=True )

    eclf = VotingClassifier(estimators=[('Logistic', clf1), ('SVM_linear', clf2), ('SVM_rbf', clf3)], voting='soft',
                            weights=[1,1,1])

    eclf = eclf.fit(X, y)
    clf1 = clf1.fit(X,y)
    clf2 = clf2.fit(X,y)
    clf3 = clf3.fit(X,y)

    y_pred = eclf.predict(ensemble_x)
    scores = eclf.score(ensemble_x, ensemble_y)

    print(scores)

    conf_mats = pd.crosstab(ensemble_y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    sn.heatmap(conf_mats, annot=True)
    plt.title("Ensemble's Confusion Matrix")
    plt.show()

    y_pred = clf1.predict(ensemble_x)
    conf_mats = pd.crosstab(ensemble_y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    sn.heatmap(conf_mats, annot=True)
    plt.title("Logistic Regression's Confusion Matrix")
    plt.show()

    y_pred = clf2.predict(ensemble_x)
    conf_mats = pd.crosstab(ensemble_y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    sn.heatmap(conf_mats, annot=True)
    plt.title("SVM_linear kernel's Confusion Matrix")
    plt.show()

    y_pred = clf3.predict(ensemble_x)
    conf_mats = pd.crosstab(ensemble_y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    sn.heatmap(conf_mats, annot=True)
    plt.title("SVM_rbf kernel's Confusion Matrix")
    plt.show()


def main() :
    mnist= loadmat('mnist-original.mat')
    print(mnist)
    print(type(mnist))

    mnist_data = mnist["data"].T
    mnist_label = mnist["label"][0]

    print(mnist_data.shape)
    print(mnist_label.shape)

    train_x, mnist_data, train_y, mnist_label = train_test_split(mnist_data, mnist_label, test_size=0.1, random_state=0)
    train_x, test_x ,  train_y, test_y = train_test_split(mnist_data, mnist_label, test_size= 0.33, random_state=0)
    train_x, ensemble_x, train_y , ensemble_y = train_test_split(train_x, train_y, test_size=0.5, random_state=0)

    print(len(test_x))
    print(len(ensemble_x))
    print(len(train_x))

    print(train_x.shape)
    print(train_y.shape)

    #text1 = Logistic_score (train_x, train_y , test_x , test_y)
    #print(text1)
    #text2 = SVM_score(train_x, train_y, test_x, test_y)
    #print(text2)

    ensemble(train_x,train_y,ensemble_x,ensemble_y)



main ()