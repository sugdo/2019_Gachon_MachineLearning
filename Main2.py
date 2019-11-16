import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler



import seaborn as sns
import matplotlib.pyplot as plt


def getAccurate ( kmeans , X , y) :
    correct = 0
    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = kmeans.predict(predict_me)
        if prediction[0] == y[i]:
            correct += 1
    val = correct / len(X)
    if ( val < 0.5 ) :
        val = 1 - val
    print(val)


def gc (X) :                  # increase Importance
    for i in range( len(X) ) :
        X[i][6] *= 0.1                # Pclass
        X[i][2] *= 1                  # Sex
    return X




def main () :
    print("START")
    train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
    train = pd.read_csv(train_url)
    test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
    test = pd.read_csv(test_url)
    print(train)

    print("***** Train_Set *****")
    print(train.head())
    print("\n")
    print("***** Test_Set *****")
    print(test.head())

    print("***** Train_Set *****")
    print(train.describe())
    print("\n")
    print("***** Test_Set *****")
    print(test.describe())

    print(train.columns.values)

    print("*****In the train set*****")
    print(train.isna().sum())
    print("\n")
    print("*****In the test set*****")
    print(test.isna().sum())

    train.fillna(train.mean(), inplace=True)
    test.fillna(test.mean(), inplace=True)

    print(train.isna().sum())

    print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    print(train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

    print(train.info())


    train = train.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    test = test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    labelEncoder = LabelEncoder()
    labelEncoder.fit(train['Sex'])
    labelEncoder.fit(test['Sex'])
    train['Sex'] = labelEncoder.transform(train['Sex'])
    test['Sex'] = labelEncoder.transform(test['Sex'])

    print(train.info())
    print(test.info())

    X = np.array(train.drop(['Survived'], 1).astype(float))
    y = np.array(train['Survived'])


    kmeans = KMeans(n_clusters=2)
    print(kmeans.fit(X))

    correct = 0
    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = kmeans.predict(predict_me)
        if prediction[0] == y[i]:
            correct += 1
    print(correct / len(X))

    kmeans = KMeans(n_clusters=2, max_iter=600, algorithm='auto')
    kmeans.fit(X)

    getAccurate ( kmeans , X , y)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans.fit(X_scaled)
    print(X_scaled)

    getAccurate(kmeans, X, y)
    print("***** HomeWork : Change the values of these parameters and submit them with higher accuracy. *****")
    kmeans = KMeans(n_clusters=2, init='random', n_init=100, max_iter=1000, algorithm='full')

    scaler = MinMaxScaler()


    X_scaled = scaler.fit_transform(X)

    X_scaled = gc(X_scaled)

    kmeans.fit(X_scaled)
    getAccurate(kmeans, X_scaled, y)



main()