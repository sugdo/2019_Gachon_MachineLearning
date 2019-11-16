import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def main():
    print("---start---")
    df = pd.read_csv('vgsales.csv')
    print(df.head())
    df=df.dropna()

    y = df['Global_Sales'].values
    X = df.drop(columns=['Global_Sales','Name','Platform','Genre','Rank','Publisher'])

    X = X.astype('int')
    y = y.astype('int')


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(X_train, y_train)

    print("case 1 : holdout")
    print(knn.score(X_test, y_test))

    print("case 2 :  k-fold cross validation")

    cv_scores = cross_val_score(knn, X, y, cv=5)
    print(cv_scores)
    print('cv_scores mean: {}'.format(np.mean(cv_scores)))

main()