from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNet
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

def main () :
    df = pd.read_csv('Happy.csv')
    X = df.drop(['Country','Happiness.Rank','Whisker.high','Whisker.low','Happiness.Score'],axis=1)
    y = df['Happiness.Score'].values.reshape(-1, 1)

    # Ridge Part
    ridge = Ridge()
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
    ridge_regressor.fit(X, y)

    print("Ridge's Best cross val scores :")
    print(ridge_regressor.best_params_)
    print(ridge_regressor.best_score_)

    # Rasso Part
    lasso = Lasso()
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
    lasso_regressor.fit(X, y)

    print("Rasso's cross val scores :")
    print(lasso_regressor.best_params_)
    print(lasso_regressor.best_score_)

    # ElasticNet Part
    regr = ElasticNet(random_state=0)
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    elastic_regressor = GridSearchCV(regr, parameters, scoring='neg_mean_squared_error', cv=5)
    elastic_regressor.fit(X, y)

    print("Elastic's cross val scores :")
    print(elastic_regressor.best_params_)
    print(elastic_regressor.best_score_)

main()
