#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: Importez vos modules ici
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
# TODO: Définissez vos fonctions ici
def readfile():
    path=os.getcwd()
    return pd.read_csv(os.path.join(path, 'data/winequality-white.csv'), sep=';')


def seprate_target(df, name):
    return df.drop(columns=[name]), df.filter(items=[name])


def split_df (df_filtre, quality):
    return train_test_split(df_filtre, quality, test_size=0.5, train_size=0.5)

def methode_random(X_train, y_train, X_test):
    regr = RandomForestRegressor(n_estimators=2449, random_state=0)
    regr.fit(X_train, y_train)
    return regr.predict(X_test)

def methode_linear(X_train, y_train, X_test):
    regr = LinearRegression().fit(X_train, y_train)
    return regr.predict(X_test)


def graphe_random (y_random, y_test):
    plt.plot(y_random, y_test, label='predicted values')
    plt.plot(y_test)
    plt.show()
    



if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    df=readfile()
    df_filtre, quality=seprate_target(df, "quality")
    X_train, X_test, y_train, y_test=split_df(df_filtre,quality)
    y_random=methode_random(X_train, y_train, X_test)
    y_linear=methode_linear(X_train, y_train, X_test)
    graphe_random(y_random,y_test)