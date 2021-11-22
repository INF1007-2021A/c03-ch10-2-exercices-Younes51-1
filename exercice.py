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
from sklearn.metrics import mean_squared_error


# TODO: Définissez vos fonctions ici
def readfile():
    return pd.read_csv(os.path.join(os.getcwd(), 'data/winequality-white.csv'), sep=';')


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
    x=np.arange(0,2449,1)
    plt.plot(x,y_test, c='cornflowerblue')    
    plt.plot(x,y_random, c='orange')
    plt.title('RandomForestRegressor predictions analysis')
    plt.ylabel('Quality')
    plt.xlabel('Number of samples')
    plt.legend(['Target values', 'Predicted values'], loc="upper right")
    plt.show()
    
    
def graphe_linear (y_linear, y_test):
    x=np.arange(0,2449,1)
    plt.plot(x,y_test, c='cornflowerblue')    
    plt.plot(x,y_linear, c='orange')
    plt.title('LinearRegression predictions analysis')
    plt.ylabel('Quality')
    plt.xlabel('Number of samples')
    plt.legend(['Target values', 'Predicted values'], loc="upper right")
    plt.show()
    

def erreur_random(y_random, y_test):
    return mean_squared_error(y_test, y_random)


def erreur_linear(y_linear, y_test):
    return mean_squared_error(y_test, y_linear)


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    df=readfile()
    df_filtre, quality=seprate_target(df, "quality")
    X_train, X_test, y_train, y_test=split_df(df_filtre,quality)
    y_random=methode_random(X_train, y_train, X_test)
    y_linear=methode_linear(X_train, y_train, X_test)
    graphe_linear(y_linear, y_train)
    graphe_random(y_random, y_test)
    print(f"l'erreur random est {erreur_random(y_random, y_test)}")
    print(f"l'erreur linear est {erreur_linear(y_linear, y_test)}")
    