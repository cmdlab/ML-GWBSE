#!/usr/bin/env python
from funcs import *
from ml_regression import ML_reg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay
import pickle
import scipy.stats as st

def ML_cla(X, y, cutoff, model, gl):
    z=[]
    for val in y:
        if gl=='l':
            if val<=cutoff:
                z.append(1)
            else:
                z.append(0)
        if gl=='g':
            if val>=cutoff:
                z.append(1)
            else:
                z.append(0)
    print('zeros and ones in dataset',z.count(0),z.count(1))
    if model=='RFC':
        sgd_clf = RandomForestClassifier(random_state=42)
    if model=='MLP':
        sgd_clf = MLPClassifier(random_state=1, max_iter=300)
    if model=='SGD':
        sgd_clf = SGDClassifier(random_state=42)
    if model=='ADA':
        sgd_clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    sgd_clf.fit(X, z)
    score=cross_val_score(sgd_clf, X, z, cv=5, scoring="accuracy")
    y_train_pred = cross_val_predict(sgd_clf, X, z, cv=5)
    cm=confusion_matrix(z, y_train_pred)
    print('%5.2f' %(np.mean(score)),cm)
    #plt.cla()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=sgd_clf.classes_)
    disp.plot()
    plt.savefig('./figures/cm.png', dpi=200,bbox_inches='tight', pad_inches=0.1)

    return X,y,z,sgd_clf


def mp_screen(X, sgd_clf, pred_prop):
    keep_cols=X.columns
    keep_cols=list(keep_cols)+['material_id']
    X_mp=prepare_data("./files/mpdata_wf.pkl",15.0, None, None,15)
    X_k=X_mp[keep_cols].copy()
    print(X_k.describe())
    X_k.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_k.dropna(how="any", inplace=True)
    mids = X_k.pop('material_id')
    y_pred=sgd_clf.predict(X_k)
    X_k['predicted_'+pred_prop]=y_pred
    X_k = X_k.join(mids)
    yes=list(y_pred).count(1)
    no=list(y_pred).count(0)
    yes_perc=100*yes/(yes+no)
    print('Percentage of filtered materials','%5.2f' %yes_perc,'%')
    X_k.to_pickle("./files/predicted_"+pred_prop+'.pkl')

