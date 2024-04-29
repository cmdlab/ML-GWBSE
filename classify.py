#!/usr/bin/env python
from ml_regression import ML_reg
from ml_classification import ML_cla, mp_screen

nprop=8
algo='RFC'

ebe_val={'pred_prop': 'ebe', 'clf_cut': 0.20, 'gl': 'l', "reg_cut": 6.1} 
iac_val={'pred_prop': 'iac', 'clf_cut': 10.5, 'gl': 'g', "reg_cut": 100.} 
aac_val={'pred_prop': 'aac', 'clf_cut': 0.80, 'gl': 'g', "reg_cut": 100.} 

vals=[ebe_val, iac_val, aac_val]
#vals=[ebe_val]

for val in vals:

    pred_prop=val["pred_prop"]
    clf_cut=val["clf_cut"]
    gl=val["gl"]
    reg_cut=val["reg_cut"]

    X,y,rf=ML_reg(reg_cut, nprop, pred_prop, 'RFC')
    X,y,z,sgd_clf=ML_cla(X, y, clf_cut, algo, gl)
    mp_screen(X, sgd_clf, pred_prop)

