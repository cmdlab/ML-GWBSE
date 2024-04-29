#!/usr/bin/env python
from funcs import *
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay
import pickle
import scipy.stats as st

def ML_reg(xmx, nft, pred_prop, algo):
    excluded = ["formula","structure","composition","composition_oxid",
            "dielectric_constant",'HOMO_element','LUMO_element',"source","optical_gap",
            "hmass","emass","HOMO_character","LUMO_character",'material_id',
            #'range EN difference', 'std_dev EN difference', 
            #'minimum EN difference', 'maximum EN difference','mean EN difference',
            "qp_gap","ebe","aac","iac"]
    excluded.remove(pred_prop)
    #X,y=prepare_data('./files/data_w_feature_vis.pkl', xmx, excluded, pred_prop, 15)
    X,y=prepare_data('./files/data_w_feature_uv.pkl', xmx, excluded, pred_prop, 15)
    print("There are {} possible descriptors:".format(X.shape[1]))
    print("There are {} total materials:".format(X.shape[0]))
    rf,err0=Random_Forest(X,y)
    print("Finding best {} features:".format(nft))
    excluded=imp_feat(rf,X,y,nft)
    X = X.drop(excluded, axis=1)
    print("There are {} possible descriptors:".format(X.shape[1]))

    #ALGO=Random Forest
    if algo=='RFC':
        rf,err0=Random_Forest(X,y)

    #ALGO=Multi-Layer Perceptron
    if algo=='MLP':
        rf,err0=MLP(X,y)

    #ALGO=Support Vector Regression
    if algo=='SVR':
        rf,err0=Sup_VR(X,y)

    #ALGO=Kernel Ridge Regression
    if algo=='KRR':
        rf,err0=KRR(X,y)

    #ALGO=Gaussian Process Regression
    #rf,err0=GPR(X,y)

    #check the fit of the data
    #plot_result(rf,X,y,xmx,0.1,pred_prop)

    #save the trained ML model to a file
    with open('./files/model.pkl','wb') as f:
        pickle.dump(rf,f)
    
    return X,y,rf

def mp_reg(X, rf, pred_prop):
    keep_cols=X.columns
    keep_cols=list(keep_cols)+['material_id','structure','icsd','elements','formula']
    X=prepare_data("./files/mpdata_wf.pkl", None, None, None, 15)
    X_k=X[keep_cols].copy()
    mids = X_k.pop('material_id')
    structs = X_k.pop('structure')
    icsds = X_k.pop('icsd')
    elems = X_k.pop('elements')
    fmls = X_k.pop('formula')
    X_k.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_k.dropna(how="any", inplace=True)
    y_pred=rf.predict(X_k)
    X_k['predicted_'+pred_prop+'_r']=y_pred
    X_k = X_k.join(mids)
    X_k = X_k.join(structs)
    X_k = X_k.join(icsds)
    X_k = X_k.join(elems)
    X_k = X_k.join(fmls)
    X_k.to_pickle('./files/predicted_'+pred_prop+'_r.pkl')

