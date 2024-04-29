#!/usr/bin/env python

# standard imports 
import pandas as pd
import csv
# imports
import numpy as np, os

#from ipywidgets import FloatProgress
import matplotlib.colors as colors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
stdsclr = StandardScaler()
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from collections import Counter

from pathlib import Path
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval    
from pymatgen.core.structure import Structure   
from pymongo import MongoClient

# matminer
#from matminer import PlotlyFig
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.featurizers.composition import ElementProperty, AtomicPackingEfficiency, ElectronegativityDiff, ValenceOrbital, OxidationStates, ElementFraction, AtomicOrbitals 
from matminer.featurizers.structure import ElectronicRadialDistributionFunction, GlobalSymmetryFeatures, DensityFeatures
from matminer.featurizers.bandstructure import BandFeaturizer
from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition

import matplotlib.pyplot as plt
import matplotlib as mp
from scipy.integrate import simps
# MongoDB, import data, def functions
#from automatminer import MatPipe
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor


from matminer.featurizers.bandstructure import BandFeaturizer
from matminer.featurizers.dos import DOSFeaturizer
#from pymatgen import MPRester


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42




def query_db(user,passwd,dbname,colname,analysis_data):
    print("Getting data from Database", dbname)
    conn = MongoClient(host, 27017, authsource='admin', username=user, password=passwd)
    db = conn[dbname]
    col = db.get_collection(colname)
    fws = col.find({})
    i=0
    for fw in fws:
      if "indirect_gap" in fw.keys():
        i=i+1
        epsl=np.array([fw["dielectric constant"][0][0],fw["dielectric constant"][1][1],fw["dielectric constant"][2][2]])
        col = db.get_collection('QP_Results')
        gw_item = col.find_one({"material_id": fw["material_id"], "task_label": {'$regex':'^scGW'}},{ "indirect_gap": 1, "incar": 1, "direct_gap": 1})
        col = db.get_collection('EMCPY_Results')
        emc_item = col.find_one({"material_id": fw["material_id"]},{ "hole effective_mass": 1, "electron effective_mass": 1})
        col = db.get_collection('BSE_Results')
        bse_item = col.find_one({"material_id": fw["material_id"]},{ "optical_transition": 1, 'frequency': 1, 'epsilon_1': 1, 'epsilon_2': 1})
        nan=float('nan')
        if gw_item==None or gw_item["incar"]["NOMEGA"]==1:
            qpg=nan
        else:
            qpg=gw_item["indirect_gap"]
            qpg0=gw_item["direct_gap"]
        if emc_item==None:
            emass=nan
            hmass=nan
        else:
            emass=np.array(emc_item["electron effective_mass"])
            hmass=np.array(emc_item["hole effective_mass"])
        if bse_item==None:
            opg=nan
            ebe=nan
        else:
            opg=bse_item["optical_transition"][0][0]
            freq=bse_item["frequency"]
            eps1=bse_item["epsilon_1"]
            eps2=bse_item["epsilon_2"]
            try:
                avg_iabs,anis_iabs=calc_iabs(freq,eps1,eps2,[3.5, 4.2])
            except:
                avg_iabs,anis_iabs=float('nan'),float('nan') 
            ebe=qpg0-opg

        data = {"material_id": fw["material_id"],
            "qp_gap": qpg,
            "structure": fw["structure"],
            "formula": fw["formula_pretty"],
            "dft_gap": fw["indirect_gap"],
            "dielectric_constant": epsl,
            "hmass": hmass,
            "emass": emass,
            "optical_gap": opg, 
            "ebe": ebe,
            "iac": avg_iabs,
            "aac": anis_iabs,
            "source": dbname}
        print('material entry', i)
        pdd = pd.DataFrame(data=[data])
        analysis_data = analysis_data.append(pdd, ignore_index=True)
    return analysis_data


def eps2abs(freq,eps1,eps2):
    velc=29979245800
    absp=[]
    for i in range(len(freq)):
       en=freq[i]
       valr=eps1[i]
       vali=eps2[i]
       eps=(valr**2+vali**2)**0.5
       k=((eps-valr)/2.0)**0.5
       absp.append(2*(en*2.417990504024e14)*k/velc)

    return absp


def calc_iabs(freq,eps1,eps2, freq_range):

    fmin=freq_range[0]
    fmax=freq_range[1]
    eps1x=[row[0] for row in eps1]
    eps1y=[row[1] for row in eps1]
    eps1z=[row[2] for row in eps1]

    eps2x=[row[0] for row in eps2]
    eps2y=[row[1] for row in eps2]
    eps2z=[row[2] for row in eps2]
    
    abspx=eps2abs(freq,eps1x,eps2x)
    abspy=eps2abs(freq,eps1y,eps2y)
    abspz=eps2abs(freq,eps1z,eps2z)

    iabsx=int_abs(freq,abspx,fmin,fmax)
    iabsy=int_abs(freq,abspy,fmin,fmax)
    iabsz=int_abs(freq,abspz,fmin,fmax)

    iabs=[iabsx,iabsy,iabsz]
    avg_iabs=np.mean(iabs)
    anis_iabs=min(iabs)/max(iabs)

    return avg_iabs,anis_iabs

def int_abs(x,y,lm1,lm2):
    xm=[]
    ym=[]
    for i in range(len(x)):
        if lm1<=x[i]<=lm2:
            xm.append(x[i])
            ym.append(y[i])
    I1 = simps(ym, xm)
    return I1*1e-4
                                                                

def Add_features(dbnames,outfile):
    #Remove duplicates
    frames=[]
    for dbname in dbnames:
        dfr = pd.read_pickle(dbname) 
        frames.append(dfr)
    df = pd.concat(frames)
    #df_mp = pd.read_pickle("./MPgap.pkl") 
    #if remove_db == True:
    #    df = df[df["source"]=="testingdb"]
    #df = df.merge(df_mp, on=['material_id'])
    df=df.drop_duplicates(subset =["material_id"],ignore_index=True,keep='last')
    #Format the structure column
    try:
        df['structure'] = pd.Series([Structure.from_dict(df['structure'][i])        for i in range(df.shape[0])], df.index)
    except:
        df['structure'] = pd.Series([df['structure'][i]       for i in range(df.shape[0])], df.index)
    #df=df.rename(columns={"spacegroup.number": "space_group"}) 
    print("adding Features to the dataset")
    # add features StrToComposition to var: df (var: df)                            
    df = StrToComposition().featurize_dataframe(df, "formula")                      
    # load some data from magpie about ElementProperty (var: df)                    
    ep_feat = ElementProperty.from_preset(preset_name="magpie")                     
    df = ep_feat.featurize_dataframe(df, col_id="composition")  # input the "composition" column to the featurizer
    # add some other features                                               
    df = CompositionToOxidComposition().featurize_dataframe(df, "composition")      
    os_feat = OxidationStates() 
    df = os_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)  
    os_feat=AtomicOrbitals()
    df = os_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)     
    os_feat=ElectronegativityDiff()
    df = os_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)     
    #os_feat=ElementFraction()
    #df = os_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)     
    # add density                                                                   
    df_feat = DensityFeatures()        
    df = df_feat.featurize_dataframe(df, "structure",ignore_errors=True)  
    # input the structure column to the featurizer
    #df_feat = GlobalSymmetryFeatures()        
    #df = df_feat.featurize_dataframe(df, "structure",ignore_errors=True)  
    #print(df.columns)
    # input the structure column to the featurizer
    #df = add_dosbs_features(df)
    #df=replace_effmass(df)
    df.to_pickle(outfile)        

def PrinCA(X,ncomp):
    cols=[]
    for i in range(ncomp):
        cols.append('PC'+str(i+1))
    pca = PCA(n_components=ncomp)
    Xs = stdsclr.fit_transform(X)
    X_pca = pca.fit_transform(Xs)
    PCA_df = pd.DataFrame(data = X_pca, columns = cols)
    return PCA_df

def calc_avg(row,column):
    try:
        return np.mean(row[column])
    except:
        return float('nan')

def calc_std(row,column):
    try:
        return np.std(row[column])
    except:
        return float('nan')

def calc_range(row,column):
    try:
        return max(row[column])-min(row[column])
    except:
        return float('nan')

def calc_max(row,column):
    try:
        return max(row[column])
    except:
        return float('nan')

def calc_min(row,column):
    try:
        return min(row[column])
    except:
        return float('nan')


def conv_char(df,label):
    orbitals=['s','p','d','f']
    for index, row in df.iterrows():
        if row[label] not in orbitals:
            row[label]==float("nan")

    df = df.dropna()
    df_cat = df[label]
    encoder = LabelEncoder()
    df_cat_encoded = encoder.fit_transform(df_cat)
    df[label+'_encoded']=df_cat_encoded
    return df

def replace_effmass(df):
    #df_o = df.drop(['emass','hmass'], axis=1)  
    df_o = df 
    print('------------effective masses added------------')
    df_effmass = pd.read_pickle("./db_effmass.pkl")
    df = pd.merge(df_o, df_effmass, on="material_id")
    return df

def hist_feat_comp(df,nmax):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    nsp=[]
    for val in df["composition"]:
        if 'O' in val:
            nsp.append('Oxides')
        elif any(c in val for c in ('N', 'P', 'As', 'Sb')):
            nsp.append('Pnictides')
        elif any(c in val for c in ('Cl', 'Br', 'I')):
            nsp.append('Halides')
        elif any(c in val for c in ('Se', 'Te', 'S')):
            nsp.append('Chalcogenides')
        else:
            nsp.append('Others')

    lc = Counter(nsp)
    lck=[]
    lcv=[]
    for key in lc.keys():
        lck.append(key)
        lcv.append(lc[key])
    plot_pie(lck, lcv, np.zeros(len(lck)))

def hist_feat_spg(df,nmax):
    from collections import Counter
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    nsp=[]
    #for val in df["structure"]:
    for val in df["composition"]:
        #spg=SpacegroupAnalyzer(val,symprec=0.01, angle_tolerance=5.0)
        #nsp.append(spg.get_crystal_system())
        nsp.append(len(val))

    lc = Counter(nsp)
    lck=[]
    lcv=[]
    for key in lc.keys():
        #lck.append(r'N$_{elem}$='+str(key))
        lck.append(str(key))
        lcv.append(lc[key])
    #lck[4], lck[5] = lck[5], lck[4]    
    #lcv[4], lcv[5] = lcv[5], lcv[4]    
    plot_pie(lck, lcv, np.zeros(len(lck)))
    #lf = pd.DataFrame.from_dict(lc, orient='index')
    #lf.plot(kind='barh')
    #plt.show()

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
def plot_pie(labels,sizes,explode):
    #print(labels,sizes)
    nc=len(labels)
    fig1, ax1 = plt.subplots(figsize=(8,8))
    cmap = mp.cm.get_cmap('Set2')
    outer_colors = cmap(np.arange(nc))
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.0f%%',
        shadow=False, startangle=90, colors=outer_colors,  textprops={'fontsize': 25})
    Drawing_uncolored_circle = plt.Circle( (0.0, 0.0 ),0.4 ,fill = True, facecolor='w')

    ax1.add_artist( Drawing_uncolored_circle )
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('./figures/pie.pdf', dpi=200,bbox_inches='tight', pad_inches=0.1, transparent=True)



def prepare_data(filename,xmx,excluded,pred_prop,bns):
    df_o = pd.read_pickle(filename)
    df=df_o
    #hist_feat_spg(df,4)
    #hist_feat_comp(df,5)
    #df=conv_char(df,'HOMO_character')
    #df=conv_char(df,'LUMO_character')
    

    df["mean_dc"]=df.apply (lambda row: calc_avg(row,"dielectric_constant"), axis=1) 
    df["range_dc"]=df.apply (lambda row: calc_range(row,"dielectric_constant"), axis=1)
    #df["min_dc"]=df.apply (lambda row: calc_min(row,"dielectric_constant"), axis=1) 
    #df["max_dc"]=df.apply (lambda row: calc_max(row,"dielectric_constant"), axis=1)

    df["mean_emass"]=df.apply (lambda row: calc_avg(row,"emass"), axis=1)
    df["range_emass"]=df.apply (lambda row: calc_range(row,"emass"), axis=1)
    #df["min_emass"]=df.apply (lambda row: calc_min(row,"emass"), axis=1)
    #df["max_emass"]=df.apply (lambda row: calc_max(row,"emass"), axis=1)

    df["mean_hmass"]=df.apply (lambda row: calc_avg(row,"hmass"), axis=1)
    df["range_hmass"]=df.apply (lambda row: calc_range(row,"hmass"), axis=1)
    #df["min_hmass"]=df.apply (lambda row: calc_min(row,"hmass"), axis=1)
    #df["max_hmass"]=df.apply (lambda row: calc_max(row,"hmass"), axis=1)

    df = df[df['mean_emass'] < np.inf]
    df = df[df['mean_hmass'] < np.inf]

    #fml=list(df["formula"])
    #cfml=Counter(fml)
    #for keys in cfml.keys():
    #    if cfml[keys]>1:
    #        cond = (df['formula'] == keys)
    #        qpg = df[cond].qp_gap.values
    #        dftg = df[cond].dft_gap.values
    #        mids = df[cond].material_id.values
    #        spg = df[cond].spacegroup.values
    #        for i,mid in enumerate(mids):
    #            print(mid,keys,'%5.2f' %dftg[i],'%5.2f' %qpg[i], spg[i])
    #        print()
    if pred_prop != None:
        df = df.drop(excluded, axis=1) 
        #df = df[df['mean_dc'] <= 25.0]
        df = df[df[pred_prop] <= xmx]
        df = df[df[pred_prop] > 0.0]
        df = df.dropna()
        X= df.drop([pred_prop], axis=1)
        y = df[pred_prop].values     
        plt_hst(df, xmx, pred_prop, bns)
        print('Number of materials before and after removing nans:', df_o.shape[0],df.shape[0])
        return X,y
    else:
        X= df
        print('Number of materials before and after removing nans hh:', df_o.shape[0],df.shape[0])
        return X

def Random_Forest(X,y):
    print('Performing RF ..')
    rf = RandomForestRegressor(n_estimators=15, random_state=1)
    rf.fit(X, y)
    err=cv_check(rf,X,y)
    return rf,err

def Lasso(X,y):
    print('Performing LASSO')
    std      = np.std(X, axis=0)
    X /= std
    rf = linear_model.Lasso(alpha=5000,
                               positive=True,
                               fit_intercept=False,
                               max_iter=1000,
                               tol=0.0001)
    rf.fit(X, y)
    err=cv_check(rf,X,y)
    return rf,err


def MLP(X,y):
    print('Performing MLP ..')
    Xs = stdsclr.fit_transform(X)
    regr = MLPRegressor(hidden_layer_sizes=(64,32,16,8), solver='adam',random_state=1, max_iter=1000).fit(Xs, y)
    err=cv_check(regr,Xs,y)
    return regr,err

def KRR(X,y):
    print('Performing KRR ..')
    #Xs = stdsclr.fit_transform(X)
    kr = KernelRidge(alpha=1.0)
    kr.fit(X,y)
    err=cv_check(kr,X,y)
    return kr,err

def Sup_VR(X,y):
    print('Performing SVR ..')
    Xs = stdsclr.fit_transform(X)
    regr = SVR(C=1.0, epsilon=0.2)
    regr.fit(Xs, y)
    err=cv_check(regr, Xs, y)
    return regr,err

def GPR(X,y):
    print('Performing GPR ..')
    Xs = stdsclr.fit_transform(X)
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    #kernel = 1 * RBF(1.0)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gaussian_process.fit(Xs, y)
    cv_check(gaussian_process,Xs,y)
    return gaussian_process

def cv_check(rf,X,y):
    trn_res=[rf.score(X, y),np.sqrt(mean_squared_error(y_true=y, y_pred=rf.predict(X)))]
    #print('training R2 = %.2f' %trn_res[0]+' RMSE = %.2f' % trn_res[1])
    # compute cross validation scores for models
    crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
    r2_scores = cross_val_score(rf, X, y, scoring='r2', cv=crossvalidation, n_jobs=-1)
    scores = cross_val_score(rf, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=-1)
    #scores = cross_val_score(rf, X, y, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=-1)
    for score in r2_scores:
        if score<0:
            ind=np.where(r2_scores == score)
            r2_scores=np.delete(r2_scores,ind)
            scores=np.delete(scores,ind)
    rmse_scores = [np.sqrt(abs(s)) for s in scores]
    #rmse_scores = scores
    #print('Cross-validation results:')
    tst_res=[np.mean(np.abs(r2_scores)),np.mean(np.abs(rmse_scores))]
    print('R2 = %.2f' %tst_res[0]+' RMSE = %.2f' % tst_res[1])
    return trn_res[0], trn_res[1], tst_res[0], tst_res[1]

def test_train_check(rf_reg,frac,X,y):
    rmse=0.
    for rs in [6789,9876,1234,4321,5678,8765]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=frac, random_state=rs)
        rf_reg = RandomForestRegressor(n_estimators=50, random_state=1)
        rf_reg.fit(X_train, y_train)
        rmse=rmse+np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg.predict(X_train)))
        #rmse=rmse+np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg.predict(X_test)))

    rmse=rmse/6.0 
    # get fit statistics
    print('training R2 = ' + str(round(rf_reg.score(X_train, y_train), 3)))
    print(len(y_train),rmse)
    print('training RMSE =','%5.3f' %(rmse/6.0))
    print('test R2 = ' + str(round(rf_reg.score(X_test, y_test), 3)))
    print('test RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg.predict(X_test))))

def plot_result(rf,X,y,xmx,ex_l,pred_prop):
    if pred_prop=='ebe':
        bestft='mean_dc'
        proplabel="EBE"
        bestftlabel=r'$E^{DFT}_{g}$'
        cbticks=[5,20]
    if pred_prop=='qp_gap':
        bestft='dft_gap'
        proplabel="QPG"
        bestftlabel=r'$E^{DFT}_{g}$'
        cbticks=[2,8]
    fig, ax1 = plt.subplots(figsize=(7,5))
    #plt.title('Random Forest Regression')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    Z_test=X_test[bestft]
    Z_train=X_train[bestft]
    Z=X[bestft]
    ax1.plot([0,xmx+ex_l],[0,xmx+ex_l],c='k',ls='--')
    #y1=np.array(x)-0.2
    #y2=np.array(x)+0.2
    #ax.fill_between(x, y1, y2,alpha=0.2,facecolor='k')
    plt.xlim(0,xmx+ex_l)
    plt.ylim(0,xmx+ex_l)
    plt.xticks([0.0,xmx/3,2*xmx/3,xmx],fontsize=25)
    plt.yticks([0.0,xmx/3,2*xmx/3,xmx],fontsize=25)
    plt.xlabel(r"Computed "+proplabel+" (eV)",fontsize=25)
    ax1.set_ylabel(r"Predicted "+proplabel+" (eV)",fontsize=25)
    #plt.legend(fontsize=25)
    ax2 = ax1.twinx()
    ax2.set_yticks([])
    ax2.hist(y, bins=15, lw=1, ec="b", fc='k', alpha=0.2, histtype='stepfilled')
    if pred_prop=='ebe':
        vmn=Z.min()
        vmx=25.0
        ax1.scatter(y_train, rf.predict(X_train), c=Z_train, 
             norm=colors.LogNorm(vmin=vmn, vmax=vmx), alpha=0.5, s=100, 
             label='Train', cmap='viridis', marker='P', edgecolor='black')
        ax1.scatter(y_test, rf.predict(X_test), c=Z_test,
             norm=colors.LogNorm(vmin=vmn, vmax=vmx), alpha=0.5, s=100, 
             label='Test', cmap='viridis', edgecolor='black')
    if pred_prop=='qp_gap':
        vmn=1e-2
        vmx=Z.max()
        ax1.scatter(y_train, rf.predict(X_train), c=Z_train, 
             norm=colors.Normalize(vmin=vmn, vmax=vmx), alpha=0.5, s=100, 
             label='Train', cmap='viridis', marker='P', edgecolor='black')
        ax1.scatter(y_test, rf.predict(X_test), c=Z_test,
             norm=colors.Normalize(vmin=vmn, vmax=vmx), alpha=0.5, s=100, 
             label='Test', cmap='viridis', edgecolor='black')
    sc = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmn, vmax=vmx))
    cbar=plt.colorbar(sc, ticks=cbticks)
    cbar.ax.tick_params(labelsize=25) 
    cbar.set_label(bestftlabel,fontsize=25, labelpad=0)
    plt.savefig('./figures/fit.pdf',dpi=200,bbox_inches='tight', pad_inches=0.1,transparent=True)

def imp_feat(rf,X,y,nf):
    fig, ax = plt.subplots(figsize=(4,8),facecolor="w")
    importances = rf.feature_importances_
    included = X.columns.values
    indices = np.argsort(importances)[::-1]
    xb=[]
    labels=[]
    list_f=[]
    for ind in range(nf,len(included[indices])):
        list_f.append(included[indices[ind]])
    for ind in range(nf):
        xb.append(nf-ind)
        if 'MagpieData maximum MeltingT' in str(included[indices][ind]):
            labels.append(r'$T^{max}_{melt}$')
        elif 'MagpieData avg_dev Row' in str(included[indices][ind]):
            labels.append(r'$Row^{\sigma}$')
        elif 'range EN difference' in str(included[indices][ind]):
            labels.append(r'END$^{range}$')
        elif 'minimum EN difference' in str(included[indices][ind]):
            labels.append(r'END$^{min}$')
        elif 'std_dev EN difference' in str(included[indices][ind]):
            labels.append(r'END$^{s\sigma}$')
        elif 'MagpieData avg_dev GSvolume_pa' in str(included[indices][ind]):
            labels.append(r'$VPA^{\sigma}$')
        elif 'MagpieData mean GSvolume_pa' in str(included[indices][ind]):
            labels.append(r'$VPA^{avg}$')
        elif 'MagpieData range MeltingT' in str(included[indices][ind]):
            labels.append(r'$T^{range}_{melt}$')
        elif 'MagpieData mean MeltingT' in str(included[indices][ind]):
            labels.append(r'$T^{avg}_{melt}$')
        elif 'MagpieData mean Row' in str(included[indices][ind]):
            labels.append(r'$Row_{mean}$')
        elif 'MagpieData mean CovalentRadius' in str(included[indices][ind]):
            labels.append(r'$R^{Covalent}_{mean}$')
        elif 'MagpieData maximum CovalentRadius' in str(included[indices][ind]):
            labels.append(r'$R^{Covalent}_{max}$')
        elif 'MagpieData minimum NValence' in str(included[indices][ind]):
            labels.append(r'$N_{val}^{min}$')
        elif 'MagpieData avg_dev MeltingT' in str(included[indices][ind]):
            labels.append(r'$T_{melt}^{\sigma}$')
        elif 'MagpieData avg_dev Number' in str(included[indices][ind]):
            labels.append(r'$N_{atomic}^{\sigma}$')
        elif 'mean_dc' in str(included[indices][ind]):
            labels.append(r'$\epsilon^{avg}$')
        elif 'range_dc' in str(included[indices][ind]):
            labels.append(r'$\epsilon_{range}$')
        elif 'mean_emass' in str(included[indices][ind]):
            labels.append(r'$\mu^{avg}_e$')
        elif 'range_emass' in str(included[indices][ind]):
            labels.append(r'$\mu^{range}_e$')
        elif 'mean_hmass' in str(included[indices][ind]):
            labels.append(r'$\mu^{avg}_h$')
        elif 'range_hmass' in str(included[indices][ind]):
            labels.append(r'$\mu^{range}_h$')
        elif 'dft_gap' in str(included[indices][ind]):
            labels.append(r'$E_g^{DFT}$')
        elif 'packing fraction' in str(included[indices][ind]):
            labels.append('APF')
        elif 'MagpieData avg_dev NpUnfilled' in str(included[indices][ind]):
            labels.append(r'$Np_{emp}^{\sigma}$')
        elif 'MagpieData mean NpUnfilled' in str(included[indices][ind]):
            labels.append(r'$Np_{emp}^{avg}$')
        elif 'MagpieData mode Electronegativity' in str(included[indices][ind]):
            labels.append(r'$\chi^{mode}$')
        else:
            labels.append(included[indices][ind])
        #print(included[indices][ind])    
        ax.annotate(str('%5.1f' %(importances[indices][ind]*100)+'%'), xy=(0.1, nf-ind), xycoords='data',
            xytext=(0.1, nf-0.1-ind), fontsize=20)
    ax.barh(xb,importances[indices][0:nf]*100,facecolor='g',alpha=0.7)
    plt.yticks(xb, labels,fontsize=20)
    plt.xticks([0,20,40,60],fontsize=20)
    plt.xlabel('Importance in %',fontsize=20)
    plt.savefig('./figures/barplot.pdf',dpi=200,bbox_inches='tight', pad_inches=0.1,transparent=True)
    #plt.savefig('./figures/barplot.png',dpi=200,bbox_inches='tight', pad_inches=0.1)
    plt.cla()

    return list_f


def plt_hst(df,xmx,pred_prop,bns):
    fig, ax = plt.subplots(figsize=(4,8),facecolor="w")
    ax.set_ylabel("Frequency",fontsize=15)
    ax.tick_params(labelsize=15)
    ax.hist(df.where(df[pred_prop] != 0)[pred_prop], bins=bns, density=False,alpha=0.5,facecolor='r')
    ax.set_xlabel("EBE in eV",fontsize=15)
    #ax.set_xlim(0,xmx)
    #ax.set_xticks([0,5,10])
    #plt.savefig('hist.png',dpi=200,bbox_inches='tight', pad_inches=0.1)
    #plt.show()
    plt.cla()
