from ml_regression import ML_reg, mp_reg

pred_prop='ebe'
reg_cut=6.1
nprop=8
algo='RFC'

#algos=['RFC','MLP','SVR','KRR']
#for algo in algos:
#for nprop in range(1,12):
X,y,rf=ML_reg(reg_cut, nprop, pred_prop, algo)
#mp_reg(X, rf, pred_prop)
#for nprop in range(2,10):
#    print(nprop)
#    X,y,rf=ML_reg(reg_cut, nprop, pred_prop, algo)

