import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure   
from collections import Counter

api_key=<MP api key>

def make_formula(word):
  newword=[]
  for chrc in word:
    if chrc.isdigit():
      newword.append('$_'+chrc+'$')
    else:
      newword.append(chrc)
  sep=''
  return sep.join(newword)


def get_formula(mid):
    with MPRester(api_key) as m:
        data=m.get_doc(mid)
        #formula=data["pretty_formula"]
        #spgnum=data["spacegroup"]["number"]
        #spg=data["spacegroup"]["crystal_system"]
        #if len(data["icsd_ids"])==0:
        #    icsd='No'
        #else:
        #    icsd='Yes'
        #elements=data["elements"]
        #struct=data["structure"]
        ehull=data['e_above_hull']
        #return formula,icsd,elements,struct,spgnum,spg
        return ehull 

def chrc_spg(structure):
    spgan=SpacegroupAnalyzer(structure)
    spg=spgan.get_crystal_system()
    spgnum=spgan.get_space_group_number()
    return spg,spgnum

def chrc_comp(nsp, nssp, val):
    classified=False
    if any(c in val for c in ('Pb', 'Cd')):
        nsp.append('Toxic')
    if any(c in val for c in ('Cl', 'Br', 'I')):
        nsp.append('Halides')
        if 'Cl' in val:
            nssp.append('Chloride')
        if 'B' in val:
            nssp.append('Bromide')
        if 'I' in val:
            nssp.append('Iodide')
        classified=True
    if any(c in val for c in ('Se', 'Te', 'S')):
        nsp.append('Chalcogenides')
        if 'Se' in val:
            nssp.append('Selenides')
        if 'Te' in val:
            nssp.append('Tellurides')
        if 'S' in val:
            nssp.append('Sulphides')
        classified=True
    if any(c in val for c in ('N', 'P', 'As', 'Sb')):
        nsp.append('Pnictides')
        if 'N' in val:
            nssp.append('Nitride')
        if 'P' in val:
            nssp.append('Phosphide')
        if 'As' in val:
            nssp.append('Arsenide')
        if 'Sb' in val:
            nssp.append('Antimonide')
        classified=True
    if 'O' in val:
        nsp.append('Oxides')
        classified=True
    if not classified:
        nsp.append('Others')
    return nsp, nssp

df_1 = pd.read_pickle("./files/predicted_aac.pkl")
df_2 = pd.read_pickle("./files/predicted_iac.pkl")
df_3 = pd.read_pickle("./files/predicted_ebe.pkl")
df_4 = pd.read_pickle("./files/predicted_qp_gap_r.pkl")
df_5 = pd.read_pickle("./files/predicted_ebe_r.pkl")

df_1=df_1[["predicted_aac","material_id"]]
df_2=df_2[["predicted_iac","material_id"]]
df_3=df_3[["predicted_ebe","material_id"]]
df_4=df_4[["predicted_qp_gap_r","material_id","dft_gap","structure","icsd","elements","formula"]]
df_5=df_5[["predicted_ebe_r","material_id"]]

df = df_1.merge(df_2, on=['material_id'])
df = df.merge(df_3, on=['material_id'])
df = df.merge(df_4, on=['material_id'])
df = df.merge(df_5, on=['material_id'])

print(df['predicted_ebe'].value_counts()[1],df['predicted_iac'].value_counts()[1],df['predicted_aac'].value_counts()[1])

dfs = df[df['predicted_aac'] == 1]
print('aac sorting',dfs.shape)
dfs = dfs[dfs['predicted_iac'] == 1]
print('iac sorting',dfs.shape)
dfs = dfs[dfs['predicted_ebe'] == 1]
print('ebe sorting',dfs.shape)
mids=list(dfs['material_id'])
print(len(mids))
#for n,mid in enumerate(mids):
#    print(mid,fmls[n])
#print(dfs.describe())

nsp=[]
nssp=[]
for n, mid in enumerate(mids):
    #formula,icsd,elements,struct_d,spgnum,spg=get_formula(mid)
    cond = (dfs['material_id'] == mid)

    qpg = dfs[cond].predicted_qp_gap_r.values[0]
    dftg = dfs[cond].dft_gap.values[0]
    ebe = dfs[cond].predicted_ebe_r.values[0]
    struct= dfs[cond].structure.values[0]
    icsd=dfs[cond].icsd.values[0]
    elements=dfs[cond].elements.values[0]
    formula=dfs[cond].formula.values[0]

    nsite=struct.num_sites
    nsp, nssp=chrc_comp(nsp, nssp, elements)
    spg,spgnum=chrc_spg(struct)
    fml=make_formula(formula)
    ehull=get_formula(mid)

    #print(n+1, mid, fml, icsd, spgnum, spg, nsite, '%5.2f' %dftg, '%5.2f' %qpg, '%5.2f' %ebe, sep=' & ', end='')
    #print(n+1, mid, fml, icsd, spgnum, spg, nsite, '%5.2f' %ehull, '%5.2f' %dftg, '%5.2f' %qpg, '%5.2f' %ebe, sep=' & ', end='')
    if ehull==0.:
        print('stable',n+1, mid, fml, icsd, spgnum, spg, nsite, '%5.2f' %ehull, '%5.2f' %dftg, '%5.2f' %qpg, '%5.2f' %ebe)
    else:
        print(n+1, mid, fml, icsd, spgnum, spg, nsite, '%5.2f' %ehull, '%5.2f' %dftg, '%5.2f' %qpg, '%5.2f' %ebe)
#    print(' '+r'\\')
#    print('\hline')
#
lc = Counter(nsp)
lck=[]
lcv=[]
labels = ['Halides', 'Others', 'Chalcogenides', 'Pnictides', 'Oxides','Toxic']
for key in labels:
    lck.append(key)
    lcv.append(lc[key])
    #print(mid,formula,spgr)
print(lck,lcv)

lc = Counter(nssp)
lck=[]
lcv=[]
labels=['Nitride','Phosphide','Arsenide','Antimonide','Selenides','Tellurides','Sulphides']
for key in labels:
    lck.append(key)
    lcv.append(lc[key])
    #print(mid,formula,spgr)
print(lck,lcv)
##plt.scatter(df['dft_gap'], df['predicted_qpg'],s=10,alpha=0.3)
#maxq=max(df['predicted_qpg'])+1
#plt.plot([0,maxq],[0,maxq],c='k',lw=2,ls='--')

#plt.xlim(0,maxq)
#plt.ylim(0,maxq)
#plt.xlabel('DFT gap',fontsize=15)
#plt.ylabel('Predicted QP gap',fontsize=15)
#plt.xticks([0,5,10,15],fontsize=15)
#plt.yticks([0,5,10,15],fontsize=15)
#plt.savefig('predicted_qpg.png', dpi=200,bbox_inches='tight', pad_inches=0.1)
