
import pandas as pd
import glob
import os

#%%



renames = {
        "MGLDiag" : "MGCo",
        "MGLF2" :   "MGF2",
        "MGLF11" :  "MGF11",
        "MGLF26" :  "MGF26",
        "MGLF51" :  "MGF51",
        "MGLFull" : "MGFull",
        }
ordering = ["data", "loss", "AdaGrad", "GDnorm", "GDt", "MGCo", "MGF2", "MGF11", "MGF26", "MGF51", "MGFull"]


wd = os.getcwd()
parent = os.path.dirname(wd)
resultfolder = parent + "/save/results"

allfiles = glob.glob(resultfolder + "/*.dat")

#%%

resultlist = [pd.read_pickle(res) for res in allfiles]

#%%
results = pd.concat(resultlist)
results = results.fillna(value = 0.0)
results = results.groupby(['data', 'loss'], axis = 0, as_index = False).sum()
results.rename(columns = renames, inplace=True)
results = results.reindex(columns= ordering)
results.to_csv('results.csv', na_rep = " ", index = False)
