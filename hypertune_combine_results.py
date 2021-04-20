# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:07:50 2020

@author: dirkv
"""
import pandas as pd
import glob
import os
import numpy as np


#%%

wd = os.getcwd()
parent = os.path.dirname(wd)
allfiles = glob.glob("hypertune*.csv")

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
   

def get_regrets(file):
    regrets = pd.read_csv(file, index_col="D")
    regrets.rename(columns = renames, inplace=True)    
    regrets = pd.DataFrame(regrets.min()).transpose()
    parts = file.split("+")
    regrets["data"] = parts[0][10:];
    regrets["loss"] = parts[1][:-4];
    regrets = regrets.reindex(columns= ordering)
    #regrets = pd.concat([pd.DataFrame([[file]], columns=["file"]), regrets])
    return regrets

resultlist = [get_regrets(res) for res in allfiles]

#%%
results = pd.concat(resultlist)
results = results.groupby(['data', 'loss'], axis = 0, as_index = False).sum()

results.to_csv('hypertune_results.csv', na_rep = " ", index = False)
