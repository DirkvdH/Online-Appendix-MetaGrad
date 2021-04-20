
import sys
import pandas as pd
from matplotlib.pyplot import *

if len(sys.argv) == 3:
    dataname = sys.argv[1]
    lossfname = sys.argv[2]
else:
    dataname = "australian_scale"
    lossfname = "logistic"
  
renames = {
        "MGLDiag" : "MGCo",
        "MGLF2" :   "MGF2",
        "MGLF11" :  "MGF11",
        "MGLF26" :  "MGF26",
        "MGLF51" :  "MGF51",
        "MGLFull" : "MGFull",
        }
ordering = ["AdaGrad", "GDnorm", "GDt", "MGCo", "MGF2", "MGF11", "MGF26", "MGF51", "MGFull"]
   

stem = "hypertune_" + dataname + "+" + lossfname
regrets = pd.read_csv(stem + ".csv", index_col="D")

regrets.rename(columns = renames, inplace=True)
regrets = regrets.reindex(columns= ordering)


matplotlib.pyplot.rcParams['figure.figsize'] = [6, 4]
# this is how I made the plot for australian
regrets.plot()
ubd = np.maximum(2*np.max(np.min(regrets)), 2*np.min(regrets[regrets.index==1], axis=1).squeeze())
rg = regrets.index[np.any(regrets <= ubd, axis=1)];
ylim(np.min(.9*np.min(regrets)), ubd)
xlim(0, np.maximum(1.1, rg[-1])); 
legend(loc="lower right", fontsize="small")
xlabel("factor wrt theoretical tuning")
ylabel("regret")
vlines(1, 0, 1e9,colors="k", linestyles="dashed")
savefig(stem + ".pdf")
#xscale("log");
