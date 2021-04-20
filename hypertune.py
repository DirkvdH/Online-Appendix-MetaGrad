
from RunLibsvm import load_data
import AdaGrad
import LipschitzGrad
import GD
import runoptimizer
import numpy as np
from matplotlib.pyplot import *
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
import sys


if len(sys.argv) == 3:
    dataname = sys.argv[1]
    lossfname = sys.argv[2]
else:
    dataname = "australian_scale"
    lossfname = "logistic"
 
    

stem = "hypertune_" + dataname + "+" + lossfname

data, lossf, gradf, domLinfalgs, domLACeL, domL2, d, G, comparator = load_data(dataname, lossfname, False)

#data = data[:10]

# loop over reasonable range of \eta
etas = [2**j for j in range(-7, -3)] + [*np.linspace(1/8, 3, 24)]


def get_regret(eta):
    # scale by eta the theoretical tuning
    rinf = 1/3 * domLinfalgs.radius
    r2 = 1/3 * domL2.radius
    tuneDiag = eta * rinf
    tuneAdaGrad = eta * 4 * rinf / np.sqrt(2)
    tuneFull = eta * domLACeL.radius
    tuneGD = eta * 4 * r2 / np.sqrt(2)

    fullslave = LipschitzGrad.FullSlave(D = tuneFull)
    ranksketch2 = np.min([d+1, 2])
    F2slave = LipschitzGrad.FrequentSlave(ranksketch2, D = tuneFull)
    F2slave.shortname = "F2"
    ranksketch11 = np.min([d+1, 11])
    F11slave = LipschitzGrad.FrequentSlave(ranksketch11, D = tuneFull)
    F11slave.shortname = "F11"
    ranksketch26 = np.min([d+1, 26])
    F26slave = LipschitzGrad.FrequentSlave(ranksketch26, D = tuneFull)
    F26slave.shortname = "F26"
    ranksketch51 = np.min([d+1, 51])
    F51slave = LipschitzGrad.FrequentSlave(ranksketch51, D = tuneFull)
    F51slave.shortname = "F51"    
    algs = [
        AdaGrad.AdaGrad(domain = domLinfalgs, eta = tuneAdaGrad),
        GD.GradientDescentnorm(domain = domL2, eta = tuneGD),
        GD.GradientDescentt(domain = domL2, G = G, eta = tuneGD, Gupdate = True, Ghat = True),
        LipschitzGrad.MetaGradL(domLACeL, fullslave),
        LipschitzGrad.MetaGradL(domLACeL, F2slave),
        LipschitzGrad.MetaGradL(domLACeL, F11slave),
        LipschitzGrad.MetaGradL(domLACeL, F26slave),
        LipschitzGrad.MetaGradL(domLACeL, F51slave),
        LipschitzGrad.DiagGradL(domLinfalgs, D=tuneDiag)
        ]
    
    # measure regret
    Opts = runoptimizer.RunOpt(optlist = algs, data = data, loss = lossf, gradient = gradf)
    Opts.run()
    q = Opts.printregret(comparator, "blurf", "fnork")
    return q

if __name__ == '__main__':
    outs = Parallel(n_jobs=multiprocessing.cpu_count(), backend='multiprocessing')(delayed(get_regret)(eta) for eta in etas)
    regrets = pd.concat(outs)
    regrets.index = etas # TODO: do this directly?
    regrets.index.name = "D"
    regrets = regrets.iloc[:,0:9]
    
    regrets.plot()
    xscale("log")
    savefig(stem + ".svg")
    savefig(stem + ".pdf")
    
    
    regrets.to_csv(stem + ".csv", na_rep = " ")
