


# import everything
import domains
import LipschitzGrad
import AdaGrad
import GD
import loss
import runoptimizer
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
import customregression
from sklearn.linear_model import LinearRegression
import sys


def load_data(dataname, lossfun, coefonly = False, seed = 841992):
    
    # classification sets
    allc = ["a9a",
        "australian_scale",
        "breast-cancer_scale",
        "covtype_scale",
        "diabetes_scale",
        "heart_scale",
        "ijcnn1",
        "ionosphere_scale",
        "phishing",
        "splice_scale",
        "w1atest",
        "w8a"]
    
    # regression sets
    allr = ["abalone_scale",
            "bodyfat_scale",
            "cpusmall_scale",
            "housing_scale",
            "mg_scale",
            "space_ga_scale"]
    
    if dataname in allr:
        regression = True
    elif dataname in allc:
        regression = False
    else:
        sys.exit("Unknown data set " + dataname)
        
    
    
    if not regression:
        bigdata = np.array(["covtype", "covtype_scale", "ijcnn1"])
        if dataname in bigdata:
            if dataname == "ijcnn1":
                mydata = load_svmlight_file(dataname + '.t.bz2')
            if dataname == "covtype":
                mydata = load_svmlight_file(dataname + '.libsvm.binary.bz2')
            if dataname == "covtype_scale":
                mydata = load_svmlight_file('covtype.libsvm.binary.scale.bz2')
        else:
            mydata = load_svmlight_file(dataname + '.txt')

        tmp = pd.DataFrame(mydata[0].toarray())

        temp = tmp.values
        temp = np.append(np.ones((temp.shape[0], 1)), temp, axis = 1)

        normalydata = np.array(["australian", "australian_scale", "a9a", 'cod-rna', 'diabetes',
                                'diabetes_scale', "heart", "heart_scale", "ijcnn1", "ionosphere_scale",
                                "splice", "splice_scale", "w1a", "w1atest", "w8a"])
        if dataname in normalydata:
            y = mydata[1]
        onehalfydata = np.array(["covtype", "covtype_scale", "mushrooms"])
        if dataname in onehalfydata:
            y = np.sign(mydata[1] - 1.5)
        threeydata = np.array(["breast-cancer", "breast-cancer_scale"])
        if dataname in threeydata:
            y = np.sign(mydata[1] - 3)
        zerooneydata = np.array(['phishing'])
        if dataname in zerooneydata:
            y = np.sign(mydata[1] - 0.5)
            
        temp[:, -1] = y



    if regression:
        mydata = load_svmlight_file(dataname + '.txt')

        tmp = pd.DataFrame(mydata[0].toarray())

        temp = tmp.values
        temp = np.append(np.ones((temp.shape[0], 1)), temp, axis = 1)
        X = temp
        y = mydata[1]
        temp[:, -1] = y

    data = temp
    np.random.seed(seed)
    np.random.shuffle(data)

    X = data[:, 0:-1]
    y = data[:, -1]

    if lossfun == 'logistic':
        if coefonly:
            clf = LogisticRegression(C = np.Inf, solver='lbfgs', fit_intercept = False, max_iter = 5000).fit(X, y)
            woptlogistic = clf.coef_
            woptlogistic = woptlogistic.flatten()
            np.save(dataname + "+" + lossfun + "+coef.npy", woptlogistic)
        else:
            woptlogistic = np.load(dataname + "+" + lossfun + "+coef.npy")
            woptlogistic = woptlogistic.flatten()

        C = np.max(np.abs(np.matmul(X, woptlogistic)))

        d = X.shape[1]
        r2 = np.sqrt(np.dot(woptlogistic, woptlogistic))
        rinf = np.max(np.abs(woptlogistic))
        G = 3

        domLACeL = domains.LACeL(radius = 1 * r2, dimension = d, C = 3 * C)
        domL2 = domains.L2Ball(radius = 3 * r2, dimension = d)
        domLinfalgs = domains.LinfBall(radius = 3 * rinf, dimension = d, diagonal = True)

        lossf = loss.logistic
        gradf = loss.gradlogistic

        comparatoryhat = np.matmul(X, woptlogistic)
        comparator = np.log(1 + np.exp(-comparatoryhat.flatten() * y.flatten()))

    elif lossfun == 'hinge':
        if coefonly:
            hestimator = customregression.CustomLinearModel(X = X, Y = y,
                                                    loss_function = customregression.hingelossF,
                                                    grad_function = customregression.hingelossFgrad,
                                                    optimizer = 'SLSQP', beta_init = np.zeros(X.shape[1]),
                                                    regularization = 0)
            hestimator.fit(maxiter = 5000)
            wopthinge = hestimator.beta
            wopthinge = wopthinge.flatten()
            np.save(dataname + "+" + lossfun + "+coef.npy", wopthinge)
            print("for " + dataname + " the maximum number of iterations was not reached: " + str(hestimator.converge))
        else:
            wopthinge = np.load(dataname + "+" + lossfun + "+coef.npy")
            wopthinge = wopthinge.flatten()


        C = np.max(np.abs(np.matmul(X, wopthinge)))

        d = X.shape[1]
        r2 = np.sqrt(np.dot(wopthinge, wopthinge))
        rinf = np.max(np.abs(wopthinge))
        G = 3

        domLACeL = domains.LACeL(radius = 1 * r2, dimension = d, C = 3 * C)
        domL2 = domains.L2Ball(radius = 3 * r2, dimension = d)
        domLinfalgs = domains.LinfBall(radius = 3 * rinf, dimension = d, diagonal = True)



        lossf = loss.hinge
        gradf = loss.gradhinge

        comparatoryhat = np.matmul(X, wopthinge)
        temphinge = 1 - comparatoryhat.flatten() * y.flatten()
        temphinge = temphinge.flatten()
        comparator = np.maximum(0, temphinge)

    elif lossfun == 'squared':
        if coefonly:
            clf = LinearRegression(fit_intercept = False).fit(X, y)
            woptsquare = clf.coef_
            woptsquare = woptsquare.flatten()
            np.save(dataname + "+" + lossfun + "+coef.npy", woptsquare)
        else:
            woptsquare = np.load(dataname + "+" + lossfun + "+coef.npy")
            woptsquare = woptsquare.flatten()


        C = np.max(np.abs(np.matmul(X, woptsquare)))

        d = X.shape[1]
        r2 = np.sqrt(np.dot(woptsquare, woptsquare))
        rinf = np.max(np.abs(woptsquare))
        G = 1

        domLACeL = domains.LACeL(radius = 1 * r2, dimension = d, C = 3 * C)
        domL2 = domains.L2Ball(radius = 3 * r2, dimension = d)
        domLinfalgs = domains.LinfBall(radius = 3 * rinf, dimension = d, diagonal = True)

        lossf = loss.squaredloss
        gradf = loss.gradsquaredloss

        comparatoryhat = np.matmul(X, woptsquare)
        comparator = (comparatoryhat - y)**2

    elif lossfun == 'absolute':
        if coefonly:
            abstimator = customregression.CustomLinearModel(X = X, Y = y,
                                                            loss_function = customregression.abslossF,
                                                            grad_function = customregression.abslossFgrad,
                                                            optimizer = 'SLSQP',
                                                            beta_init = np.zeros(X.shape[1]),
                                                            regularization = 0)
            abstimator.fit(maxiter = 5000)
            woptabsolute = abstimator.beta
            woptabsolute = woptabsolute.flatten()
            np.save(dataname + "+" + lossfun + "+coef.npy", woptabsolute)
            print("for " + dataname + " the maximum number of iterations was not reached: " + str(abstimator.converge))
        else:
            woptabsolute = np.load(dataname + "+" + lossfun + "+coef.npy")
            woptabsolute = woptabsolute.flatten()


        C = np.max(np.abs(np.matmul(X, woptabsolute)))

        d = X.shape[1]
        r2 = np.sqrt(np.dot(woptabsolute, woptabsolute))
        rinf = np.max(np.abs(woptabsolute))
        G = 1

        domLACeL = domains.LACeL(radius = 1 * r2, dimension = d, C = 3 * C)
        domL2 = domains.L2Ball(radius = 3 * r2, dimension = d)
        domLinfalgs = domains.LinfBall(radius = 3 * rinf, dimension = d, diagonal = True)

        lossf = loss.absoluteloss
        gradf = loss.gradabsoluteloss

        comparatoryhat = np.matmul(X, woptabsolute)
        comparator = np.abs(comparatoryhat - y)
    else:
        raise ValueError("Unknown loss function " + lossfun)

    return (data, lossf, gradf, domLinfalgs, domLACeL, domL2, d, G, comparator)


# NOTE: coefonly=True means we are writing the coefficient file
#       coefonly=False means we are running the algorithm in singleton list algs.
#       you can only do that if the coefficient file exists.
#       we will not generate it for you. That is dangerous in parallel execution.

def RunLibsvmPap(dataname, lossfun, alg, path, coefonly = False, seed = 841992):

    data, lossf, gradf, domLinfalgs, domLACeL, domL2, d, G, comparator = load_data(dataname, lossfun, coefonly, seed)

    rinf = 1/3 * domLinfalgs.radius
    r2 = 1/3 * domL2.radius

    if not coefonly: 
        optlistsmart = []

        
        if 'AdaGrad' in alg: 
            AG = AdaGrad.AdaGrad(domain = domLinfalgs, eta = 4*rinf/np.sqrt(2))
            optlistsmart.append(AG)
            
        if 'GDn' in alg:
            GDnorm = GD.GradientDescentnorm(domain = domL2, eta = 4 * r2 / np.sqrt(2))
            optlistsmart.append(GDnorm)
            
        if 'GDt' in alg:
            GDt = GD.GradientDescentt(domain = domL2, G = G, eta = 4 * r2 / np.sqrt(2), Gupdate = True, Ghat = True)
            optlistsmart.append(GDt)
        
        if 'MGFull' in alg:
            fullslave = LipschitzGrad.FullSlave()
            MGLfull = LipschitzGrad.MetaGradL(domLACeL, fullslave)
            if d <= 300:
                optlistsmart.append(MGLfull)

        for a in alg:
            if a[0:3] == 'MGF' and a[3:].isdecimal():
                ranksketch = int(a[3:])
                Fslave = LipschitzGrad.FrequentSlave(np.min([ranksketch,d+1]))
                Fslave.shortname = "F%d" % ranksketch
                MGLF = LipschitzGrad.MetaGradL(domLACeL, Fslave)
                optlistsmart.append(MGLF)

        if 'MGdiag' in alg:
            MGLdiag = LipschitzGrad.DiagGradL(domLinfalgs)
            optlistsmart.append(MGLdiag)
        

        if all(['MGFull' == algi for algi in alg]) and d > 300:
            optlistsmart.append(MGLfull)
            Opts = runoptimizer.RunOpt(optlist = optlistsmart, data = data, loss = lossf, gradient = gradf)
        else:
            Opts = runoptimizer.RunOpt(optlist = optlistsmart, data = data, loss = lossf, gradient = gradf)
            Opts.run()

        Opts.saveregretpickle(comparator, path = path, dataname = dataname, lossname = lossfun)
