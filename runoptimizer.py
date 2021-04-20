
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx
from copy import deepcopy
from pandas.plotting import table
import pandas as pd
import sys

#%%



class RunOpt:
    def __init__(self, optlist, data, loss, gradient, clip = 10e30, verbose = True, angle = False, ed = False):
        self.optlist = optlist
        self.data = data
        self.loss = loss
        self.gradient = gradient
        self.T = data.shape[0]
        self.counter = 0
        names = [opt.getname() for opt in optlist]
        numbers = list(range(len(optlist)))
        self.namnum = dict(zip(names, numbers))
        self.clip = clip
        self.verbose = verbose
        self.angle = angle
        self.ed = ed
        self.losslist = dict([(opt.getname(), []) for opt in self.optlist])
    
    def run(self, verbose = False):
        losslist = dict([(opt.getname(), []) for opt in self.optlist])
        # Disable storing of gradients and weights, which is a huge memory hog!
        #gradlist = dict([(opt.getname(), []) for opt in self.optlist])
        #weightlist = dict([(opt.getname(), []) for opt in self.optlist])
        anglelist = dict([(opt.getname(), []) for opt in self.optlist])
        edlist = dict([(opt.getname(), []) for opt in self.optlist])
        
        if verbose:
            losstot = dict([(opt.getname(), 0) for opt in self.optlist])
        
        for t in range(self.T):
            i = 0
            if t % 1000 == 0 and self.verbose:
                print(str(t) +  " out of " + str(self.T) + " updates done: " + str(np.round(t/self.T * 100, 1)) + "%")
            for opt in self.optlist:
                if opt.domain.name == 'LACeL':
                    opt.LACeLupdate(self.data[t, :])
                weights = opt.getweights()
                oldw = opt.getweights()
                losslist[opt.getname()].append(self.loss(weights, self.data[t, :]))
                grad = self.gradient(weights, self.data[t, :], self.clip)
                #gradlist[opt.getname()].append(grad)
                opt.update(grad)
                #weightlist[opt.getname()].append(opt.getweights())
                self.optlist[i] = opt
                if verbose:
                    losstot[opt.getname()] += losslist[opt.getname()][t]
                i += 1
                if self.angle:
                    neww = opt.getweights()
                    if (np.sqrt(np.dot(oldw, oldw)) == 0) or (np.sqrt(np.dot(oldw, oldw)) == 0):
                        anglelist[opt.getname()].append(1)
                    else:
                        angle = np.arccos(np.dot(oldw, neww)/(np.sqrt(np.dot(oldw, oldw))*np.sqrt(np.dot(neww, neww))))
                        anglelist[opt.getname()].append(angle)
                if self.ed:
                    edlist[opt.getname()].append(opt.effectiveDim())
            self.counter += 1    
            if verbose:
                print("T = " + str(t + 1) + ". " + str(losstot))
                
        self.losslist = losslist
        #self.gradlist = gradlist
        #self.weightlist = weightlist
        self.anglelist = anglelist
        self.edlist = edlist
    
    def plotloss(self):
        plt.rc('lines', linewidth=4)
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'gray', 'chartreuse', 'forestgreen']) +
                           cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--'])))
        f, ax = plt.subplots()
        xaxis = np.array(list(range(self.T))) + 1
        for key, value in self.losslist.items():
            ax.plot(xaxis, np.cumsum(value), label = key)
        ax.legend()
        ax.set_xlabel("Round")
        ax.set_ylabel("Loss")
        plt.show()
        return(ax)
    
    def plotregret(self, comparitor):
        plt.rc('lines', linewidth=4)
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'gray', 'chartreuse', 'forestgreen']) +
                           cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--'])))
        #plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'm', 'k', 'c'])))
        f, ax = plt.subplots()
        xaxis = np.array(list(range(self.T))) + 1
        for key, value in self.losslist.items():
            ax.plot(xaxis, np.cumsum(value - comparitor), label = key)
        ax.legend()
        ax.set_xlabel("Round")
        ax.set_ylabel("Regret")
        plt.show()
        return(ax)
    
    def saveplotregret(self, comparitor, path, title = ""):
        plt.rc('lines', linewidth=4)
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'gray', 'chartreuse', 'forestgreen']) +
                           cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--'])))
        f, ax = plt.subplots()
        xaxis = np.array(list(range(self.T))) + 1
        for key, value in self.losslist.items():
            ax.plot(xaxis, np.cumsum(value - comparitor), label = key)
        ax.legend()
        ax.set_xlabel("Round")
        ax.set_ylabel("Regret")
        ax.set_title(title)
        f.savefig(path)
        plt.show()
        return(ax)
     
    def getweights(self, name):
        return(dict([(name, self.optlist[self.namnum[name]].getweights())]))
    
    def getallweights(self):
        return(dict([(opt.getname(), opt.getweights()) for opt in self.optlist]))
    
    def getloss(self, name):
        return(dict([(name, self.losslist[name])]))
    
    def getallloss(self):
        return(self.losslist)
    
    #def getweighthistory(self, name):
    #    return(dict([(name, self.weightlist[name])]))
    
    #def getallweighthistory(self):
    #    return(self.weightlist)
    
    def getnames(self):
        return(list(self.namnum.keys()))
    
    def sumloss(self):
        return({key: sum(self.losslist[key]) for key in self.losslist})
        
    def sumangle(self):
        if self.angle:
            return({key: sum(self.anglelist[key]) for key in self.anglelist})
        else:
            return("did not keep track of angles")
    
    def sumed(self):
        if self.angle:
            return({key: sum(self.edlist[key]) for key in self.edlist})
        else:
            return("did not keep track of effective dimension")
            
    def ploted(self):
        plt.rc('lines', linewidth=4)
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'gray', 'chartreuse', 'forestgreen']) +
                           cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--'])))
        #plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'm', 'k', 'c'])))
        f, ax = plt.subplots()
        xaxis = np.array(list(range(self.T))) + 1
        for key, value in self.edlist.items():
            ax.plot(xaxis, value, label = key)
        ax.legend()
        ax.set_xlabel("Round")
        ax.set_ylabel("Effective dimension")
        plt.show()
        return(ax)
    
    def plotcumangle(self):
        plt.rc('lines', linewidth=4)
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'gray', 'chartreuse', 'forestgreen']) +
                           cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--'])))
        #plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'm', 'k', 'c'])))
        f, ax = plt.subplots()
        xaxis = np.array(list(range(self.T))) + 1
        for key, value in self.anglelist.items():
            ax.plot(xaxis, np.cumsum(value), label = key)
        ax.legend()
        ax.set_xlabel("Round")
        ax.set_ylabel("cumulative angles")
        plt.show()
        return(ax)
            
    
    def savetableregret(self, comparator, path):
        ax = plt.subplot(111, frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        losses = self.sumloss()
        losses = pd.DataFrame(losses, index=[0])
        regrets = losses - np.sum(comparator)
        table(ax, regrets)  # where df is your data frame
        
        plt.savefig(path)
        
    def saveregretpickle(self, comparator, path, dataname = "", lossname = ""):
        try:
            lossesold = pd.read_pickle(path)
            lossold = True
        except:
            lossold = False
        regret = self.sumloss()
        regret = pd.DataFrame(regret, index=[0]) - np.sum(comparator)
        regret['data'] = dataname
        regret['loss'] = lossname
        if lossold:
            test = [dataname, lossname]
            if lossesold.query('@test[0] == data and @test[1] == loss').empty:
                lossesnew = pd.concat([lossesold, regret])
                lossesnew = lossesnew.sort_values(by = ["data", "loss"])
                pd.to_pickle(lossesnew, path)
            if not lossesold.query('@test[0] == data and @test[1] == loss').empty:
                lossesold = lossesold[(lossesold.data != dataname) | (lossesold.loss != lossname)]
                lossesnew = pd.concat([lossesold, regret])
                lossesnew = lossesnew.sort_values(by = ["data", "loss"])
                pd.to_pickle(lossesnew, path)
        else:
            lossesnew = regret
            lossesnew = lossesnew.sort_values(by = ["data", "loss"])
            pd.to_pickle(lossesnew, path)

    def printregret(self, comparator, dataname = "", lossname = ""):
        regret = self.sumloss()
        regret = pd.DataFrame(regret, index=[0]) - np.sum(comparator)
        regret['data'] = dataname
        regret['loss'] = lossname
        return(regret)
        
    def getallG(self):
        Glist = []
        allnames = self.getnames()
        for i in range(len(self.optlist)):
            try:
                Glist.append((allnames[i], self.optlist[i].G))
            except:
                Glist.append((allnames[i], "not available"))
        return(Glist)
        
        



        
        
        
        
        