
from copy import deepcopy
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
import cvxpy as cvx



def hinge(weights, datarow):
    loss = np.max([0, 1 - datarow[-1] * (np.dot(datarow[0:-1], weights))])
    return(loss)

def gradhinge(weights, datarow, clip = 10e30):
    if (datarow[-1] * (np.dot(datarow[0:-1], weights)) >= 1):
        grad = np.zeros(len(weights))
    else:
        grad = -datarow[-1] * datarow[0:-1]
    return(grad)

def Gboundhinge(ybound, xbound, wbound):
    return(ybound * xbound)
      
def logistic(weights, datarow):
    s = -datarow[-1] * (np.dot(datarow[0:-1], weights))
    if s < 0:
        loss = np.log(1 + np.exp(s))
    else:
        loss = s + np.log(1 + np.exp(-s))
    return(loss)

def gradlogistic(weights, datarow, clip = 10e30):
    s = datarow[-1] * (np.dot(datarow[0:-1], weights))
    if(s < 0):
        grad = -datarow[-1] * datarow[0:-1] /(1 + np.exp(s))    
    else:
        grad = -datarow[-1] * datarow[0:-1] * np.exp(-s)/(1 + np.exp(-s))
    return(grad)

def Gboundlogistic(ybound, xbound, wbound, clip = 10e30):
    return (ybound * xbound * 1/(1 + np.exp(-ybound * wbound * xbound)))

def squaredloss(weights, datarow):
    loss = (np.dot(datarow[0:-1], weights) - datarow[-1])**2
    return(loss)

def gradsquaredloss(weights, datarow, clip = 10e30):
    grad = 2*datarow[0:-1]*(np.dot(datarow[0:-1], weights) - datarow[-1])
    return(grad)

def Gboundsquaredloss(ybound, xbound, wbound):
    return(2 * xbound * (xbound*wbound + ybound))

def absoluteloss(weights, datarow):
    loss = np.abs(np.dot(datarow[0:-1], weights) - datarow[-1])
    return(loss)

def gradabsoluteloss(weights, datarow, clip = 10e30):
    #loss = np.abs(np.inner(datarow[0:-1], weights) - datarow[-1])
    grad = datarow[0:-1]*np.sign(np.dot(datarow[0:-1], weights) - datarow[-1])#(np.inner(datarow[0:-1], weights) - datarow[-1])/loss
    return(grad)

def Gboundabsoluteloss(ybound, xbound, wbound):
    return(xbound)

#%%

        
        
        
        
        
        
        
        
        
        
        
        