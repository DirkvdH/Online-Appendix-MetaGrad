
import numpy as np
import copy as cp
import domains as dommi
import scipy.sparse.linalg

#%%

class MetaGradL:
    # TODO: pass the baseslave _type_ instead of an instance
    def __init__(self, domain, slave, outcome = "last"):
        self.domain = domain
        self.baseslave = slave
        self.B = 0.
        self.dimension = domain.dimension
        self.Awake = np.array([])
        self.slaves = np.array([])
        self.center = np.repeat(0.0, self.dimension)
        self.lowersum = 0.
        self.G = 0.
        self.masterp = np.array([])
        self.outcome = outcome
        self.w = self.center
        self.Bstart = 0.
        self.Brestart = 0.
        self.factor = 0.
    
    def getname(self):
        return("MGL" + self.baseslave.shortname)
    
    def getetas(self):
        if len(self.slaves) > 0:
            return(np.array([slave.geteta() for slave in self.slaves]))
        else:
            return(np.array([]))
    def getijs(self):
        return(np.array([slave.geti() for slave in self.slaves]))
            
    def getweights(self):
        return(self.w)
    
    def predict(self, x = 0):
        ijs = self.getijs() # pronounce as i's
        # first we kill the old slaves
        if len(ijs) > 0:
            index = ijs <= -np.log2(2 * self.B)
            self.slaves = self.slaves[index]
            self.masterp = self.masterp[index]
        if self.B == 0:
            oldgrid = []
        else:
            oldgrid = np.arange(np.ceil(-np.log2(2 * (self.lowersum + self.B))), np.floor(-np.log2(2 * self.B)) + 1)
            oldgrid = np.flip(oldgrid)
        
        # now we create the new slaves
        indexnewslaves = np.invert(np.isin(oldgrid, ijs))    
        for i in range(len(oldgrid)):
            if indexnewslaves[i]:
                newslave = cp.deepcopy(self.baseslave)
                newslave.initialize(oldgrid[i], self.domain)
                self.slaves = np.append(self.slaves, newslave)
                self.masterp = np.append(self.masterp, 1.0)
                
        if len(self.slaves) == 0:
            self.w = self.center
            return(self.center)
        else:
            pred = self.center
            norm = 0.0
            for i in range(len(self.slaves)):
                slave = self.slaves[i]
                if self.domain.name == 'LACeL':
                    pred = pred + slave.geteta() * slave.predict(x) * self.masterp[i]
                else:
                    pred = pred + slave.geteta() * slave.w * self.masterp[i]
                norm = norm + slave.geteta() * self.masterp[i]
            self.w = pred/norm
            return(self.w)
            
            
    def LACeLupdate(self, datarow):
        if self.outcome == "last":
            xt = datarow[0:-1]
        if self.outcome == "first":
            xt = datarow[1:]
        if self.outcome == "none":
            xt = datarow
        self.predict(xt)
        self.xlast = xt
        
    def update(self, gradient):
        
        if self.B == 0:
            oldgrid = []
        else:
            oldgrid = np.arange(np.ceil(-np.log2(2 * (self.lowersum + self.B))), np.floor(-np.log2(2 * self.B)) + 1)
            oldgrid = np.flip(oldgrid)

        if self.domain.name == 'LACeL':
            G = 0
            # this is dangerous if there is no intercept!
            for i in range(len(self.xlast)):
                if self.xlast[i] != 0:
                    G = np.abs(gradient[i]/self.xlast[i])
                    break
            self.G = np.max([G, self.G])
            b = self.domain.C * G + np.abs(np.inner(self.w, gradient))
            Bt = np.max([b, self.B])
            if Bt == 0:
                self.factor = 0
            else:
                self.factor = self.B/Bt
            self.lowersum = self.lowersum + np.min([b, self.B])
            self.B = np.max([self.B, Bt])
        else:
            G = np.sqrt(np.dot(gradient, gradient))
            self.G = np.max([self.G, G])
            b = 2 * self.domain.radius * G
            Bt = np.max([b, self.B])
            if Bt == 0:
                self.factor = 0
            else:
                self.factor = self.B/Bt
            
            self.lowersum = self.lowersum + np.min([b, self.B])
            self.B = np.max([self.B, Bt])
        
        if self.B == 0:
            return(None)
        
        
        surloss = []
        clipgrad = self.factor * gradient
        for i in range(len(oldgrid)):
            iregeta = self.slaves[i].geteta() * np.dot(self.w - self.slaves[i].w, clipgrad).flatten()
            assert abs(iregeta) <= 1/2, "impending prod bound violation"
            surloss = np.append(surloss, - iregeta + iregeta ** 2)
            self.slaves[i].update(self.w, gradient)
            
        Cbefore = 0.0
        Cafter = 0.0
        if len(oldgrid) > 0:
            Cbefore = np.sum(self.masterp)
            m = np.min(surloss)
            self.masterp = self.masterp * np.exp(-surloss + m)
            Cafter = np.sum(self.masterp)           
            self.masterp = self.masterp * Cbefore / Cafter
                
        self.Brestart = self.Brestart + b/Bt
        if self.Bstart == 0:
            restart = True
        else:
            restart = Bt/self.Bstart > self.Brestart
        if restart:
            self.masterp = np.repeat(1.0, len(oldgrid))
            self.Bstart = Bt
        if self.domain.name != "LACeL":
            self.predict() 
            
        



#%%
            
class FullSlave:
    def __init__(self, D = "usedomainradius"):
        self.alpha = 1
        self.name = "FullSlave"
        self.shortname = "Full"
        self.D = D
        
    def initialize(self, i, domain):
        self.i = i
        self.domain = domain
        self.eta = 2.0**i
        if self.D == "usedomainradius":
            self.D = domain.radius
            if domain.dimension == 1:
                self.D = 1/3 * self.D # undo scaling by 3
        self.H = np.diag(np.repeat(np.float64(self.D**2), self.domain.dimension))
        self.w = domain.center(1).flatten()
        self.inverse = np.diag(np.repeat(np.float64(self.D**2), self.domain.dimension))
    
    def geteta(self):
        return(self.eta)
        
    def geti(self):
        return(self.i)
                
    def update(self, masterw, gradient):
        notM = np.dot(gradient, self.w - masterw)
        shiftgrad = (self.eta + 2*self.eta**2*notM)*gradient 
        ghat = np.sqrt(2) * self.eta * gradient
        q = np.dot(self.H, ghat)
        self.H = self.H - np.outer(q, q)/(1 + np.dot(q, ghat))
        wtilde = self.w - np.matmul(self.H, shiftgrad)
        if not self.domain.name == 'LACeL' and self.domain.dimension > 1:
            M = np.outer(gradient, gradient)
            self.inverse = self.inverse + 2 * self.eta**2 * M 
        if not self.domain.testdomain(wtilde):
            if self.domain.dimension == 1:
                iH = 1/self.H
            else:
                iH = self.inverse
            self.w = self.domain.project(wtilde, iH).reshape(self.domain.dimension)
        else:
            self.w = wtilde
    
    def predict(self, x):
        if self.domain.testdomainpred(self.w, x):
            return(self.w)
        else:
            self.w = self.domain.futureproject(self.w, self.H, x)
            return(self.w)
    
            
    

    
    
#%%

class FrequentSlave:
    def __init__(self, rank, D = "usedomainradius"):
        self.alpha = 1
        self.name = "FrequentSlave" + str(rank)
        self.shortname = "F" + str(rank)
        self.rank = rank
        self.D = D
        
    def initialize(self, i, domain):
        self.i = i
        self.domain = domain
        self.eta = 2.0**i
        if self.D == "usedomainradius":
            self.D = domain.radius
        self.H = np.diag(np.repeat(np.float64(self.D**2), 2*self.rank))
        self.H2 = np.zeros((self.rank, self.rank))
        self.w = domain.center(1).flatten()

        self.S = np.zeros((2*self.rank, self.domain.dimension))
        # S[self.rank+tau-1:2*self.rank, :] is all zero
        self.hessian = np.diag(np.repeat(np.float64(self.D**2), self.domain.dimension))
        self.tau = 0
    
    def geteta(self):
        return(self.eta)
        
    def geti(self):
        return(self.i)
                
    def update(self, masterw, gradient):
        notM = np.dot(gradient, self.w - masterw)
        shiftgrad = (self.eta + 2*self.eta**2*notM)*gradient 
        ghat = np.sqrt(2) * self.eta * gradient
        assert(np.all(self.S[self.rank+self.tau-1, :] == 0))
        self.S[self.rank+self.tau-1, :] = ghat

        if self.tau < self.rank:
            etau = np.zeros(2 * self.rank)
            etau[self.rank + self.tau - 1] = 1.0
            q = np.dot(self.S, ghat) - np.dot(ghat, ghat)/2 * etau
            # BONUS: write the below update to H in obviously symmetric form
            Hq = np.dot(self.H, q)
            eH = np.dot(etau, self.H)
            self.H = self.H - np.outer(Hq, eH)/(1.0 + np.dot(eH, q))
            He = np.matmul(self.H, etau)
            qH = np.matmul(q, self.H)
            self.H = self.H - np.outer(He, qH)/(1.0 + np.dot(qH, etau))
            self.tau = self.tau + 1
        else:
            if self.S.shape[1] > self.rank:
                [u,s,vt] = scipy.sparse.linalg.svds(self.S, self.rank, which='LM',return_singular_vectors='vh')
                ix = np.flip(np.argsort(s))
                U2 = s[ix]**2
                V = vt[ix,:]
            else:                
                # scipy.linalg.sparse.svds, cannot handle the case that you ask for
                # the maximum # of possible singular values.             
                [L, U, V] = np.linalg.svd(self.S, full_matrices=False)
                U2 = U[:self.rank]**2 # truncate by hand
                V = V[:self.rank,:] # truncate by hand
                if len(U2) < self.rank:
                    # dimension is less than rank, so padd U2 and V with zeros
                    # (Since we keep the top rank-1 eigenvalues, it is reasonable
                    # to set rank = dimension + 1, so this CAN happen.)
                    V = np.append(V, np.zeros((self.rank-len(U2),V.shape[1])),axis=0)
                    U2 = np.append(U2, np.zeros(self.rank-len(U2)))
                    
            self.S[:self.rank,:] = np.matmul(np.diag(np.sqrt(U2 - U2[self.rank - 1])), V)
            self.S[self.rank:,:] = 0
            self.H = np.diag(1/np.append(1/(self.D**2) + (U2 - U2[self.rank-1]), np.repeat(1/self.D**2, self.rank)))

            self.tau = 0


        SHS = np.matmul(np.matmul(self.S.transpose(), self.H), self.S)
        self.hessian = self.D**2*(np.diag(np.repeat(1, self.domain.dimension)) - SHS)
        wtilde = self.w - np.matmul(self.hessian, shiftgrad)
        
        if not self.domain.testdomain(wtilde):
            iH = self.iH0 + np.matmul(self.S.transpose(), self.S)
            self.w = self.domain.project(wtilde, iH).reshape(self.domain.dimension)
        else:
            self.w = wtilde
    
    def predict(self, x):
        if self.domain.testdomainpred(self.w, x):
            return(self.w)
        else:
            self.w = self.domain.futureproject(self.w, self.hessian, x)
            return(self.w)


#%%
class DiagGradL:
    def __init__(self, domain, outcome = "last", D = "usedomainradius"):
        self.domain = domain
        baseslave = FullSlave(D)
        DiagDom = dommi.LinfBall(domain.radius, 1, diagonal = True)
        baseMG = MetaGradL(domain = DiagDom, slave = baseslave)
        self.dimMGs = [cp.deepcopy(baseMG) for i in range(domain.dimension)]
        self.dimension = domain.dimension
        self.outcome = outcome
        self.center = np.repeat(0.0, self.dimension)
        self.w = self.center
        self.LACeLw = self.center
        self.insideLACeL = True
    
    def getname(self):
        return("MGLDiag")
    
            
    def getweights(self):
        if self.domain.name == "LACeL":
            return(self.LACeLw)
        return(self.w)
    
    def predict(self, x = [0]):
        w = np.array([])
        if len(x) < self.dimension:
            x = self.center
        if self.domain.name == "LACeL":
            self.LACeLupdate(x, "none")
            return(self.LACeLw)
        else:
            w = np.array([dimMG.predict([0]) for dimMG in self.dimMGs])
            self.w = w
            return(self.w)
            
    def LACeLupdate(self, datarow):
        if self.outcome == "last":
            xt = datarow[0:-1]
        if self.outcome == "first":
            xt = datarow[1:]
        if self.outcome == "none":
            xt = datarow
        w = np.array([dimMG.predict([0]) for dimMG in self.dimMGs])
        self.w = w
        self.insideLACeL = self.domain.testdomainpred(self.w, xt)
        if not self.insideLACeL:
            self.LACeLw = self.domain.futureproject(self.w, np.diag(np.repeat(1, self.dimension)), xt)
        else:
            self.LACeLw = self.w
        
        
    def update(self, gradient):
        if self.domain.name == 'LACeL':
            if not self.insideLACeL:
                gnorm = np.sqrt(np.inner(gradient, gradient))
                dif = self.w - self.LACeLw
                wnorm = np.sqrt(np.inner(dif, dif))
                addgrad = gnorm * dif/wnorm
                for i in range(self.dimension):
                    gradi = 0.5 * gradient[i] + 0.5 * addgrad[i]
                    self.dimMGs[i].update(gradi)
            else:
                for i in range(self.dimension):
                    self.dimMGs[i].update(0.5 * gradient[i])
        else:
            for i in range(self.dimension):
                self.dimMGs[i].update(gradient[i])
            
        if self.domain.name != "LACeL":
            self.predict() 
            
