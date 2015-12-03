# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 09:19:09 2015

@author: Chuan Pham
"""
import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

rho=0.3
alpha=0.01
Q=5000
n=3  #number of offices
I=3  # number of DCs
class HVAC:

    G=np.zeros(n) # heat gain for n office
    K=np.zeros(n) # conductive of n office
    M=0.14 # energy transform
    Tn=np.zeros(n) # temperature indoor of n office
    Tc=0 # temperature outdoor
    #constant related to comfort temp
    a2=17.8
    #correlation between indoor and ourdoor
    b2=0.31
    omegaCf=0.4
    k=2.0/7
    eReduct=np.zeros(n) # energy reduction of office i
    e=np.zeros(n)
    Ln=np.zeros(n)
    #heat loss
    R=10
    """__init__() functions as the class constructor"""
    def __init__(self, Tc=None):

        self.G=[1.1, 1.2, 1.3, 1.4, 1.5]
        self.K=[70.5, 118.5, 162, 201, 249]
        self.M=0.14
        self.Tc=Tc
        self.e=np.zeros(n)
        self.Tn=np.zeros(n)
        self.Ln=np.zeros(n)
        self.k=2.0/7
    """Tc(To) function"""
    def comfTemp(self):
        return self.a2+self.b2*self.Tc

    '''set temperature Tn'''
    def setTn(self,Tn):

        #print(Tn)
        for i in range(n):

            Tmax=(3.0-self.G[i])/self.k
            if Tn[i]<0:
                self.Tn[i]=0
            else:
                if Tn[i]>Tmax:
                    self.Tn[i]=Tmax
                else:
                    self.Tn=Tn
        #print(self.Tn)
    '''calculation of user comfort cost'''
    def userComfCost(self):

        #L=np.zeros(n)
        #tempTn=np.zeros(n)
        for i in range (n):
            Tmax=(3.0-self.G[i])/self.k
            if(self.Tn[i]>self.R):
                self.L[i]=3
            else:
                self.Ln[i]=2.0/7*self.Tn[i]
            #print(self.Tn[i])
            #tempTn[i]=self.Ln[i]/self.k
        #self.setTn(tempTn)
        #print(self.Ln)
        return self.omegaCf*(sum(self.G)+ sum(self.Ln))/3

    def userComfCosti(self,i):
        self.userComfCost()
        return self.omegaCf*(self.G[i]+ self.Ln[i])/3



    """Energy reduction"""
    def enReduc(self):
        sumE=0
        for i in range (n):
            sumE=self.K[i]/self.M*(self.Ti[i]-self.Tc)
        return sumE*3600*0.0027

    def updateEn(self,sigman,en_):
        Kn=np.zeros(n)
        tempTn=np.zeros(n)
        #print(en_)
        for i in range(n):
            Tmax=(3-self.G[i])/self.k
            Kn[i]=self.K[i]/self.M
            self.e[i]=en_[i]+sigman[i]/rho-self.omegaCf/rho*self.k/(3*Kn[i])
            if self.e[i]>Kn[i]*Tmax:
                self.e[i]=Kn[i]*Tmax
            else:
                if self.e[i]<0:
                    self.e[i]=0
            #set Ti
            tempTn[i]=self.e[i]/Kn[i]

        self.setTn(tempTn)
        #self.userComfCost()



#####################################
class DC:
    # number of tenant
    I=3
    S=np.zeros(I) # number of servers for tenants
    lamb=np.zeros(I) # arrival rate for tenants
    mu=np.zeros(I) # service rate for tenats
    s=np.zeros(I) # number of turn-off servers for tenants
    D=1 # SLA threshold delay per second
    e=np.zeros(I)
    omegadc=0.3
    gamma=200*0.1
    qi=200
    """__init__() functions as the class constructor"""
    def __init__(self,mu,S):
        # number of tenant

        self.D=1 # SLA threshold delay per second
        self.omegadc=0.3
        self.gamma=200*0.1
        self.mu=mu
        self.S=S
        self.qi=200
    """ set number of turn-off servers"""
    def setSwitchSever(self,s):
        self.s=s
        mins=self.minServer()

        for i in range(I):
            if s[i]<0:
                self.s[i]=mins[i]
            else:
                if s[i]>self.S[i]:
                    self.s[i]=self.S[i]
                else:
                    self.s[i]=s[i]
    """ calculate energy reduction"""
    def enReduc(self):
        # power consumption of active server

        return qi*self.s

    """ set number of arrival rate lambda"""
    def setlamb(self,lamb):
        self.lamb=lamb
        #print(self.lamb)

    def minServer(self):
        mins=np.zeros(I)
        for i in range(I):
            mins[i]=self.lamb[i]/(self.mu[i]-1/self.D)
            #print(np.round(mins[i]))
        #print(mins)
        return mins
    def delay(self):
        dl=np.zeros(I)
        for i in range(I):
            dl[i]= (1.0/(self.mu[i]-(self.lamb[i]/(self.S[i]-self.s[i]))))
        return dl
    """ calculte SLA cost (delay)"""


    def slaCost(self):
        sumDelay=0.0
        mins=self.minServer()
        for i in range(I):
            if self.s[i]+mins[i]>self.S[i]:
                return False
            else:
                sumDelay=sumDelay + (1.0/(self.mu[i]-(self.lamb[i]/(self.S[i]-self.s[i]))))
                return sumDelay

    """ calculate DC cost"""
    def dcCost(self):
        sumC=0
        mins=self.minServer()
        dl=self.delay()
        for i in range(I):
            sumC=sumC+alpha*self.s[i]/(self.S[i]-mins[i])+(1-alpha)*dl[i]/self.D
        return sumC*self.omegadc
    def dcCosti(self,i):

        mins=self.minServer()
        dl=self.delay()
        return (self.omegadc*(alpha*self.s[i]/(self.S[i]-mins[i])+(1-alpha)*dl[i]/self.D))

    def funcs(self,x,ei_,sigma,i):

        # tempi=np.zeros(I)
        # tempn=np.zeros(n)


        mins=self.minServer()

        return x-(ei_+sigma/rho-self.omegadc/rho*(alpha/mins[i]+(1-alpha)/self.D*(self.lamb[i])/(((self.S[i]-x/self.gamma)*self.mu[i]-self.lamb[i])**2))/self.gamma)

    # def solx(self):
    #     print(self.funcs)
    #     x0 = fsolve(self.funcs, [1], args=(1,2,1,))
    #     print x0


    def updateEi(self,sigmai,ei_):



        mins=self.minServer()
        eimax=np.zeros(I)
        ctemp=np.zeros(I)
        stemp=np.zeros(I)
        for i in range(I):
            eimax[i]=self.gamma*(self.S[i]-mins[i])
            self.e[i]=fsolve(self.funcs, [1], args=(ei_[i],sigmai[i],i,))

            if self.e[i]>eimax[i]:
                self.e[i]=eimax[i]
            else:
                if self.e[i]<0:
                    self.e[i]=0
            stemp[i]=self.e[i]/self.gamma

        self.setSwitchSever(stemp)
#####################################
class bkGen:
    ez=0
    omegabg=0.3
    def __init__(self):
        self.ez=0
        self.omegabg=0.3
    def bgCost(self):

        return self.omegabg*self.ez/Q
    def updateEz(self,sigmaz,ez_):

        self.ez=self.ez+sigmaz/rho-self.omegabg/(rho*Q)
        if self.ez>Q:
            self.ez=Q
        else:
            if self.ez<0:
                self.ez=0

#####################################
class operator:

    sigmai=np.zeros(I)
    sigman=np.zeros(n)
    sigmaz=0
    e_i=np.zeros(I)
    e_n=np.zeros(n)
    e_z=0

    """__init__() functions as the class constructor"""
    def __init__(self,sigmai,sigman,e_i,e_n,e_z):
        self.sigmai=sigmai
        self.sigman=sigman
        self.e_i=e_i
        self.e_n=e_n
        self.e_z=e_z

    def updateehat(self, ei,en,ez):
        v=fsolve(self.funcv, [1],args=(ei,en,ez))

        for i in range(I):
            self.e_i[i]=ei[i]-(v+self.sigmai[i])/rho
            if self.e_i[i]<0:
                self.e_i[i]=0.0

        for i in range(n):
            self.e_n[i]=en[i]-(v+self.sigman[i])/rho
            if self.e_n[i]<0:
                self.e_n[i]=0.0

        self.e_z=ez-(v+self.sigmaz)/rho

    def updateSigma(self, ei,en,ez):

        for i in range(I):
            self.sigmai[i]=self.sigmai[i]+rho*(self.e_i[i]-ei[i])

        for i in range(n):
            self.sigman[i]=self.sigman[i]+rho*(self.e_i[i]-en[i])

        self.sigmaz=self.sigmaz+rho*(self.e_z-ez)


    def funcv(self,v,ei,en,ez):
        tempi=np.zeros(I)
        tempn=np.zeros(n)

        for i in range(I):
            tempi[i]=(v+self.sigmai[i])/rho
        for i in range(n):
            tempn[i]=(v+self.sigman[i])/rho
        #print ('result',(sum(ei)+sum(en)+ez))
        return (sum(ei)+sum(en)+ez)-(sum(tempi)+sum(tempn)+(v+self.sigmaz)/rho)-Q

    def calv(self):
        x0 = fsolve(self.funcv, [1])
        return x0
#####################################
#Tc=25




