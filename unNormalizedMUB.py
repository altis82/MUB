import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

rho=1
Q=5000
n=3  #number of offices
I=3  # number of DCs
omega_wat=[1,1,1]
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
    #omegaCf=0.4
    omegaCf=1
    k=2.0/7
    eReduct=np.zeros(n) # energy reduction of office i
    e=np.zeros(n)
    Ln=np.zeros(n)
    #heat loss
    R=10
    """__init__() functions as the class constructor"""
    def __init__(self, Tc=None,omegaCf=None):
        self.omegaCf=omegaCf
        self.G=[1.1, 1.2, 1.3, 1.4, 1.5]
        self.Kb=np.array([70.5, 261.5, 350.0, 201, 249])*3600/77 # 77  is the average temperature (K)
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

            if Tn[i]<0:
                self.Tn[i]=0
            else:
                self.Tn[i]=Tn[i]
    '''calculation of user comfort cost'''
    def userComfCost(self):

        for i in range (n):
            #Tmax=(3.0-self.G[i])/self.k
            if(self.Tn[i]<0):
                self.Tn[i]=0
                self.L[i]=3
            else:
                self.Ln[i]=self.k*self.Tn[i]

        return self.omegaCf*(sum(self.G)+ sum(self.Ln))

    def userComfCosti(self,i):
        self.userComfCost()
        return self.omegaCf*(self.G[i]+ self.Ln[i])



    """Energy reduction"""
    def enReduc(self):
        sumE=0
        for i in range (n):
            sumE=self.Kb[i]*(self.Ti[i]-self.Tc)
        return sumE

    def updateEn(self,sigman,en_):
        Kn=np.zeros(n)
        tempTn=np.zeros(n)
        #print(en_)
        for i in range(n):
            Tmax=(3-self.G[i])/self.k
            Kn[i]=self.Kb[i]/self.M
            self.e[i]=en_[i]+(sigman[i]/rho)-(self.omegaCf/rho)*(self.k/(Kn[i]))
            
            if self.e[i]<0:
                self.e[i]=0
            tempTn[i]=self.e[i]/Kn[i]
        #print('en',self.e)
        self.setTn(tempTn)
        #self.userComfCost()



#####################################
class DC:
    # number of tenant

    S=np.zeros(I) # number of servers for tenants
    lamb=np.zeros(I) # arrival rate for tenants
    mu=np.zeros(I) # service rate for tenats
    s=np.zeros(I) # number of turn-off servers for tenants
    D=1 # SLA threshold delay per second
    e=np.zeros(I)
    #omegadc=0.3
    omegasla=np.zeros(I)
    gamma=np.zeros(I)
    qi=200
    """__init__() functions as the class constructor"""
    def __init__(self,mu,S,omegadc,omegasla):
        # number of tenant

        self.D=1 # SLA threshold delay per second
        self.omegadc=omegadc
        self.gamma=np.array([200*1.2,200*1.3,200*1.4])
        self.mu=mu
        self.S=S
        self.qi=200
        self.omegasla=omegasla

    """ set number of turn-off servers"""
    def setSwitchSever(self,s):
        #self.s=s
        mins=self.minServer()
        #print(s)
        for i in range(I):
            if s[i]<0:
                self.s[i]=0
            else:
                self.s[i]=s[i]
    """ calculate energy reduction"""
    def enReduc(self):
        # power consumption of active server

        return qi*sum(self.s)

    """ set number of arrival rate lambda"""
    def setlamb(self,lamb):
        self.lamb=lamb
        #print(self.lamb)

    def minServer(self):
        mins=np.zeros(I)
        for i in range(I):
            mins[i]=self.lamb[i]/(self.mu[i]-1/self.D)
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
            if self.s[i]<0:
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
            sumC=sumC+(omega_wat[i]*self.s[i])+(self.omegasla[i])*dl[i]*self.lamb[i]
        return sumC*self.omegadc

    def dcCosti(self,i):

        mins=self.minServer()
        dl=self.delay()
        return (self.omegadc*(omega_wat[i]*self.s[i]+(self.omegasla[i])*dl[i]))

    def funcs(self,x,ei_,sigma,i):

        # tempi=np.zeros(I)
        # tempn=np.zeros(n)
        mins=self.minServer()
        #print(x,ei_,sigma,i)
        
        return x-(ei_+(sigma/rho)-(self.omegadc/rho)*(omega_wat[i]+(self.omegasla[i])*((self.lamb[i])**2)/(((self.S[i]-(x/self.gamma[i]))*self.mu[i]-self.lamb[i])**2))/self.gamma[i])



    def updateEi(self,sigmai,ei_):

        stemp=np.zeros(I)
        mins=self.minServer()
        maxe=np.zeros(I)
        for i in range(I):
            maxe[i]=(self.S[i]-mins[i])*self.gamma[i]
            temp=fsolve(self.funcs, [1], args=(ei_[i],sigmai[i],i,))
#            sigma=sigmai[i]            
#            tpmei_=ei_[i]
#            print('kq',(tpmei_+(sigma/rho)-(self.omegadc/rho)*(alpha[i]+(1-alpha[i])*((self.lamb[i])**2)/(((self.S[i]-(temp/self.gamma))*self.mu[i]-self.lamb[i])**2))/self.gamma))
#            print('x',temp)
            
            self.e[i]=temp
            if self.e[i]<0:
                self.e[i]=0
            else:
                if self.e[i]>maxe[i]:
                    self.e[i]=maxe[i]
                    print('Warning: reducing si lower than minimum active server!!!!')
            
            stemp[i]=self.e[i]/self.gamma[i]
        #print('ei',self.e)
        self.setSwitchSever(stemp)
#####################################
class bkGen:
    ez=0
    omegabg=0.3
    def __init__(self,omegabg=None):
        self.ez=20
        self.omegabg=omegabg
    def bgCost(self):

        return self.omegabg*self.ez

    def updateEz(self,sigmaz,ez_):

        self.ez=self.ez+sigmaz/rho-self.omegabg/rho
        if self.ez<0:
            self.ez=0
        # if self.ez>Q:
        #     self.ez=Q
        # else:
        #     if self.ez<0:
        #         self.ez=0
        #print(self.ez)

#####################################
class operator:

    sigmai=np.zeros(I)
    sigman=np.zeros(n)
    sigmaz=0
    e_i=np.zeros(I)
    e_n=np.zeros(n)
    e_z=0

    """__init__() functions as the class constructor"""
    def __init__(self,sigmai,sigman,sigmaz,e_i,e_n,e_z):
        self.sigmai=sigmai
        self.sigman=sigman
        self.e_i=e_i
        self.e_n=e_n
        self.e_z=e_z
        self.sigmaz=sigmaz

    def updateehat(self, ei,en,ez):
        v=fsolve(self.funcv, [1],args=(ei,en,ez))
        #print(ei,en,ez)
        #print(v)
        for i in range(I):
            self.e_i[i]=ei[i]-(v+self.sigmai[i])/rho
            if self.e_i[i]<0:
                self.e_i[i]=0.0

        for i in range(n):
            self.e_n[i]=en[i]-(v+self.sigman[i])/rho
            if self.e_n[i]<0:
                self.e_n[i]=0.0

        self.e_z=ez-(v+self.sigmaz)/rho
        if self.e_z<0:
            self.e_z=0
        #print(self.e_i,self.e_n,self.e_z)
    def updateSigma(self, ei,en,ez):

        for i in range(I):
            self.sigmai[i]=self.sigmai[i]+rho*(self.e_i[i]-ei[i])

        for i in range(n):
            self.sigman[i]=self.sigman[i]+rho*(self.e_n[i]-en[i])

        self.sigmaz=self.sigmaz+rho*(self.e_z-ez)
        #print(self.sigman,self.sigmai,self.sigmaz)

    def funcv(self,v,ei,en,ez):
        tempi=np.zeros(I)
        tempn=np.zeros(n)

        for i in range(I):
            tempi[i]=(v+self.sigmai[i])/rho
            
        for i in range(n):
            tempn[i]=(v+self.sigman[i])/rho
        

        res= (sum(ei)+sum(en)+ez)-(sum(tempi)+sum(tempn)+(v+self.sigmaz)/rho)-Q
        return res
        #return((sum(ei)+sum(en)+ez)-(v[0]*self.sigman[0])/rho-(v[1]*self.sigman[1])/rho)

    def calv(self):        
        x0 = fsolve(self.funcv, [1,1],args=([1,1,1],[1,1,1],1))
        return x0

################################################
def saveFile(filename,obj):
    import pickle
    output=open(filename, "wb" )
    pickle.dump(obj,output)
    output.close()
    
def loadFile(filename,obj):
    import pickle
    output=open(filename, "rb" )
    obj=pickle.load(output)    
    output.close()
    return obj
################################################