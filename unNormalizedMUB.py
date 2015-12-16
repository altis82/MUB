import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import cvxpy as cvx
import string
import pickle
omega=np.array([0.5,0.01/5,0.3])
rho=1
Q=0
n=3  #number of offices
I=3  # number of DCs
omega_wat=np.array([0.5,0.5,0.5])
#sla weight
omega_sla=np.array([0.5,0.5,0.5])
runTime = 100
slv=cvx.SCS
class HVAC:

    G=np.zeros(n) # heat gain for n office
    #K=np.zeros(n) # conductive of n office
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
        self.G=[1.7, 1.2, 1.2]
        self.Kb=np.array([300.5,200.5,200.5])/1000 # 77  is the average temperature (K)
        self.M=0.05
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
            if(self.Tn[i]>6.50):
                self.Tn[i]=6.5
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
            sumE=self.Kb[i]/self.M*(self.Ti[i]-self.Tc)
        return sumE

    def updateEn2(self,sigman,en_):
        sigman=np.array(sigman)
        en_=np.array(en_)
        en=cvx.Variable(n)
        T=cvx.Variable(n)
        constraints=[0<=T]

        comfCost=self.omegaCf*cvx.sum_entries(np.array(self.G)+self.k*T)
        #print (comfCost)
        for i in range(n):
            constraints+=[en[i]==(self.Kb[i]/self.M)*T[i]]
        objective = cvx.Maximize(np.transpose(sigman)*en-comfCost-rho/2*cvx.sum_squares(en-np.array(en_)))
        prob = cvx.Problem(objective, constraints)
        result = prob.solve(solver=slv)


        #print ("obj",result)
        self.e=en.value.A1
        #print("En",self.e)

    def updateEn(self,sigman,en_):
        Kn=np.zeros(n)
        tempTn=np.zeros(n)
        print("input",sigman,en_)
        if self.omegaCf==0:
            self.e=[0,0,0]
        else:
            for i in range(n):
                Tmax=(3-self.G[i])/self.k
                Kn[i]=self.Kb[i]/(self.M)
                self.e[i]=en_[i]+(sigman[i]/rho)-(self.omegaCf/rho)*(self.k/(Kn[i]))
                
                if self.e[i]<0:
                    self.e[i]=0
                if self.e[i]>(6.5*Kn[i]):
                    self.e[i]=6.5*Kn[i]
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
    omegawat=np.zeros(I)
    gamma=np.zeros(I)
    qi=200
    """__init__() functions as the class constructor"""
    def __init__(self,mu,S,omegadc,omegasla,omegawat):
        # number of tenant

        self.D=1 # SLA threshold delay per second
        self.omegadc=omegadc
        self.gamma=np.array([0.200*1.2,0.200*1.2,0.200*1.2])
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

        return qi*sum(self.s)/1000.0

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


    def updateEi2(self,sigmai,ei_):
        sigmai=np.array(sigmai)
        ei_=np.array(ei_)
        ei=cvx.Variable(I)
        si=cvx.Variable(I)
        constraints=[0<=si]
        slacost=cvx.sum_entries(cvx.mul_elemwise(self.lamb,cvx.inv_pos(self.mu-cvx.mul_elemwise(self.lamb,cvx.inv_pos(self.S-si)))))
        watcost=np.array(self.omegawat)*si
        for i in range(I):
            constraints+=[ei[i]==self.gamma[i]*si[i]]
        objective = cvx.Minimize(-np.transpose(sigmai)*ei+self.omegadc*(slacost+watcost)+rho/2*cvx.sum_squares(ei-np.array(ei_)))
        prob = cvx.Problem(objective, constraints)
        result = prob.solve(solver=slv)

        #print ei.value.A1
        #print result
        self.e=ei.value.A1
        mins=self.minServer()
        maxe=np.zeros(I)
        for i in range(I):
            maxe[i]=(self.S[i]-mins[i])*self.gamma[i]
            if self.e[i]<0:
                self.e[i]==0
            else:
                if self.e[i]>maxe[i]:
                    self.e[i]=maxe[i]
                    print('\nWarning: reducing si lower than minimum active server!!!!')
        #print("ei:",self.e)

        self.setSwitchSever(self.e)
    def updateEi(self,sigmai,ei_):

        stemp=np.zeros(I)
        mins=self.minServer()
        maxe=np.zeros(I)
        for i in range(I):
            maxe[i]=(self.S[i]-mins[i])*self.gamma[i]
            temp=fsolve(self.funcs, [self.e[i]], args=(ei_[i],sigmai[i],i,))
#            sigma=sigmai[i]            
#            tpmei_=ei_[i]
#            print('kq',(tpmei_+(sigma/rho)-(self.omegadc/rho)*(alpha[i]+(1-alpha[i])*((self.lamb[i])**2)/(((self.S[i]-(temp/self.gamma))*self.mu[i]-self.lamb[i])**2))/self.gamma))
            #print('x',temp)
            
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
        self.ez=0
        self.omegabg=omegabg
    def bgCost(self):

        return self.omegabg*self.ez

    def updateEz2(self,sigmaz,ez_):

        ez=cvx.Variable()
        constraints=[0<=ez]
        objective = cvx.Maximize(np.array(sigmaz)*ez-ez*self.omegabg-rho/2*cvx.square(ez-ez_))
        prob = cvx.Problem(objective, constraints)
        result = prob.solve(solver=slv)
        #print(ez.value)
        self.ez=ez.value

    def updateEz(self,sigmaz,ez_):

        self.ez=ez_+sigmaz/rho-self.omegabg/rho
        if self.ez<0:
            self.ez=0
        print("ez",self.ez)

#####################################
class operator:
    
    sigmai=np.zeros(I)
    sigman=np.zeros(n)
    sigmaz=0
    e_i=np.zeros(I)
    e_n=np.zeros(n)
    e_z=0
    tempen=np.zeros(n)
    tempei=np.zeros(I)    
    tempez=0
    omega=[0,0,0]
    setbaseline=0
    """__init__() functions as the class constructor"""
    def __init__(self,sigmai,sigman,sigmaz,e_i,e_n,e_z,omega,baseline):
        self.sigmai=sigmai
        self.sigman=sigman
        self.e_i=e_i
        self.e_n=e_n
        self.e_z=e_z
        self.sigmaz=sigmaz
        self.tempen=np.zeros(n)
        self.tempei=np.zeros(I)    
        self.tempez=0
        self.omega=omega
        self.setbaseline=baseline
        if self.setbaseline==1:
            self.sigman=[0,0,0]
            self.e_n=[0,0,0]
        if self.setbaseline==2:
            self.sigmai=[0,0,0]
            self.e_i=[0,0,0]
    def funcv(self,v,ei,en,ez):
        tempi=np.zeros(I)
        tempn=np.zeros(n)

        for i in range(I):
            if self.omega[1]==0:
                tempi[i]=0
                ei[i]=0
            else:
                tempi[i]=(v+self.sigmai[i])/rho

        for i in range(n):
            if self.omega[0]==0:
                tempn[i]==0
                en[i]=0
            else:
                tempn[i]=(v+self.sigman[i])/rho

        res= (sum(ei)+sum(en)+ez)-(sum(tempi)+sum(tempn)+(sum(v)+self.sigmaz)/rho)-Q
        return res

    def updateehat2(self,ei,en,ez):
        ehat =cvx.Variable(n+I+1)
        sigmatmp=np.zeros(n+I+1)
        etmp=np.zeros(n+I+1)
        for i in range(n):
            sigmatmp[i]=self.sigman[i]
            etmp[i]=en[i]
            #etmp[i]=ehat[i]-en[i]
        for i in range(I):
            sigmatmp[i+n]=self.sigmai[i]
            etmp[i+n]=ei[i]
        sigmatmp[I+n]=self.sigmaz
        etmp[n+I]=ez
        #print("sigma",sigmatmp)
        print("e",etmp)
        objective = cvx.Minimize(sigmatmp*ehat+rho/2*cvx.sum_squares((ehat-etmp)))
        constraints=[sum(ehat)==Q,0<=ehat]

        if self.setbaseline==1:
            for i in range(n):
                constraints+=[ehat[i]==0]
        else:
            if self.setbaseline==2:
                for i in range(I):
                    constraints+=[ehat[n+i]==0]
        prob = cvx.Problem(objective, constraints)
        #print(prob.status)
        result = prob.solve(solver=cvx.SCS)

        #print(prob.status)
        #sum(ehat.value.A1)
        #print result


        for i in range(0,(n+I+1)):
            if i<n:
                self.e_n[i]=ehat.value.A1[i]
                if self.e_n[i]<0:
                    self.e_n[i]=0
            if i<I+n and i>=n:

                self.e_i[i-n]=ehat.value.A1[i]
                if self.e_i[i-n]<0:
                    self.e_i[i-n]=0

            if i==n+I:
                self.e_z=ehat.value.A1[i]
                if self.e_z<0:
                    self.e_z=0
        #print(self.e_n,self.e_i,self.e_z)

    def updateehat(self, ei,en,ez):
        v=fsolve(self.funcv, [1],args=(ei,en,ez))
        #print(v)
        
        for i in range(I):
            self.e_i[i]=ei[i]-(v+self.sigmai[i])/rho
            if self.e_i[i]<0:
                self.e_i[i]=0.0
        #print(self.e_i)
        for i in range(n):
            self.e_n[i]=en[i]-(v+self.sigman[i])/rho
            if self.e_n[i]<0:
                self.e_n[i]=0.0
        #print(self.e_n)
        self.e_z=ez-(v+self.sigmaz)/rho
        if self.e_z<0:
            self.e_z=0
        #print(self.e_z)
        #print(sum(self.e_i)+sum(self.e_n)+self.e_z)

    def updateSigma(self, ei,en,ez):
        if self.omega[1]==0:
            self.sigmai=[0,0,0]
        else:
            for i in range(I):
                self.sigmai[i]=self.sigmai[i]+rho*(self.e_i[i]-ei[i])
        if self.omega[0]==0:
            self.sigman=[0,0,0]
        else:
            for i in range(n):
                self.sigman[i]=self.sigman[i]+rho*(self.e_n[i]-en[i])

        self.sigmaz=self.sigmaz+rho*(self.e_z-ez)
        #print(self.sigman,self.sigmai,self.sigmaz)


def MUB(lamb,baseline,hvac,dc,bk,mub):
    # hvac= HVAC(25,omega[0])
    # Tnhat=[2.0,4.0,5.0,2.0,5.0]
    # hvac.setTn(Tnhat)
    # bk=bkGen(omega[2])
    # dc =DC([2.5,4.,4.],[2000,2000,2000],omega[1],omega_sla,omega_wat)
    #
    # mub=operator([1,2,3],[1,2,3],100,[1,2,3],[1,2,3],200,omega,baseline)
    dc.setlamb([lamb[0],lamb[1],lamb[2]])
    mub.setbaseline=baseline

    bk_totalcost= np.zeros(runTime)
    T=np.ones((3,runTime))
    s=np.ones((3,runTime))
    sigma = np.ones((7,runTime))
    Totalcost=np.zeros(runTime)
    bk_totalcost= np.zeros(runTime)
    energy = np.ones((7,runTime))
    cost = np.ones((7,runTime))
    Tnhat1=np.zeros(runTime)
    mub.setbaseline=baseline

    for i in range(runTime):

        hvac.updateEn2(mub.sigman,mub.e_n)
        #    def updateEi(self,sigmai,ei_):
        if baseline==1:
            hvac.e=[0,0,0]
        energy[0][i] = hvac.e[0]
        energy[1][i] = hvac.e[1]
        energy[2][i] = hvac.e[2]
        T[0][i]=hvac.Tn[0]
        T[1][i]=hvac.Tn[1]
        T[2][i]=hvac.Tn[2]
        #def updateEi(self,sigmai,ei_):
        #dc.updateEi(mub.sigmai,mub.e_i)
        dc.updateEi2(mub.sigmai,mub.e_i)
        if baseline==2:
            dc.e=[0,0,0]
        s[0][i]=dc.s[0]
        s[1][i]=dc.s[1]
        s[2][i]=dc.s[2]

        energy[3][i] = dc.e[0]
        energy[4][i] = dc.e[1]
        energy[5][i] = dc.e[2]


        #    def updateEz(self,sigmaz,ez_):
        #bk.updateEz(mub.sigmaz,mub.e_z)
        bk.updateEz2(mub.sigmaz,mub.e_z)
        energy[6][i] = bk.ez
        #def updateehat(self, ei,en,ez):
        #mub.updateehat(dc.e,hvac.e,bk.ez)
        # def updateehat2(self,ei,en,ez):

        mub.updateehat2(dc.e,hvac.e,bk.ez)

        #  def updateSigma(self, ei,en,ez):
        mub.updateSigma(dc.e,hvac.e,bk.ez)

        sigma[0][i] = mub.sigman[0]
        sigma[1][i] = mub.sigman[1]
        sigma[2][i] = mub.sigman[2]

        sigma[3][i] = mub.sigmai[0]
        sigma[4][i] = mub.sigmai[1]
        sigma[5][i] = mub.sigmai[2]
        sigma[6][i] = mub.sigmaz

        cost[0][i]=hvac.userComfCosti(0)
        cost[1][i]=hvac.userComfCosti(1)
        cost[2][i]=hvac.userComfCosti(2)
        cost[3][i]=dc.dcCosti(0)
        cost[4][i]=dc.dcCosti(1)
        cost[5][i]=dc.dcCosti(2)
        cost[6][i]=bk.bgCost()
        Totalcost[i]=cost[0][i]+cost[1][i]+cost[2][i]+cost[3][i]+cost[4][i]+cost[5][i]+cost[6][i]
        #print("total",Totalcost[i])
    return hvac,dc,bk,energy,cost,Totalcost
################################################
def saveFile(filename,obj):

    output=open(filename, "wb" )
    pickle.dump(obj,output)
    output.close()
    
def loadFile(filename,obj):

    output=open(filename, "rb" )
    obj=pickle.load(output)    
    output.close()
    return obj
################################################
def trace_lamb():
    maxtimeslot=60
    base_lamb=1500
    trace=np.zeros(maxtimeslot)
    with open('facebook.csv', 'r') as csvfile:
        i=0
        for row in csvfile:
            if i>=maxtimeslot:
                break
            trace[i]=string.atof(row)
            i=i+1

    #normalize trace
    trace=np.array(trace)/np.max(trace)
    #create lambda for DC
    lamb=np.ones((3,maxtimeslot/3))
    for i in range(maxtimeslot/3):
        lamb[0][i]=trace[i]*base_lamb
        lamb[1][i]=trace[i+10]*base_lamb
        lamb[2][i]=trace[i+20]*base_lamb
    return lamb
#hvac=HVAC(25,1)
#T=np.zeros(6)
#
#s=np.zeros(6)
#for i in range(0,6):
#    hvac.setTn([i,i,i])
#    s[i]=hvac.userComfCost()
#
#
#dc =DC([2.5,4.5,6.5,3.5, 4.5],[2000,2000,2000,2000,2000],0.01,[0.5,0.5,0.5])
#dc.setlamb([1100.0,1100.0,1100.0,1500.0,1600.0])
#si=np.zeros(6)
#bk=np.zeros(6)
#for i in range(0,6):
#    dc.setSwitchSever([i*100,i*100,i*100])
#    si[i]=dc.dcCost()
#    bk[i]=0.3*i*10
#plt.plot(si,label='dc')
#plt.plot(s,label='hvac')
#plt.plot(bk,label='bk')
#plt.legend(loc=1)
#plt.show()