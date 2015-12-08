import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

#import unNormalizedMUB as umub
import  unNormalizedMUB as nmub
reload(nmub)

#omega(hvac,dc,bk)
omega=np.array([0.140*0.1,1.5*0.3,0.3*200]) #0.3$/KWh
#omega=np.array([0,1.0, 0.3*1e-3]) #set baseline1
#wear and tear cost
omega_wat=np.array([0.5,0.5,0.5])
#sla weight
omega_sla=np.array([0.5,0.5,0.5])


hvac= nmub.HVAC(25,omega[0])
Tnhat=[2.0,4.0,5.0,2.0,5.0]
hvac.setTn(Tnhat)
bk=nmub.bkGen(omega[2])
rho=50

Q=300 #Wh ( 1MWh)
n=3 #number of offices
I=3  # number of DCs

nmub.omega_wat=omega_wat
nmub.Q=Q
nmub.n=n
nmub.I=I
nmub.rho=rho
#hvac.userComfCost()
#hvac.enReduc()
#hvac.optimizeOfficei(1,5)
#initialize the datacenter with services rate of each tenant per s and max server
#nmub.DC(mu,S,omega)
dc =nmub.DC([2.5,2.5,2.5,3.5, 4.5],[2000,2000,2000,2000,2000],omega[1],omega_sla)

#arrival rate
dc.setlamb([1100.0,1100.0,1100.0,1500.0,1600.0])
#dc.setSwitchSever([200,100,100,200,100])
#dc.minServer()
#dc.solx()
#def __init__(self,sigmai,sigman,e_i,e_n,e_z):
mub=nmub.operator([1,2,3,4,5],[1,2,3,4,5],100,[1,2,3,4,5],[1,2,3,4,5],200,omega)
#mub.calv()
#    def updateEn(self,sigman,en_):
runTime = 30
#
#
#
#
bk_totalcost= np.zeros(runTime)
T=np.ones((3,runTime))
s=np.ones((3,runTime))
energy = np.ones((7,runTime))
sigma = np.ones((7,runTime))
cost = np.ones((7,runTime))

Tnhat1=np.zeros(runTime)
Totalcost=np.zeros(runTime)

for i in range(runTime):
    pre=sum(mub.sigmai)+sum(mub.sigman)+mub.sigmaz
    #print('pre',pre)
    hvac.updateEn(mub.sigman,mub.e_n)
    #    def updateEi(self,sigmai,ei_):
    #print ("Ti",hvac.Tn)
    energy[0][i] = hvac.e[0]
    energy[1][i] = hvac.e[1]
    energy[2][i] = hvac.e[2]
    T[0][i]=hvac.Tn[0]
    T[1][i]=hvac.Tn[1]
    T[2][i]=hvac.Tn[2]
    #def updateEi(self,sigmai,ei_):
    dc.updateEi(mub.sigmai,mub.e_i)
    s[0][i]=dc.s[0]
    s[1][i]=dc.s[1]
    s[2][i]=dc.s[2]
    
    energy[3][i] = dc.e[0]
    energy[4][i] = dc.e[1]
    energy[5][i] = dc.e[2]   
    #print(sum(hvac.e)+sum(dc.e)+bk.ez)
    
    #    def updateEz(self,sigmaz,ez_):
    bk.updateEz(mub.sigmaz,mub.e_z)
    energy[6][i] = bk.ez
    #def updateehat(self, ei,en,ez):
    mub.updateehat(dc.e,hvac.e,bk.ez)


    
    mub.tempei=dc.e
    mub.tempen=hvac.e
    mub.tempez=bk.ez     
    #mub.update_e()
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

    curr=sum(mub.sigmai)+sum(mub.sigman)+mub.sigmaz
#
#x=np.arange(runTime)

def plot_lines(datas, numb_of_line, markerstyle, labels, title ):
   for line in range(numb_of_line):
       plt.plot(datas[line], marker = markerstyle, markersize=4, label=labels[line])
   plt.legend(loc=1)
   plt.title(title)
   plt.show()

labels = ['Office 1','Office 2','Office 3','DC 1','DC 2','DC 3','BK']
plot_lines(energy,7,".",labels,"Energy(MWh)")
plot_lines(sigma,7,"+",labels,"Sigma")
labels = ['Office 1','Office 2','Office 3','DC 1','DC 2','DC 3','BK']
plot_lines(cost,7,"+",labels,"Cost")
labels = ['Office 1','Office 2','Office 3']
plot_lines(T,3,"+",labels,"Temperature")
labels = ['DC 1','DC 2','DC 3']
plot_lines(s,3,"+",labels,"s")