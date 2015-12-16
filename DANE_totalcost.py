import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import string
#import unNormalizedMUB as umub
import  unNormalizedMUB as nmub
import plot_MUB as pltmub
reload(pltmub)
reload(nmub)

omega=np.array([0.5,0.01/5,0.3])#0.3$/KWh
#omega=np.array([0,1.0, 0.3*1e-3]) #set baseline1
#wear and tear cost
omega_wat=np.array([0.5,0.5,0.5])
#sla weight
omega_sla=np.array([0.5,0.5,0.5])


hvac= nmub.HVAC(25,omega[0])
Tnhat=[2.0,4.0,5.0,2.0,5.0]
hvac.setTn(Tnhat)
bk=nmub.bkGen(omega[2])
rho=0.03

Q=300 #Wh ( 1MWh)
n=3 #number of offices
I=3  # number of DCs



######################################################
trace=np.zeros(300)
with open('facebook.csv', 'r') as csvfile:    
     i=0   
     for row in csvfile:
         if i>=300:
             break
         trace[i]=string.atof(row)
         i=i+1

#normalize trace
trace=np.array(trace)/np.max(trace)   
#create lambda for DC
lamb=np.ones((3,100))
for i in range(100):
    lamb[0][i]=trace[i]*1500
    lamb[1][i]=trace[i+10]*1500
    lamb[2][i]=trace[i+20]*1500
################################################
    
nmub.omega_wat=omega_wat
nmub.Q=Q
nmub.n=n
nmub.I=I
nmub.rho=rho



totalcost=np.zeros(100)
energy = np.ones((7,100))
cost = np.ones((7,100))
for t in range(100):
    nmub.MUB(lamb,0)

    cost[0][i]=hvac.userComfCosti(0)
    cost[1][i]=hvac.userComfCosti(1)
    cost[2][i]=hvac.userComfCosti(2)
    cost[3][i]=dc.dcCosti(0)
    cost[4][i]=dc.dcCosti(1)
    cost[5][i]=dc.dcCosti(2)
    cost[6][i]=bk.bgCost()
        
    energy[0][t] = hvac.e[0]
    energy[1][t] = hvac.e[1]
    energy[2][t] = hvac.e[2]
    energy[3][t] = dc.e[0]
    energy[4][t] = dc.e[1]
    energy[5][t] = dc.e[2]   
    energy[6][t] = bk.ez    
    cost[0][t]=hvac.userComfCosti(0)
    cost[1][t]=hvac.userComfCosti(1)
    cost[2][t]=hvac.userComfCosti(2)
    cost[3][t]=dc.dcCosti(0)
    cost[4][t]=dc.dcCosti(1)
    cost[5][t]=dc.dcCosti(2)
    cost[6][t]=bk.bgCost()
    
    totalcost[t]=hvac.userComfCost()+dc.dcCost()+bk.bgCost()
    
    

################################################
nmub.saveFile('DANE.p',totalcost)
plt.plot(totalcost)

labels = ['Office 1','Office 2','Office 3','DC 1','DC 2','DC 3','BK']
def totalCost(data,opt):
    #
    fig=plt.figure(figsize=(6,3.5))  
    
    plt.plot(data, color='black',marker = '^', markersize=3, label='Total cost')
    #optimal value    
    plt.plot(np.ones(runTime)*opt,color='black',ls='--', markersize=4, label='Optimal')
    plt.xlim([0,runTime])
    plt.ylim([35,60])
    plt.legend(loc=1)
    plt.xlabel('Iterations')
    plt.ylabel('Total cost')
    plt.show()
  
    #fig.savefig('DANE_conv.pdf')

#labels = ['Office 1','Office 2','Office 3','DC 1','DC 2','DC 3','BK']
#plot_lines(energy,7,".",labels,"Energy")
#plot_lines(sigma,7,"+",labels,"Sigma")
#labels = ['Office 1','Office 2','Office 3','DC 1','DC 2','DC 3','BK']
#plot_lines(cost,7,"+",labels,"Cost")
#labels = ['Office 1','Office 2','Office 3']
#plot_lines(T,3,"+",labels,"Temperature")
#labels = ['DC 1','DC 2','DC 3']
#plot_lines(s,3,"+",labels,"s")