import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import string
from pylab import *
import plot_settings as ps
import cvxpy as cvx


#import unNormalizedMUB as umub
import  unNormalizedMUB as nmub
reload(nmub)
#ps.set_mode("tiny")
#omega(hvac,dc,bk)
omega=np.array([0.05,0.01/5,0.3])#0.3$/KWh
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
dc =nmub.DC([2.5,4.,4.],[2000,2000,2000],omega[1],omega_sla,omega_wat)

#arrival rate
dc.setlamb([1100.0,1100.0,1100.0])
#dc.setSwitchSever([200,100,100,200,100])
#dc.minServer()
#dc.solx()
#def __init__(self,sigmai,sigman,e_i,e_n,e_z):
mub=nmub.operator([1,2,3],[1,2,3],100,[1,2,3],[1,2,3],200,omega)
#mub.calv()
#    def updateEn(self,sigman,en_):
runTime = 500
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

    #hvac.updateEn(mub.sigman,mub.e_n)
    #print("en",hvac.e)
    #hvac.updateEn2(self,sigman,en_)
    hvac.updateEn2(mub.sigman,mub.e_n)
    #    def updateEi(self,sigmai,ei_):
    #print ("Ti",hvac.Tn)
    energy[0][i] = hvac.e[0]
    energy[1][i] = hvac.e[1]
    energy[2][i] = hvac.e[2]
    T[0][i]=hvac.Tn[0]
    T[1][i]=hvac.Tn[1]
    T[2][i]=hvac.Tn[2]
    #def updateEi(self,sigmai,ei_):
    #dc.updateEi(mub.sigmai,mub.e_i)
    dc.updateEi2(mub.sigmai,mub.e_i)
    s[0][i]=dc.s[0]
    s[1][i]=dc.s[1]
    s[2][i]=dc.s[2]
    
    energy[3][i] = dc.e[0]
    energy[4][i] = dc.e[1]
    energy[5][i] = dc.e[2]   
    #print(sum(hvac.e)+sum(dc.e)+bk.ez)
    
    #    def updateEz(self,sigmaz,ez_):
    #bk.updateEz(mub.sigmaz,mub.e_z)
    bk.updateEz2(mub.sigmaz,mub.e_z)
    energy[6][i] = bk.ez
    #def updateehat(self, ei,en,ez):
    #mub.updateehat(dc.e,hvac.e,bk.ez)
    # def updateehat2(self,ei,en,ez):

    mub.updateehat2(dc.e,hvac.e,bk.ez)
    
    # mub.tempei=dc.e
    # mub.tempen=hvac.e
    # mub.tempez=bk.ez
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
    Totalcost[i]=cost[0][i]+cost[1][i]+cost[2][i]+cost[3][i]+cost[4][i]+cost[5][i]+cost[6][i]
    print(Totalcost[i])
#x=np.arange(runTime)
ps.set_mode("tiny")
plt.plot(Totalcost)
plt.show()
#ax.plot(x, y, color=next(color_cycler), label = current_label, marker = next(marker_cycler), markevery=stride)
def plot_lines(datas, numb_of_line, ls,markerstyle, title, labels, xlb='', ylb=''):

   fig=plt.gcf()
   stride=20
   for line in range(numb_of_line):
       plt.plot(datas[line], ls=ls[line],marker = markerstyle[line], markersize=4, label=labels[line],markevery=stride)
   plt.legend(loc=1)
   #plt.xlim([0,50])
   #plt.ylim(ylim)
   plt.title(title)
   plt.xlabel=xlb
   plt.ylabel=ylb
   plt.show()
   name=ylb+'.pdf'
   fig.savefig(name)
#####################################################################
def sub_energy(energy):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_color('none')
  ax.spines['left'].set_color('none')
  ax.spines['right'].set_color('none')
  ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
  ax.set_xlabel('common xlabel')
  ax.set_ylabel('common ylabel')
  
  axarr0=subplot(3,1,1)
  
  axarr0.plot(energy[0],ls='-',marker='',label='Office 1')
  axarr0.plot(energy[1],ls='-',marker='+',label='Office 2')
  axarr0.plot(energy[2],ls='-',marker='.',label='Office 3')
  
  axarr0.set_xlim([0,30])
 
  axarr0.legend(loc="right", ncol=1, shadow=True,  fancybox=False)
  xticks([]), yticks([0,20,40,60])
 
  axarr1=subplot(3,1,2)
  axarr1.plot(energy[3],ls='--',marker='+',label='DC 1')
  axarr1.plot(energy[4],ls='--',marker='.',label='DC 2')
  axarr1.plot(energy[5],ls='--',marker='',label='DC 3')
  axarr1.set_xlim([0,30])

 
  axarr1.legend(loc="right", ncol=1, shadow=True,  fancybox=False)
 
  xticks([]), yticks([55,60,65,70])  
  
  axarr2=subplot(3,1,3)
  axarr2.plot(energy[6],ls='-',marker='.',label='Backup')
  
 
  axarr2.legend(loc="right", ncol=1, shadow=True,  fancybox=False)
  axarr2.set_ylim([75,80])
  axarr2.set_xlim([0,30])
  yticks([75,77,79])
  fig.text(0.5, 0.04, 'Iterations', ha='center', va='center')
  fig.text(0.06, 0.5, 'Energy (Kwh)', ha='center', va='center', rotation='vertical')
  plt.show()
  #fig.savefig('energy_conv.pdf')
#############################################################
def totalCost(data,opt):
    #
    fig=plt.figure(figsize=(6,3.5))  
    
    plt.plot(data, marker = '^', markersize=3, label='Total cost')
    #optimal value    
    plt.plot(np.ones(runTime)*opt,ls='--', markersize=4, label='Optimal')
    plt.xlim([0,runTime])
    #plt.ylim([35,60])
    plt.legend(loc=1)
    plt.xlabel('Iterations')
    plt.ylabel('Total cost')
    plt.show()
  
    #fig.savefig('totalcost_conv.pdf')

#########################################################
def costConv(cost):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_color('none')
  ax.spines['left'].set_color('none')
  ax.spines['right'].set_color('none')
  ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
  ax.set_xlabel('common xlabel')
  ax.set_ylabel('common ylabel')
  
  axarr0=subplot(3,1,1)
  
  axarr0.plot(cost[0],ls='-',marker='',label='Office 1')
  axarr0.plot(cost[1],ls='-',marker='+',label='Office 2')
  axarr0.plot(cost[2],ls='-',marker='.',label='Office 3')
  
  axarr0.set_xlim([0,30])
 
  axarr0.legend(loc="right", ncol=1, shadow=True,  fancybox=False)
  xticks([]), yticks([])
 
  axarr1=subplot(3,1,2)
  axarr1.plot(cost[3],ls='--',marker='+',label='DC 1')
  axarr1.plot(cost[4],ls='--',marker='.',label='DC 2')
  axarr1.plot(cost[5],ls='--',marker='',label='DC 3')
  axarr1.set_xlim([0,30])
  axarr1.set_ylim([55,70])
 
  axarr1.legend(loc="right", ncol=1, shadow=True,  fancybox=False)
 
  xticks([]), yticks([])  
  
  axarr2=subplot(3,1,3)
  axarr2.plot(cost[6],ls='-',marker='.',color='black',label='Backup')
  
 
  axarr2.legend(loc="right", ncol=1, shadow=True,  fancybox=False)
  
  axarr2.set_xlim([0,30])
  yticks([75,77,79])
  fig.text(0.5, 0.04, 'Iterations', ha='center', va='center')
  fig.text(0.06, 0.5, 'Energy (Kwh)', ha='center', va='center', rotation='vertical')
  fig.savefig('energy_conv.pdf')

#plot convergence of totalcost
#totalCost(Totalcost,57.9)
#print(Totalcost)

#plot converge of energy
#sub_energy(energy)


#a.plot(energy[0])
#xticks([]), yticks([])
##text(0.5,0.5, 'subplot(2,1,1)',ha='center',va='center',size=24,alpha=.5)
#
#subplot(2,1,2)
#xticks([]), yticks([])
#text(0.5,0.5, 'subplot(2,1,2)',ha='center',va='center',size=24,alpha=.5)

  
#
labels = ['Office 1','Office 2','Office 3','DC 1','DC 2','DC 3','BK']
ls=['-','-','-','--','--','--','-.']
plot_lines(energy,7,ls,['+','o','','d','+','',''],'energy',labels,'Iterations','Energy(KWh)')


#def plot_lines(datas, numb_of_line, ls,markerstyle, title, labels, xlb='', ylb=''):
#plot_lines(sigma,7,ls,['+','o','','d','+','',''],'sigma',labels,'Iterations','Energy(KWh)')
# labels = ['Office 1','Office 2','Office 3','DC 1','DC 2','DC 3','BK']
#plot_lines(cost,7,ls,['+','o','','d','+','',''],'cost',labels,'Iterations','Energy(KWh)')
# labels = ['Office 1','Office 2','Office 3']
#plot_lines(T,3,"+",labels,"Temperature")
# labels = ['DC 1','DC 2','DC 3']
# plot_lines(s,3,"+",labels,"s")


# plt.plot(Totalcost, lw='0.5',marker = '^', markersize=3, label='Total cost')
# plt.plot(np.ones(runTime)*45.1,ls='--', markersize=4, label='Optimal')
# plt.xlim([0,runTime])
# plt.ylim([30,60])
# plt.legend(loc=1)
# plt.xlabel('Iterations')
# plt.ylabel('Total cost')

#fig1=plt.gcf()
#plt.figure(figsize=(9, 4.9))
#plt.show()
#plt.draw()
#
#fig1.savefig('totalcost_conv.pdf')


#plt.savefig('common_labels.png', dpi=300)


  