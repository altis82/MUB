import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import string
from pylab import *
import plot_settings as ps
import cvxpy as cvx
import plot_MUB as pltmub

#import unNormalizedMUB as umub
import  unNormalizedMUB as nmub
reload(nmub)
#ps.set_mode("tiny")
#omega(hvac,dc,bk)
omega=np.array([0.1,0.01,0.03])#0.3$/KWh
#omega=np.array([0,1.0, 0.3*1e-3]) #set baseline1
#wear and tear cost
omega_wat=np.array([0.5,0.5,0.5])
#sla weight
omega_sla=np.array([0.5,0.5,0.5])


hvac= nmub.HVAC(25,omega[0])
Tnhat=[2.0,4.0,5.0,2.0,5.0]
hvac.setTn(Tnhat)
bk=nmub.bkGen(omega[2])
dc =nmub.DC([2.5,4.,4.],[2000,2000,2000],omega[1],omega_sla,omega_wat)
mub=nmub.operator([1,2,3],[1,2,3],100,[1,2,3],[1,2,3],200,omega,0)

rho=0.03

Q=1000 #Wh ( 1MWh)
n=3 #number of offices
I=3  # number of DCs

nmub.omega_wat=omega_wat
nmub.Q=Q
nmub.n=n
nmub.I=I
nmub.rho=rho
nmub.omega=omega
nmub.omega_sla=omega_sla
nmub.omega_wat=omega_wat

runTime = 50
maxtimeslot=60
pltmub.maxtimeslot=maxtimeslot
pltmub.runTime=runTime
nmub.runTime=runTime
# ##################################################
# # #plot convergence
#
# [hvac0,dc0,bk0,energy0,cost,DANE]=nmub.MUB([1100.0,1100.0,1100.0],0,hvac,dc,bk,mub)
# hvac= nmub.HVAC(25,omega[0])
# Tnhat=[2.0,4.0,5.0,2.0,5.0]
# hvac.setTn(Tnhat)
# bk=nmub.bkGen(omega[2])
# dc =nmub.DC([2.5,4.,4.],[2000,2000,2000],omega[1],omega_sla,omega_wat)
# mub=nmub.operator([1,2,3],[1,2,3],100,[1,2,3],[1,2,3],200,omega,1)
# [hvac1,dc1,bk1,energy1,cost1,baseline1]=nmub.MUB([1100.0,1100.0,1100.0],1,hvac,dc,bk,mub)
# hvac= nmub.HVAC(25,omega[0])
# Tnhat=[2.0,4.0,5.0,2.0,5.0]
# hvac.setTn(Tnhat)
# bk=nmub.bkGen(omega[2])
# dc =nmub.DC([2.5,4.,4.],[2000,2000,2000],omega[1],omega_sla,omega_wat)
# mub=nmub.operator([1,2,3],[1,2,3],100,[1,2,3],[1,2,3],200,omega,2)
# [hvac2,dc2,bk2,energy2,cost2,baseline2]=nmub.MUB([1100.0,1100.0,1100.0],2,hvac,dc,bk,mub)
# #
# # #plot convergence of totalcost
# pltmub.totalCost(DANE,baseline1,baseline2,3.3)
#
# #
# # #plot converge of energy
# pltmub.sub_energy(energy0)
# #################################################
lamb=nmub.trace_lamb()
DANE_trace=np.zeros(maxtimeslot/3)
baseline1_trace=np.zeros(maxtimeslot/3)
baseline2_trace=np.zeros(maxtimeslot/3)
for i in range(maxtimeslot/3):
    hvac= nmub.HVAC(25,omega[0])
    Tnhat=[2.0,4.0,5.0,2.0,5.0]
    hvac.setTn(Tnhat)
    bk=nmub.bkGen(omega[2])
    dc =nmub.DC([2.5,4.,4.],[2000,2000,2000],omega[1],omega_sla,omega_wat)
    mub=nmub.operator([1,2,3],[1,2,3],100,[1,2,3],[1,2,3],200,omega,2)
    [hvac0,dc0,bk0,energy0,cost0,total]=nmub.MUB([lamb[0][i],lamb[1][i],lamb[2][i]],0,hvac,dc,bk,mub)
    DANE_trace[i]=total[runTime-1]
    [hvac0,dc0,bk0,energy0,cost0,total]=nmub.MUB([lamb[0][i],lamb[1][i],lamb[2][i]],1,hvac,dc,bk,mub)
    baseline1_trace[i]=total[runTime-1]
    [hvac0,dc0,bk0,energy0,cost0,total]=nmub.MUB([lamb[0][i],lamb[1][i],lamb[2][i]],2,hvac,dc,bk,mub)
    baseline2_trace[i]=total[runTime-1]
    data=[DANE_trace,baseline1_trace,baseline2_trace]
    data=nmub.saveFile('data_plot.p',data)
# nmub.saveFile('DANE_trace.p',DANE_trace)
# nmub.saveFile('baseline1_trace.p',baseline1_trace)
# nmub.saveFile('base2.p',baseline2_trace)
# DANE_trace=nmub.loadFile('DANE_trace.p',DANE_trace)
# baseline1_trace=nmub.loadFile('baseline2_trace.p',baseline1_trace)
# baseline2_trace=nmub.loadFile('base2.p',baseline2_trace)
#
data=nmub.loadFile('data_plot.p',data)
print(data)
ls=['-','--','--']
markerstyle=['o','+','*']
title=''
labels=['DANE','Baseline 1','Baseline 2']

pltmub.plot_lines(data, 3, ls,markerstyle, title, labels, xlb='Time slots', ylb='Total cost')


  