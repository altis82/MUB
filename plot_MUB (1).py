# -*- coding: utf-8 -*-
"""
Filename: plot_colo.py
Authors: Nguyen H. Tran
LastModified: 11/2014
"""
from __future__ import division

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['legend.fancybox'] = True
import plot_settings as ps
from pylab import *
reload(ps)



num_days = 1
scale = 100
markers = ['o','x', 's','v','d','^','<','>','+','*','x']
linestyles = ['-','--']#,'-.']
linewidth = 1.5

start_iter = 0
runTime = 100
maxtimeslot=100
### Traces
def plot_traces(Lamb_array): 
    ps.set_mode("small")
    fig, ax = plt.subplots()
    x = np.arange(np.size(Lamb_array[0])) + 1
    name = ['MSR', 'FIU', 'Syn.1','Syn.2','Syn.3']
    for i in range(len(name)):
        y = Lamb_array[i]
        if i < len(linestyles):
            ax.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = name[i], ls = linestyles[i])
        else:
            ax.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = name[i], ls = linestyles[-1], marker = markers[i]) 
    fig.tight_layout()
    ax.set_xlabel(r'Time (h)')
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_ylim([0., 1])
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig('traces.pdf')
    plt.show()

def plot_lines(datas, numb_of_line, ls,markerstyle, title, labels, xlb, ylb):
    ps.set_mode("small")
    fig=plt.gcf()
    stride=20
    for line in range(numb_of_line):
        plt.plot(datas[line], ls=ls[line],marker = markerstyle[line], markersize=5, label=labels[line],markevery=stride)
    plt.legend(loc=1)
    #plt.xlim([0,50])
    #plt.ylim(0,20)
    plt.title(title)
    plt.xlabel=xlb
    plt.ylabel=ylb
    plt.show()
    name=ylb+'.pdf'
    fig.savefig(name)
#####################################################################
def sub_energy(energy):
    ps.set_mode("small")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax.set_xlabel('common xlabel')
    ax.set_ylabel('common ylabel')

    axarr0=subplot(2,1,1)
    axarr0.plot(energy[0],ls='-',marker='',label='Office 1')
    axarr0.plot(energy[1],ls='-',marker='+',label='Office 2')
    axarr0.plot(energy[2],ls='-',marker='.',label='Office 3')
    axarr0.set_xlim([0,runTime])
    axarr0.set_ylim([0,200])
    axarr0.legend(loc=1, ncol=1, shadow=True,  fancybox=False)
    #xticks([]), yticks([0,20,40,60,80])

    axarr1=subplot(2,1,2)
    axarr1.plot(energy[3],ls='--',marker='+',label='DC 1')
    axarr1.plot(energy[4],ls='--',marker='.',label='DC 2')
    axarr1.plot(energy[5],ls='--',marker='',label='DC 3')
    axarr1.plot(energy[6],ls='-',marker='',label='BK')
    axarr1.set_xlim([0,runTime])
    axarr1.set_ylim([-5,300])


    axarr1.legend(loc=1, ncol=1, shadow=True,  fancybox=False)

    #xticks([]), yticks([0,20,40,60,80])

    # axarr2=subplot(3,1,3)
    # axarr2.plot(energy[6],ls='-',marker='.',label='Backup')
    #
    #
    # axarr2.legend(loc="right", ncol=1, shadow=True,  fancybox=False)
    # axarr2.set_ylim([75,80])
    # axarr2.set_xlim([0,30])
    # yticks([75,77,79])
    fig.text(0.5, 0.04, 'Iterations', ha='center', va='center')
    fig.text(0.06, 0.5, 'Energy (Kwh)', ha='center', va='center', rotation='vertical')
    plt.show()
    fig.savefig('energy_conv.pdf')
#############################################################
def totalCost(DANE,baseline1,baseline2,opt):
    #
    fig=plt.figure(figsize=(6,3.5))
    stride=5
    plt.plot(DANE, marker = '^', markersize=5, label='DANE',markevery=stride)
    plt.plot(baseline1-0.3, ls='--',marker = '+', markersize=5, label='Baseline 1',markevery=stride)
    plt.plot(baseline1, ls='--',marker = 'o', markersize=5, label='Baseline 2',markevery=stride)
    #optimal value
    plt.plot(np.ones(runTime)*opt,ls='--', markersize=5, label='Optimal')
    plt.xlim([0,50])
    #plt.ylim([0,10])
    #plt.ylim([35,60])
    plt.legend(loc=1)
    plt.xlabel('Iterations')
    plt.ylabel('Total cost')
    plt.show()

    fig.savefig('plot/totalcost_conv.pdf')

#########################################################
def totalCost_trace(DANE,baseline1,baseline2):
    #

    fig=plt.figure(figsize=(6,3.5))
    stride=5
    plt.plot(DANE, marker = '^', markersize=5, label='DANE',markevery=stride)
    plt.plot(baseline1-0.3, ls='--',marker = '+', markersize=5, label='Baseline 1',markevery=stride)
    plt.plot(baseline1, ls='--',marker = 'o', markersize=5, label='Baseline 2',markevery=stride)
    #optimal value
    #plt.plot(np.ones(runTime)*opt,ls='--', markersize=5, label='Optimal')
    plt.xlim([0,50])
    #plt.ylim([0,10])
    #plt.ylim([35,60])
    plt.legend(loc=1)
    plt.xlabel('Time slots')
    plt.ylabel('Total cost')
    plt.show()

    fig.savefig('plot/totalcost_trace.pdf')
#######################################################
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

### Alpha comparison    
def plot_alpha(nu_SWO, price_convg_list_alpha, alpha_prov_pay_list, alpha_list):
    ps.set_mode("small") 
    fig, (ax1, ax2) = plt.subplots(1, 2)

    x_range = range(len(price_convg_list_alpha) + 1)    
    stride = 10.
    for i in x_range:
        if i != max(x_range):
            x = range(len(price_convg_list_alpha[i]))       
            y = price_convg_list_alpha[i]
            current_label = r"$\alpha=${}".format(alpha_list[i])
        else:
            x = range(len(nu_SWO))
            y = nu_SWO
            current_label = r'$\nu^{*}$'
#        if i < len(linestyles):
#            ax1.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, ls = linestyles[i])
#        else:
        ax1.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label =current_label, ls = linestyles[0], marker = markers[i], markevery=stride)
      
 #   ax1.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, marker = markers[i])
            
    y_max = max([i[-1] for i in price_convg_list_alpha])
    ax1.set_ylim(0.5*y_max, 2.5*y_max )
    ax1.set_xlabel('iterations')
    ax1.set_ylabel(r'$g(\Theta^{ne})$')
# Legend   
    leg = ax1.legend(loc="upper center", ncol=2, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 1.10))
#    leg.get_frame().set_facecolor('0.9')    # set the frame face color to light gray
#    leg.get_frame().set_alpha(0.99)
#    for t in leg.get_texts():
#        t.set_fontsize('small')    # the legend text fontsize
#    for l in leg.get_lines():
#        l.set_linewidth(1.0)  # the legend line width
    
    x = alpha_list
    y = alpha_prov_pay_list
    ax2.plot(x,y, lw=linewidth, alpha=0.6)
    
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'$\sum\limits_{i} R_i$')
    
    fig.tight_layout()
    plt.savefig('alpha.pdf')
    plt.show() 
        
#def plot_provider_payment_alpha(alpha_prov_pay_list, alpha_list):
#    ps.set_mode("small") 
#    fig, ax = plt.subplots()
#    
#    x = alpha_list
#    y = alpha_prov_pay_list
#    ax.plot(x,y, lw=linewidth, alpha=0.6)
#    
#    ax.set_xlabel(r'$\alpha$')
#    ax.set_ylabel(r'$\sum\limits_{i} R_i$')
#    ax.legend(loc="best")
#    plt.savefig('total_payment_alpha.pdf')
#    plt.show() 
    
### Convergence    
def plot_price_convg(provider, k): 
    ps.set_mode("small") 
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)    
    
    x = np.arange(k)[start_iter:]
    y = provider.price_convg_list[start_iter:]
    ax1.plot(x,y, lw=linewidth, alpha=0.6, marker = markers[1])     
    y_max = provider.price_convg_list[-1]
    ax1.set_ylim(0.5*y_max, 1.5*y_max )   
    ax1.set_xlabel('iterations')
    ax1.set_ylabel(r'$g(\Theta^{ne})$')
   
    
    
    y = provider.price_convg_SWO_list[start_iter:]
    ax2.plot(x,y, lw=linewidth, alpha=0.6, marker = markers[2])   
    y_max = provider.price_convg_SWO_list[-1]
    ax2.set_ylim(0.5*y_max, 1.5*y_max )
    ax2.set_xlabel('iterations')
    ax2.set_ylabel(r'$\nu^*$')
    ax2.legend(loc="upper left")
    
    fig.tight_layout()
    plt.savefig('price_convg.pdf')
    plt.show() 
#def plot_price_convg_SWO(provider, k): 
#    ps.set_mode("small") 
#    fig, ax = plt.subplots()
#    x = np.arange(k)[start_iter:]
#    y = provider.price_convg_SWO_list[start_iter:]
#    ax.plot(x,y, lw=linewidth, alpha=0.6, marker = markers[1]) 
#    
#    y_max = provider.price_convg_SWO_list[-1]
#    ax.set_ylim(0., 1.5*y_max )
#    ax.set_xlabel('iterations')
#    ax.set_ylabel(r'$\nu^*$')
#    ax.legend(loc="best")
#    plt.show() 
    
def plot_bid_convg(tenant_list, k):
    ps.set_mode("small")    
    fig, ax = plt.subplots()
    x = np.arange(k)[start_iter:]
    stride = max( int(len(x)/10), 1 )
    for i in range(len(tenant_list)):
        y = tenant_list[i].bid_convg_list[start_iter:]
        current_label = r'Tenant {}'.format(i+1)
        
#        if i < len(linestyles):
#            ax.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, ls = linestyles[i])
#        else:
        ax.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, ls = linestyles[0], marker = markers[i], markevery=stride) 
   
    ax.set_xlabel('iterations')
    ax.set_ylabel(r'$\theta_i$')
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig('bid_convg.pdf')
    plt.show() 
def plot_reward_convg(tenant_list, k):
    ps.set_mode("small")    
    fig, ax = plt.subplots()
    x = np.arange(k)[start_iter:]
    stride = max( int(len(x)/10), 1 )
    for i in range(len(tenant_list)):
        y = tenant_list[i].reward_convg_list[start_iter:]
        current_label = r'Tenant {}'.format(i+1)
#        if i < len(linestyles):
#            ax.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, ls = linestyles[i])
#        else:
        ax.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, ls = linestyles[0], marker = markers[i], markevery=stride)
      
    y_max = max([i.reward_convg_list[-1] for i in tenant_list])
    ax.set_ylim(0., 1.5*y_max )
    ax.set_xlabel('iterations')
    ax.set_ylabel(r'$R_i$')
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig('reward_convg.pdf')
    plt.show() 
    
def plot_e_convg_SWO(tenant_list, k):
    ps.set_mode("small")    
    fig, ax = plt.subplots()
    x = np.arange(k)[start_iter:]
    stride = max( int(len(x)/10), 1 )
    for i in range(len(tenant_list)):
        y = (np.array(tenant_list[i].e_convg_SWO_list))[start_iter:]*scale
        current_label = r'Tenant {}'.format(i+1)
#        if i < len(linestyles):
#            ax.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, ls = linestyles[i])
#        else:
        ax.plot(x,y, lw=linewidth, color=ps.colors[i], alpha=0.6, label = current_label, ls = linestyles[0], marker = markers[i], markevery=stride)
    ax.set_xlabel('iterations')
    ax.set_ylabel(r'$\Delta e_i$')
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig('e_convg_SWO.pdf')
    plt.show() 


### Hours comparison   
def plot_tenant_reward(tenant_reward_hours_array, num_hours, num_tenants,scheme=1): 
    ps.set_mode("small") 
    fig, ax = plt.subplots()
    
    width = 0.5
    patterns = ["|||", "\\"  ,"///" , "...", "---" , "xxx" , "+++" ,    "o",   "O",  "*" ]
    ind = np.arange(num_hours) + 1
    name = ['Tenant 1', 'Tenant 2', 'Tenant 3','Tenant 4','Tenant 5']    
    bottom = np.zeros(num_hours)
    
    for i in range(num_tenants):
        ax.bar(ind, tenant_reward_hours_array[:,i], width=width, color= 'white', align='center', edgecolor='black', \
        label=name[i], hatch=patterns[i],  bottom=bottom)
        bottom += tenant_reward_hours_array[:,i]
        
    ax.set_xlabel(r'Time (h)')
    ax.set_ylabel(r'$R_i$')
    ax.set_xlim([-0.5, num_hours + 1])
    
    if scheme == 1:
       # ax.set_title('EPM')
        plt.savefig('tenant_reward_EPM.pdf')
    elif scheme ==2:
       # ax.set_title('SWO')
        plt.savefig('tenant_reward_SWO.pdf')
    else:
        ax.legend(loc="best")
       # ax.set_title('RAND') 
        plt.savefig('tenant_reward_RAND.pdf')
    plt.show() 
def plot_tenant_e(tenant_e_hours_list, num_hours, num_tenants, scheme=1): 
    ps.set_mode("small") 
    fig, ax = plt.subplots()
    
    width = 0.5
    patterns = ["|||", "\\"  ,"///" , "...", "---" , "xxx" , "+++" ,    "o",   "O",  "*" ]
    index = np.arange(num_hours) + 1
    name = ['Tenant 1', 'Tenant 2', 'Tenant 3','Tenant 4','Tenant 5']    
    bottom = np.zeros(num_hours)
    
    for i in range(num_tenants):
        ax.bar(index, tenant_e_hours_list[:,i]*scale, width=width, color= 'white', align='center', edgecolor='black', \
        label=name[i], hatch=patterns[i],  bottom=bottom)
        bottom += tenant_e_hours_list[:,i]*scale
   
    ax.set_xlabel(r'Time (h)')
    ax.set_ylabel(r'$\Delta e_i$')
    ax.set_xlim([-0.5, num_hours+1])
        
    if scheme == 1:
       # ax.set_title('EPM')
        plt.savefig('tenant_e_EPM.pdf')
    elif scheme ==2:
       # ax.set_title('SWO')
        plt.savefig('tenant_e_SWO.pdf')
    else:
        ax.legend(loc="best")
       # ax.set_title('RAND') 
        plt.savefig('tenant_e_RAND.pdf')
    plt.show()  
    
def plot_tenant_cost(tenant_cost_hours_array, num_hours, num_tenants, scheme=1):
    ps.set_mode("small") 
    fig, ax = plt.subplots()
    
    width = 0.5
    patterns = ["|||", "\\"  ,"///" , "...", "---" , "xxx" , "+++" ,    "o",   "O",  "*" ]
    ind = np.arange(num_hours) + 1
    name = ['Tenant 1', 'Tenant 2', 'Tenant 3','Tenant 4','Tenant 5']    
    bottom = np.zeros(num_hours)
    
    for i in range(num_tenants):
        ax.bar(ind, tenant_cost_hours_array[:,i], width=width, color= 'white', align='center', edgecolor='black', \
        label=name[i], hatch=patterns[i],  bottom=bottom)
        bottom += tenant_cost_hours_array[:,i]
   
    ax.set_xlabel(r'Time (h)')
    ax.set_ylabel(r'$C_i(m_i)$')
    ax.set_xlim([-0.5, num_hours + 1])
     
    if scheme == 1:
       # ax.set_title('EPM')
        plt.savefig('tenant_cost_EPM.pdf')
    elif scheme ==2:
       # ax.set_title('SWO')
        plt.savefig('tenant_cost_SWO.pdf')
    else:
        ax.legend(loc="best")
       # ax.set_title('RAND') 
        plt.savefig('tenant_cost_RAND.pdf')
    plt.show() 
     

def plot_sum_cost(sum_cost_all_array, num_hours, num_tenants):
    ps.set_mode("small") 
    fig, ax = plt.subplots()    
    index = np.arange(num_hours)+1
    bar_width = 0.25
    space = 0.01
    opacity = 0.99
    error_config = {'ecolor': '0.3'}
    patterns = ["|||", "\\"  ,"///" , "...", "---" , "xxx" , "+++" ,    "o",   "O",  "*" ]
    
    alg1 = ax.bar(index, sum_cost_all_array[0,:]*scale, bar_width, 
                 alpha=opacity,
                 color= 'white',
                 error_kw=error_config, hatch=patterns[1],
                 label='EPM', align='center')  
    alg2 = ax.bar(index + bar_width + space, sum_cost_all_array[1,:]*scale, bar_width,
                 alpha=opacity,
                 color= 'white',
                 error_kw=error_config, hatch=patterns[2],
                 label='SWO', align='center')               
    alg3 = ax.bar(index + 2*(bar_width + space), sum_cost_all_array[2,:]*scale, bar_width,
                 alpha=opacity,
                 color= 'white',
                 error_kw=error_config, hatch=patterns[3],
                 label='RAND',align='center') 
#    autolabel(alg1)
#    autolabel(alg2, 0.01)
#    autolabel(alg3, 0.03)
    plt.xticks(index + bar_width, index, rotation='0', fontsize=8)
    ax.set_xlim(0,len(index) + 2)
    y_max = sum_cost_all_array.max()*scale
    ax.set_ylim(0., 1.1*y_max )
    ax.set_ylabel(r"$\sum\nolimits_{i} C_i(m_i)$")
    ax.set_xlabel(r'Time (h)')
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig("sum_cost.pdf")
    
def plot_tenant_util(tenant_reward_hours_array, tenant_cost_hours_array, num_hours, num_tenants, scheme=1):
    ps.set_mode("small") 
    fig, ax = plt.subplots()
    
    width = 0.5
    patterns = ["|||", "\\"  ,"///" , "...", "---" , "xxx" , "+++" ,    "o",   "O",  "*" ]
    ind = np.arange(num_hours) + 1
    name = ['Tenant 1', 'Tenant 2', 'Tenant 3','Tenant 4','Tenant 5']    
    bottom = np.zeros(num_hours)
    y = tenant_reward_hours_array - tenant_cost_hours_array
    print " Tenants' utility is: {} ".format(y)
    for i in range(num_tenants):
        ax.bar(ind, y[:,i], width=width, color= 'white', align='center', edgecolor='black', \
        label=name[i], hatch=patterns[i],  bottom=bottom)
        bottom += y[:,i]
   
    ax.set_xlabel(r'Time (h)')
    ax.set_ylabel("Tenants' utility")
    ax.set_xlim([-0.5, num_hours + 1])
     
    if scheme == 1:
       # ax.set_title('EPM')
        plt.savefig('tenant_util_EPM.pdf')
    elif scheme ==2:
       # ax.set_title('SWO')
        plt.savefig('tenant_util_SWO.pdf')
    else:
        ax.legend(loc="best")
       # ax.set_title('RAND') 
        plt.savefig('tenant_util_RAND.pdf')
    plt.show() 