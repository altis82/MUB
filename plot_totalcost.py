import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import  unNormalizedMUB as umub

baseline1=np.zeros(100)
DANE=np.zeros(100)

baseline1=umub.loadFile('baseline.p',baseline1)
DANE=umub.loadFile('DANE.p',baseline1)
baesline2=umub.loadFile('baseline2.p',baseline2)

plt.plot(baseline1,marker = '+', markersize=4, label='Baseline 1')
plt.plot(DANE,color='r',marker = 'o', markersize=4, label='DANE 1')
plt.plot(baseline2,marker = '+', markersize=4, label='Baselin 2')
plt.legend(loc=1)