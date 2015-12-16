import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import cvxpy as cvx

# Problem data.
# m = 30
# n = 3
# numpy.random.seed(1)
# A = numpy.array([1,1,1])
# b = numpy.random.randn(m)
#
# # Construct the problem.
# y=numpy.zeros(n)
# x = Variable(n)
#
# objective = Minimize(sum_squares(A*x))
# constraints = [0 <= x, x <= 1]
# prob = Problem(objective, constraints)
#
# # The optimal objective is returned by prob.solve().
# #result = prob.solve()
# time_start = time.clock()
#


I=3
n=3
sigmai=np.random.randn(I)
mu=[2,3,4]
S=[2000,2000,2000]
lamb=[1000,1000,1000]
ei_=np.array([100,100,100])

ei =cvx.Variable(I)
si=cvx.Variable(I)
constraints=[0<=ei]
constraints+=[0<=si]
slacost=cvx.sum_entries(0.5*cvx.mul_elemwise(lamb,cvx.inv_pos(mu-cvx.mul_elemwise(lamb,cvx.inv_pos(S-si)))))
watcost=cvx.sum_entries(0.5*si)
for i in range(I):
    constraints+=[ei[i]==0.2*si[i]]
objective = cvx.Minimize(-sigmai*ei+0.3*(slacost+watcost)+0.5/2*cvx.sum_squares(ei-(ei_)))
prob = cvx.Problem(objective, constraints)
result = prob.solve(solver=cvx.ECOS)
print result


