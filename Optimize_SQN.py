import numpy as np
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint
from Surrogate import model_test as surrogate_model
from Nomarlize import *
import pickle


if __name__ == '__main__':
    # define the target lateral capacity
    target_capacity = [2056.133951, 3832.835886]

    # define the bounds for the variables
    d_bounds = (2, 10)
    t_bounds = (0.02, 0.25)
    l_bounds = (6.0, 150.0)
    bounds = (d_bounds, t_bounds, l_bounds)

    # define the objective function to minimize (i.e., weight of the pile)
    def objective(x):
        return np.pi * x[0] * x[1] * x[2]

    # define the constraint function (i.e., lateral capacity should be greater than target capacity)
    def constraint(x):

        diameter = x[0]
        thickness = x[1]
        length = x[2]
        Ip = (1 / 64) * np.pi * (diameter ** 4 - (diameter - 2 * thickness) ** 4)
        c1, c2 = surrogate_model(length, Ip)
        return c2 - target_capacity[1]

    diam = []
    leng= []
    thick = []
    vol = []
    h05 = []
    h1 = []

    def callback(xk):
        diam.append(xk[0])
        thick.append(xk[1])
        leng.append(xk[2])
        Ip = (1 / 64) * np.pi * (xk[0] ** 4 - (xk[0] - 2 * xk[1]) ** 4)
        c1, c2 = surrogate_model(xk[2], Ip)
        vol.append(objective(xk))
        h05.append(c1)
        h1.append(c2)
        print(len(diam))
        # print('Current variable values:', xk)
        # print('Current objective function value:', objective(xk))
        # print('Current value of c1:', c1, 'Target: 2056.133951')
        # print('Current value of c2:', c2, 'Target: 3832.835886')
        # print('***********************************')


    # initial guess for the variables
    x0 = np.array([2.5, 0.09, 36.0])

    # define the constraint as NonlinearConstraint
    nonlinear_constraint = NonlinearConstraint(constraint, 0, np.inf)

    # optimize the variables subject to the constraint
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, tol = 0.01,\
                      constraints = [nonlinear_constraint], options={'maxiter':1000, 'ftol':0.01},\
                      callback=callback)

    # print the optimized variables and the minimum weight
    print('Optimized variables:', result.x)
    print('Minimum weight:', result.fun)

    # save the optimization solution
    with open('result.pkl', 'wb') as f:
        pickle.dump(result, f)
    pd.DataFrame({'Diameter':diam, 'thickness': thick, 'length':leng, 'volume':vol, 'h05':h05, 'h1':h1}).to_csv('history.v01.csv')

