# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 09:00:01 2021

@author: Eugene
"""

%reset

import numpy as np
import pandas as pd

# Import Autodyn simulation results of parameter sweep. Parameters of interest include: mesh size of projectile, ratio of mesh size of target to mesh size of projectile, mesh size of target, erosion criteria of target and projectile material 

df = pd.read_excel('Parameter Sweep 2.xlsx')
df = df.drop(labels = [0,1], axis = 0).reset_index(drop = True)
df = df.rename(columns = df.iloc[0]).drop(df.index[0]).reset_index(drop = True)

# Referencing results by V. Madhu, T. Balakrishna and N.K. Gupta, accessed from https://core.ac.uk/download/pdf/333720448.pdf

ref_vel = 411
ref_thick = 20

df_obj = df[['Remaining Thickness','Residual Vel']]
df_obj = df_obj.iloc[0:33,:]
df_obj = df_obj.fillna(0)

df2 = df.iloc[0:33,:]
df2 = df2.drop(['factor name'], axis = 1)

df3 = df2[['Threat Mesh (x,y)','Target mesh (x,y) (proportion of Target mesh)','Erosion Criteria AP','Erosion Criteria Target']]
df3.columns = ['Threat Mesh',' Target Mesh Ratio','Erosion Criteria Threat','Erosion Criteria Target']

df4 = df2[['Threat Mesh (x,y)','Target mesh (x,y)','Erosion Criteria AP','Erosion Criteria Target']]
df4.columns = ['Threat Mesh',' Target Mesh','Erosion Criteria Threat','Erosion Criteria Target']

# initial approach used a sum of the scaled values of the residual thickness (if the projectile did not penetrate the target) or residual velocity (if it did) as the cost function

from sklearn.preprocessing import StandardScaler

trans1 = StandardScaler()
trans2 = StandardScaler()

#obj1 = np.array(df_obj['Residual Vel'] - ref_vel)
obj1 = np.array(1-abs(df_obj['Residual Vel']-ref_vel)/ref_vel)
obj1 = obj1.reshape(-1,1)

obj2 = np.array(df_obj['Remaining Thickness']).reshape(-1,1)

data1 = trans1.fit_transform(obj1)
data2 = trans2.fit_transform(obj2)

#obj = data1 + data2

obj = obj1 - obj2

# Subsequently, an elliptic paraboloid loss function was used as a scaling factor to penalise results where the projectile did not penetrate the target. The code below was referenced from a response provided in Stack Overflow (https://stackoverflow.com/questions/50711530/what-would-be-a-good-loss-function-to-penalize-the-magnitude-and-sign-difference)

# elliptic_paraboloid_loss is a scaling factor to apply to loss function

def elliptic_paraboloid_loss(x, y, c_diff_sign, c_same_sign):

    # Compute a rotated elliptic parabaloid.
    t = np.pi / 4

    x_rot = (x * np.cos(t)) + (y * np.sin(t))

    y_rot = (x * -np.sin(t)) + (y * np.cos(t))

    z = ((x_rot**2) / c_diff_sign) + ((y_rot**2) / c_same_sign)

    return(z)

c_diff_sign = 10
c_same_sign = 2

a = np.arange(-5, 5, 0.1)

b = np.arange(-5, 5, 0.1)

loss_map = np.zeros((len(obj)))

for i in range(len(obj)):

       loss_map[i] = obj[i]*elliptic_paraboloid_loss(obj[i], 1, c_diff_sign, c_same_sign)

# Split the data into the training and testing data set and fit regression models to them. As the relation between the variables are unknown, polynomial features was used to include non-linear terms. Lasso and Ridge regressions were also performed to try to minimise the number of sailent features.
        
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model

order = 2

poly = PolynomialFeatures(order)
poly_variables = poly.fit_transform(df3)

# Try fitting use the ratio of mesh sizes

score1 = 0
score2 = 0
score3 = 0

while (score2 < 0.65 or score3 < 0.65):

    poly_var_train, poly_var_test, res_train, res_test = train_test_split(poly_variables,loss_map,test_size = 0.8)    

    regression = linear_model.LinearRegression()
    ridge = linear_model.RidgeCV(alphas = [1e-3,1e-2,1e-1,1,5,10])
    lasso = linear_model.LassoCV(alphas = [1e-3,1e-2,1e-1,1,5,10], max_iter = 30000,cv = 5)
    
    linreg = regression.fit(poly_var_train, res_train)
    score1 = linreg.score(poly_var_test,res_test)
    
    linridge = ridge.fit(poly_var_train,res_train)
    score2 = linridge.score(poly_var_test,res_test)
    
    linlasso = lasso.fit(poly_var_train,res_train.ravel())
    score3 = linlasso.score(poly_var_test,res_test.ravel())
    
    print(score2)
    print(score3)
    print((score2 < 0.65 or score3 < 0.65))

# After a fit was obtained, an optimisation was attempted to find optimal parameters that would maximise the accuracy of simulations. 

from scipy.optimize import curve_fit as cf
from scipy import optimize as opt

import random

optim_param_lasso = []
optim_func_lasso = 0

optim_param_ridge = []
optim_func_ridge = 0

# define constraints

x0_bounds = ((0.5,1),(1,3),(.5,5),(.5,5))

# repeat with random seed to find the global minima

for i in range(1000):
    
    a = round(random.uniform(0.1,1),2)
    b = round(random.uniform(1,3),2)
    c = round(random.uniform(0.5,5),2)
    d = round(random.uniform(0.5,5),2)
    
    x0 = np.array([a,b,c,d])
    
    opt_param_lasso = opt.minimize(lambda x: linlasso.predict(poly.transform(np.array([x]).reshape(1,4))), x0,bounds = x0_bounds)
    opt_param_ridge = opt.minimize(lambda x: linridge.predict(poly.transform(np.array([x]).reshape(1,4))), x0,bounds = x0_bounds)
    
    if opt_param_lasso.fun < optim_func_lasso:
        
        optim_param_lasso = opt_param_lasso.x
        optim_func_lasso = opt_param_lasso.fun

    if opt_param_ridge.fun < optim_func_ridge:
        
        optim_param_ridge = opt_param_ridge.x
        optim_func_ridge = opt_param_ridge.fun

# Repeat process using mesh size of the target instead of the ratio

poly2 = PolynomialFeatures(order)
poly_variables2 = poly2.fit_transform(df4)

score4 = 0
score5 = 0
score6 = 0

while (score5 < 0.7 or score6 < 0.7):

    poly_var_train2, poly_var_test2, res_train2, res_test2 = train_test_split(poly_variables2,loss_map,test_size = 0.3)
        
    regression2 = linear_model.LinearRegression()
    ridge2 = linear_model.RidgeCV(alphas = [1e-3,1e-2,1e-1,1,5,10])
    lasso2 = linear_model.LassoCV(alphas = [1e-3,1e-2,1e-1,1,5,10], max_iter = 20000,cv = 5)
    
    linreg2 = regression2.fit(poly_var_train2, res_train2)
    score4 = linreg2.score(poly_var_test2,res_test2)
    
    linridge2 = ridge2.fit(poly_var_train2,res_train2)
    score5 = linridge2.score(poly_var_test2,res_test2)
    
    linlasso2 = lasso2.fit(poly_var_train2,res_train2)
    score6 = linlasso2.score(poly_var_test2,res_test2)   
    
    print(score5)
    print(score6)
    print(score5 < 0.7 or score6 < 0.7)

# Attempt optimisation  

optim_param_lasso2 = []
optim_func_lasso2 = 0

optim_param_ridge2 = []
optim_func_ridge2 = 0

x0_bounds2 = ((0.1,1),(1,3),(.5,5),(.5,5))

for i in range(1000):
    
    a = round(random.uniform(0.1,1),2)
    b = round(random.uniform(0.1,3),2)
    c = round(random.uniform(0.5,5),2)
    d = round(random.uniform(0.5,5),2)
    
    x02 = np.array([a,b,c,d])
    
    opt_param_lasso2 = opt.minimize(lambda x: linlasso2.predict(poly2.transform(np.array([x]).reshape(1,4))), x0,bounds = x0_bounds2)
    opt_param_ridge2 = opt.minimize(lambda x: linridge2.predict(poly2.transform(np.array([x]).reshape(1,4))), x0,bounds = x0_bounds2)
    
    if opt_param_lasso2.fun < optim_func_lasso2:
        
        optim_param_lasso2 = opt_param_lasso2.x
        optim_func_lasso2 = opt_param_lasso2.fun

    if opt_param_ridge2.fun < optim_func_ridge2:
        
        optim_param_ridge2 = opt_param_ridge2.x
        optim_func_ridge2 = opt_param_ridge2.fun

                   
