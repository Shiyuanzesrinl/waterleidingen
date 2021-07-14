#!/usr/bin/env python
# coding: utf-8

# In[7]:


#import os
# ! pip install env

import arcpy
#import sys
sys.version
#print("This is the name of the program:", sys.argv[0])
  
#print("Argument List:", str(sys.argv))


# In[8]:


from arcpy import env
import pandas as pd
import time
from IPython.display import Image
from IPython.display import HTML
import matplotlib.pyplot as plt 

 
from arcgis.gis import GIS
from arcgis.features import SpatialDataFrame
from arcgis.geometry import Geometry
from arcgis.geometry.filters import intersects


# In[9]:


leidingen_item = r'C:\Internship_work\Model\model_output.gdb\shadesleidingn_r_g_res_3'
leidingen_df = pd.DataFrame.spatial.from_featureclass(leidingen_item)
#leidingen_item_f1 = leidingen_item[0]
leidingen_df.head(5)
waterleidingen_others = r'C:\Internship_work\Model\model_input.gdb\waterleidingen_new_1'


# In[11]:


#leidingen_df[["id", "OBJECTID_1"]].head(5)
import numpy as np
 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.multiclass import type_of_target


label = leidingen_df["Problem"]
feature_names = leidingen_df.columns.values
feature_names = feature_names.tolist()
print(feature_names)
 
features = leidingen_df[["VALUE_MAX_1_IMPUTED", "address_num","SLOPE_1_IMPUTED", "overgang_num","loam_new", "ONEHOT_weg_functi_others", "ONEHOT_weg_fysiek_open_verharding","ONEHOT_MATERIAAL_AC","ONEHOT_weg_functi_fiet","ONEHOT_weg_functi_parkarea" ]]
x_train = features
y_train = label

print(np.shape(y_train))

type_of_target(y_train)


# In[12]:


#! pip install bayesian-optimization
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import random
 

def bayesian_optimization(dataset, function, parameters):
    x_train, y_train = dataset
    n_iterations = 5
#    gp_params = {"alpha": 1e-5}

    BO = BayesianOptimization(function, parameters, random_state  = 1)
    BO_m = BO.maximize(n_iter=n_iterations)
#    print(BO_m)
    print(BO.max)
    return BO_m

def rfc_optimization(cv):
    global x_train, y_train
    def function(n_estimators, max_depth, min_samples_leaf):
        rf =   RandomForestClassifier(
                   n_estimators=int(max(n_estimators,0)),                                                               
                   max_depth=int(max(max_depth,1)),
                   min_samples_leaf=int(max(min_samples_leaf,2)), 
                   oob_score=True,
                   n_jobs=-1, 
 #                  random_state = 0,
                   warm_start = True)
        
        rf.fit(x_train, y_train)
        
        def oob_error(rf):
            return rf.oob_score_
        
        def mean_score_10(rf):
            randomlist = []
            for i in range(0,10):
                oob = oob_error(rf) 
                randomlist.append(oob)
            return max(randomlist)
                
        return mean_score_10(rf)
     
    parameters = {"n_estimators": (500, 1000),
                  "max_depth": (1, 150),
                  "min_samples_leaf": (1, 5)}
    
    return function, parameters
 
        
dataset = (x_train, y_train)
cv = KFold(n_splits=2, shuffle=True, random_state = 0)
function, parameters = rfc_optimization(cv)  
function
best_solution = bayesian_optimization(dataset, function, parameters)   
 
best_solution
 

# In[ ]:


arcpy.stats.Forest("PREDICT_FEATURES", leidingen_item, "Problem", "CATEGORICAL", "VALUE_MAX_IMPUTED false;address_num false;Year_afsluiters false;loam_new true;ONEHOT_weg_functi_others true", None, None, waterleidingen_others, r"C:\Internship_work\Model\model_output.gdb\Prediction_pro_notebook_test1", None, "VALUE_MAX_IMPUTED VALUE_MAX_IMPUTED;address_num address_num;Year_afsluiters Year;loam_new loam_new;ONEHOT_weg_functi_others 'others (weg_functi_One-hot)'", None, None, None, None, "TRUE", 457, 2, 29, 100, None, 10, None, None, "FALSE", 1, "FALSE")



@Misc{,
    author = {Fernando Nogueira},
    title = {{Bayesian Optimization}: Open source constrained global optimization tool for {Python}},
    year = {2014--},
    url = " https://github.com/fmfn/BayesianOptimization"
}


# In[ ]:




