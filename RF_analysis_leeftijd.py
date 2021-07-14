#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import os
# ! pip install env

import arcpy
#import sys
sys.version
#print("This is the name of the program:", sys.argv[0])
  
#print("Argument List:", str(sys.argv))


# In[2]:


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


# In[3]:


leidingen_item = r'C:\Internship_work\Model\model_input.gdb\waterleidingen_new_1'
leidingen_df = pd.DataFrame.spatial.from_featureclass(leidingen_item)
#leidingen_item_f1 = leidingen_item[0]
leidingen_df.head(5)
print(len(leidingen_df))
leidingen_df = leidingen_df[leidingen_df.Problem.notnull()]
print(len(leidingen_df))
leidingen_df = leidingen_df[leidingen_df.leeftijd.notnull()]
print(len(leidingen_df))
 


# In[4]:


#arcpy.stats.Forest("TRAIN", leidingen_item, "prob_en_anders", "CATEGORICAL", "MATERIAAL true;VALUE_MAX_IMPUTED false;ADDRESS_NUM_IMPUTED false;age_range true;ONEHOT_weg_functi_voet false;human_new false;sand_new true;users_addr_range true", 
# None, None, None, None, None, None, None, None, None, None, "TRUE", 1000, None, None, 100, None, 20, None, None, "TRUE", 5, "FALSE")
#feature_name = leidingen_df.columns
#feature_name
leidingen_df.head(5)


# In[10]:


#leidingen_df[["id", "OBJECTID_1"]].head(5)

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.multiclass import type_of_target


label = leidingen_df["leeftijd"]
print(len(leidingen_df))
#label = label.tolist()
print(type(label))
 
feature_names = leidingen_df.columns.values
feature_names = feature_names.tolist()
print(feature_names)

features = leidingen_df[["VALUE_MIN_1_IMPUTED_1", "buiten_p", 
                         "ONEHOT_MATERIAAL_AC","ONEHOT_MATERIAAL_NGIJ","ONEHOT_MATERIAAL_PVC","ONEHOT_MATERIAAL_PVC_BVB",
                         "ONEHOT_MATERIAAL_ST","SLOPE_1_IMPUTED","address_num"]]
x = features
y = label


print(y)
print(np.shape(x),  np.shape(y))
#leidingen_df.notnull()
print(type_of_target(x),  type_of_target(y))    


from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def bayesian_optimization(dataset, function, parameters):
    x, y = dataset
    n_iterations = 5
    gp_params = {"alpha": 1e-5}

    BO = BayesianOptimization(function, parameters,random_state=1)
    BO_m = BO.maximize(n_iter=n_iterations)
    print(BO_m)
    print(BO.max)
    return BO_m

def rfc_optimization(cv):
    global x, y
    def function(n_estimators, max_depth, min_samples_leaf):
        return cross_val_score(
               RandomForestClassifier(
                   n_estimators=int(max(n_estimators,0)),                                                               
                   max_depth=int(max(max_depth,1)),
                   min_samples_leaf=int(max(min_samples_leaf,2)), 
                   n_jobs=-1, 
                   random_state = 2,   
                   class_weight="balanced"),  
               X=x, 
               y=y, 
               cv=cv,
               scoring="r2",
               n_jobs=-1).mean()
     
    parameters = {"n_estimators": (500, 1000),
                  "max_depth": (1, 100),
                  "min_samples_leaf": (1, 5)}
    
    return function, parameters



dataset = (x,y)
cv = KFold(n_splits=5, shuffle=True, random_state = 0)
function, parameters = rfc_optimization(cv)  
best_solution = bayesian_optimization(dataset, function, parameters)   
best_solution
 



@Misc{,
    author = {Fernando Nogueira},
    title = {{Bayesian Optimization}: Open source constrained global optimization tool for {Python}},
    year = {2014--},
    url = " https://github.com/fmfn/BayesianOptimization"
}

 




