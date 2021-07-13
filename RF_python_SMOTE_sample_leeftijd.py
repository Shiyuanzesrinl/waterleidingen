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
#leidingen_df = leidingen_df[leidingen_df.leeftijd_r.notnull()]
print(len(leidingen_df))
#leidingen_item_f1 = leidingen_item[0]
leidingen_df.head(5)
#points_item = r'C:\Internship_work\Model\model_input.gdb\random_poins_Limburg_new'
#point_df = pd.DataFrame.spatial.from_featureclass(points_item)


# In[4]:


leidingen_df = leidingen_df[leidingen_df.leeftijd_r.notnull()]
print(len(leidingen_df))
#leidingen_item_f1 = leidingen_item[0]
leidingen_df.head(5)


# In[5]:


#tree = arcpy.stats.Forest("TRAIN", leidingen_item, "Problem", "CATEGORICAL","VALUE_MAX_IMPUTED false;address_num false;Year_afsluiters false;loam_new true;ONEHOT_weg_functi_others true", 
#   None, None, None, None, None, None, None, None, None, None, "TRUE", 10, 10, 10, 100, 10, 0, None, None, "TRUE", 5, "FALSE")

#feature_name = leidingen_df.columns
#feature_name
feature_names = leidingen_df.columns.values
feature_names = feature_names.tolist()
print(feature_names)


# In[6]:


#leidingen_df[["id", "OBJECTID_1"]].head(5)

import numpy as np
#! pip install imbalanced-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.multiclass import type_of_target
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot


label = leidingen_df["leeftijd_r"]
 
features = leidingen_df[["buiten_p", "Grootte", "bomen_num", "Year_afsluiters", "num_bo","verloop_num",
                         "clay_new","human_new","loam_new","Deciduous_tree","overgang_num", "ONEHOT_weg_functi_voet","ONEHOT_MATERIAAL_PVC",
                         "ONEHOT_MATERIAAL_PVC_BVB", "ONEHOT_MATERIAAL_ST", "SLOPE_1_IMPUTED","address_num",
                         "ONEHOT_MATERIAAL_AC","ONEHOT_MATERIAAL_NGIJ","VALUE_MIN_1_IMPUTED_1"]]

oversample = SMOTE()
x, y = oversample.fit_resample(features, label)
x_train = x
y_train = y
print(len(y), len(x))
 
leidingen_new_df = pd.DataFrame(x)
#column_shape = point_df["SHAPE"]
 
#print(column_shape.head(5))

#for i in range(len(y)):
    
leidingen_new_df['leeftijd_r'] = y.values
     
print(leidingen_new_df['leeftijd_r'])


# In[7]:


#leidingen_new_df.spatial.to_featureclass(location=r"c:\output_examples\waterleidingen_new_3.csv")  
print(leidingen_new_df[leidingen_new_df['leeftijd_r'] == 1])
leidingen_new_df.to_csv('C:\Internship_work\leeftijd_r.csv')


# In[8]:


#! pip install bayesian-optimization
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

def bayesian_optimization(dataset, function, parameters):
    x_train, y_train = dataset
    n_iterations = 5
#    gp_params = {"alpha": 1e-5}

    BO = BayesianOptimization(function, parameters,random_state=1)
    BO_m = BO.maximize(n_iter=n_iterations)
#    print(BO_m)
    print(BO.max)
    return BO_m

def rfc_optimization(cv):
    global x_train, y_train
    def function(n_estimators, max_depth, min_samples_leaf):
        return cross_val_score(
               RandomForestClassifier(
                   n_estimators=int(max(n_estimators,0)),                                                               
                   max_depth=int(max(max_depth,1)),
                   min_samples_leaf=int(max(min_samples_leaf,2)), 
                   n_jobs=-1, 
                   random_state = 0,   
                   class_weight="balanced"),  
               X=x_train, 
               y=y_train, 
               cv=cv,
               scoring="roc_auc",
               n_jobs=-1).mean()
     
    parameters = {"n_estimators": (100, 1000),
                  "max_depth": (1, 150),
                  "min_samples_leaf": (1, 5)}
    
    return function, parameters
 
        
dataset = (x_train, y_train)
cv = KFold(n_splits=5, shuffle=True, random_state = 0)
function, parameters = rfc_optimization(cv)  
best_solution = bayesian_optimization(dataset, function, parameters)   
#best_solution
 
 


# In[ ]:


#arcpy.stats.Forest("PREDICT_FEATURES", leidingen_item, "Problem", "CATEGORICAL", "VALUE_MAX_IMPUTED false;address_num false;Year_afsluiters false;loam_new true;ONEHOT_weg_functi_others true", None, None, waterleidingen_others, r"C:\Internship_work\Model\model_output.gdb\Prediction_pro_notebook_test1", None, "VALUE_MAX_IMPUTED VALUE_MAX_IMPUTED;address_num address_num;Year_afsluiters Year;loam_new loam_new;ONEHOT_weg_functi_others 'others (weg_functi_One-hot)'", None, None, None, None, "TRUE", 457, 2, 29, 100, None, 10, None, None, "FALSE", 1, "FALSE")


# In[ ]:


#find importance of variables starting with the first variable as initial variable 
#i = 7 

#sub_name_list = []
 
#for feature_name in feature_names[i:60]:
#    sub_name_list.append(feature_name)
    
#subset = leidingen_df[leidingen_df.columns[i:60]]
#print(len(feature_names))
#subset.drop(columns='leeftij_r')
 

#for c in range(i, 60):
#    feature = subset[subset.columns[7:i+1]]
#    print(feature_set)
#    print(feature.shape)
#    i = i+1
#    train_features, test_features, train_labels, test_labels = train_test_split(feature, label, test_size = 0.20, random_state = 42)
#    print('Training Features Shape:', train_features.shape)
#    print('Training Labels Shape:', train_labels.shape)
#    print('Testing Features Shape:', test_features.shape)
#    print('Testing Labels Shape:', test_labels.shape)
    
    # Instantiate model with 1000 decision trees
#    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
#    rf.fit(train_features, train_labels, sample_weight=None)
    
    # Get numerical feature importances
#    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
#    predictions = rf.predict(test_features)
#    f1  =  f1_score(test_labels, predictions.round(), average='micro')
#    feature_f1 = [(feature, round(f1, 2)) for feature, f1 in zip(sub_name_list, f1)]
#    print(f'Variable: {feature.columns.values} f1: {f1}')  


# In[ ]:





# In[ ]:




