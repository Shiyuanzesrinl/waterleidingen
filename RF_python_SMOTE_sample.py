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
leidingen_df = leidingen_df[leidingen_df.Problem.notnull()]
print(len(leidingen_df))
#leidingen_item_f1 = leidingen_item[0]
leidingen_df.head(5)
#points_item = r'C:\Internship_work\Model\model_input.gdb\random_poins_Limburg_new'
#point_df = pd.DataFrame.spatial.from_featureclass(points_item)


# In[4]:


#tree = arcpy.stats.Forest("TRAIN", leidingen_item, "Problem", "CATEGORICAL","VALUE_MAX_IMPUTED false;address_num false;Year_afsluiters false;loam_new true;ONEHOT_weg_functi_others true", 
#   None, None, None, None, None, None, None, None, None, None, "TRUE", 10, 10, 10, 100, 10, 0, None, None, "TRUE", 5, "FALSE")

#feature_name = leidingen_df.columns
#feature_name
feature_names = leidingen_df.columns.values
feature_names = feature_names.tolist()
print(feature_names)


# In[7]:


#leidingen_df[["id", "OBJECTID_1"]].head(5)

import numpy as np
#! pip install imbalanced-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.multiclass import type_of_target
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot


label = leidingen_df["Problem"]
 
features = leidingen_df[["address_num","bomen_num","Year_afsluiters","num_aftakking","num_bo","verloop_num","Is_hilly","loam_new","buiten_p","sand_and_clay_new","ONEHOT_weg_functi_others","ONEHOT_MATERIAAL_AC","ONEHOT_weg_functi_fiet","ONEHOT_weg_functi_parkarea"]]

oversample = SMOTE()
x, y = oversample.fit_resample(features, label)
x_train = x
y_train = y
print(len(y), len(x))
 
leidingen_new_df = pd.DataFrame(x)
#column_shape = point_df["SHAPE"]
 
#print(column_shape.head(5))

#for i in range(len(y)):
    
leidingen_new_df['Problem'] = y.values
     
print(leidingen_new_df['Problem'])


# In[8]:


#leidingen_new_df.spatial.to_featureclass(location=r"c:\output_examples\waterleidingen_new_3.csv")  
print(leidingen_new_df[leidingen_new_df['Problem'] == 1])
leidingen_new_df.to_csv('C:\Internship_work\waterleidingen_new_1.csv')





