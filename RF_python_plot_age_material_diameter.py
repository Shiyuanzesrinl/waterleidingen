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


leidingen_item = r'C:\Internship_work\WML_TraceNetwork\WML_strengenanalyse\Default.gdb\waterleidingen_new'
leidingen_df = pd.DataFrame.spatial.from_featureclass(leidingen_item)
leidingen_df = leidingen_df[leidingen_df.Problem.notnull()]
print(len(leidingen_df))
#leidingen_item_f1 = leidingen_item[0]
leidingen_df.head(5)


# In[5]:


#tree = arcpy.stats.Forest("TRAIN", leidingen_item, "Problem", "CATEGORICAL","VALUE_MAX_IMPUTED false;address_num false;Year_afsluiters false;loam_new true;ONEHOT_weg_functi_others true", 
#   None, None, None, None, None, None, None, None, None, None, "TRUE", 10, 10, 10, 100, 10, 0, None, None, "TRUE", 5, "FALSE")

#feature_name = leidingen_df.columns
#feature_name
label = leidingen_df["Problem"]
feature_names = leidingen_df.columns.values
feature_names = feature_names.tolist()
print(feature_names)


# In[21]:


#leidingen_df[["id", "OBJECTID_1"]].head(5)
import numpy as np
#! pip install imbalanced-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.multiclass import type_of_target
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt



material_list = [ 'ONEHOT_MATERIAAL_AC', 'ONEHOT_MATERIAAL_Beton', 'ONEHOT_MATERIAAL_Bonna', 'ONEHOT_MATERIAAL_CU', 'ONEHOT_MATERIAAL_GR_GIJ', 'ONEHOT_MATERIAAL_GVK', 'ONEHOT_MATERIAAL_HDPE', 'ONEHOT_MATERIAAL_NGIJ', 'ONEHOT_MATERIAAL_Onbekend', 'ONEHOT_MATERIAAL_PE', 'ONEHOT_MATERIAAL_PVC', 'ONEHOT_MATERIAAL_PVC_BVB', 'ONEHOT_MATERIAAL_RVS', 'ONEHOT_MATERIAAL_ST']

def plot_f(material):
    global leidingen_df
    
    y_num_list = []
    y_list = []
#    year_list = []
    
    years = leidingen_df["DATUM_AANL"].values    
    year_list = set(years.tolist())
#    print(len(years), len(year_list))    
    for year in year_list:
          num = len(material[material["DATUM_AANL"]==year])
          print(num)
          y_num_list.append(num)
          y_list.append(year)

    return y_num_list, y_list   
        
for i in material_list:
    material = leidingen_df[leidingen_df[i]==1]
    num_list, y_list = plot_f(material)
    plt.rcParams["figure.figsize"] = [12,6]
    plt.plot(y_list, num_list, label = i)
plt.legend(fontsize=8)
 
plt.savefig('C:\Internship_work\material_age.png')
plt.show()


# In[46]:





# In[ ]:




