#!/usr/bin/env python
# coding: utf-8

# In[74]:


errors = [45.25,
48.13,
49.65,
45.88 , 
50.41,
48.38,
50.46,
48.52,
49.54,
50.86,
50.04,
54.14,
51.56,
52.03,
50.16,
49.07,
50.74,
45.31,
44.24,
44.49

]

 
#errors = sorted(errors,reverse=True)

def Average(lst):
    return sum(lst) / len(lst)
 
print(Average(errors), len(errors))


# In[75]:



standard = []
dif = 0
error_gain = []
for i in range(len(errors)):  
    print(errors[i])
    dif = abs(errors[i] - Average(errors)) + dif   
    variance = dif / (i+1)
    if abs(errors[i] - Average(errors)) > variance:
        print(f'the error gain: {abs(errors[i] - Average(errors))} is larger than average variation: {variance} obtained by adding noisy variables: {i+1}')
    error_gain.append(abs(errors[i] - Average(errors)))
    standard.append(variance)
        
print(standard, len(error_gain))        
              


# In[78]:


import matplotlib.pyplot as plt
  
# x axis values
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# corresponding y axis values
  
# plotting the points 
#plt.plot(x, standard, label = "threshold")
 

# plotting the line 2 points 
plt.rcParams["figure.figsize"] = [6,4.5]
plt.plot(x, errors, label = "OOB error")  
# naming the x axis
plt.xlabel('variable')
# naming the y axis
plt.ylabel('values of OOB error')
plt.legend()
plt.show()


# In[ ]:




