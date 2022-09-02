#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow


# In[3]:


MODELS_PATH = os.path.join("../models")
print("available models: \n")
for file in os.listdir(MODELS_PATH):
    print(file)


# In[4]:


model_name=input("Name Of the model you want to delete: ")


# In[5]:


answer = input("Delete: "+ model_name+"? \n Press Y to confirm, N to Deny: ")
if answer == "Y" or answer == "y": 
    os.remove('../models/'+ model_name +'.h5')  
    answer2 = input("\n\n\nContinue deleting models? Y/N?")
    if answer2 == "Y" or answer == "y": 
        exec(open('delete.py').read())               
    elif  answer2 == "N" or answer == "n": 
        exec(open('main.py').read())
    else:
        print("Please Choose Y(yes)/N(no).")
elif  answer == "N" or answer == "n": 
    exec(open('main.py').read())
else:
    print("Please Choose Y(yes)/N(no).")

# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:




