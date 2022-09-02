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

def dele():
    
    model_name=input("Name Of the model you want to delete: ")
    answer = input("Delete: "+ model_name+"? \n Press Y to confirm, N to Deny: ")
    if answer == "Y" or answer == "y": 
        os.remove('../models/'+ model_name +'.h5')  
    elif  answer == "N" or answer == "n": 
        os.system('cls') 
        exec(open('main.py').read()) 
    else:
        print("enter file name to delete")
        dele()

# In[5]:

dele()

answer2 = input("\n\n\nContinue deleting models? Y/N?")
if answer2 == "Y" or answer2 == "y": 
    exec(open('delete.py').read())               
elif  answer2 == "N" or answer2 == "n": 
    os.system('cls') 
    exec(open('main.py').read())


# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:




