#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow
mp_holistic =  mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


# In[5]:
os.system('CLS')

def switch():
 
# This will guide the user to choose option
    print("1: Create your own model\n2: Exiting models\n3: delete\n4: Exit")
 
# This will take option from user    
    option = int(input(" your option : "))
# Create your own sign
    def create():
        exec(open('create.py').read())
        switch()
# use existing signs
    def existing():
        exec(open('preditc.py').read())
        execfile()
        switch()
    def delete():
        exec(open('delete.py').read())
        switch()
        
    def Exit():
        exit()
# If user enters invalid option then this method will be called 
    def default():
        print("Incorrect option")

# Dictionary Mapping
    dict = {
        1 : create,
        2 : existing,
        3 : delete,
        4 : Exit,      
    }
    dict.get(option,default)() # get() method returns the function matching the argument

switch() # Call switch() method


# In[ ]:





# In[ ]:




