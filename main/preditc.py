#!/usr/bin/env python
# coding: utf-8

# ## 1.importing dependencies

# In[110]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf
import pandas


# ## 2.Keypoints using MP Holistic

# In[111]:

              


mp_holistic =  mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


# In[112]:


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


# In[113]:


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


# In[114]:


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 


# In[115]:


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# ## 3.Initialising Actions Array

# In[116]:


MODELS_PATH = os.path.join("../models")
print("available models: ")
for file in os.listdir(MODELS_PATH):
    print(file)


# In[117]:


model_to_load = input(" Which model do you want to use : (filename) ")
print("please wait for the model to load")

# In[124]:


# Path for exported data, numpy arrays
actions = np.array([])
DATA_PATH = os.path.join("../MP_Data/Phrases")
f = open('Names/'+ model_to_load + '.names', 'r')  
actions = f.read().split('\n')
f.close()

no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30


# In[ ]:





# ## 4. Preprocess data and Create labels and features

# In[125]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[126]:


label_map = {label:num for num, label in enumerate(actions)}


# In[127]:


label_map


# In[109]:


sequences= []
labels = []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


# In[87]:


X = np.array(sequences)


# In[88]:


y = to_categorical(labels).astype(int)


# In[89]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# In[ ]:





# ## 5. Loading the trained model

# In[90]:


MODELS_PATH = os.path.join("../models")


# In[91]:





# In[92]:


model = tf.keras.models.load_model("../models/" + model_to_load + ".h5")  


# In[ ]:





# ## 6. Evaluation using confusion matrix and accuracy

# In[93]:


from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# In[94]:


yhat = model.predict(X_test)


# In[95]:


ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


# In[96]:


accuracy_score(ytrue, yhat)


# In[ ]:





# ## 7. Test in realtime

# In[97]:


from scipy import stats


# In[98]:


colors = [(245,117,16), (117,245,16), (16,117,245),(245,145,145), (163,255,255), (0,204,204), (140,140,140)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


# In[99]:


# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] !=sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]
                
         # Viz probabilities
            image = prob_viz(res, actions, image, colors)
        
      
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.namedWindow("Preditions", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preditions", 1280, 720)
        
        cv2.imshow('Preditions', image)
        # set your desired size
        
       

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:
answer = input("\n\n\nContinue testing models? Y/N?")
if answer == "Y" or answer == "y": 
    exec(open('preditc.py').read())                
elif  answer == "N" or answer == "n": 
    os.system('cls') 
    exec(open('main.py').read())
else:
    print("Please Choose Y(yes)/N(no).")
