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




 # Call switch() method



def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results





def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections





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


# In[7]:

def init():
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
            cv2.putText(image, 'Raise your left hand for intialisation & press q',(60,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            #resizing the cv feed
            cv2.namedWindow("intialize", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("intialize", 1280, 720)
            
            # Show to screen   
            cv2.imshow('intialize', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


# In[ ]:





# In[ ]:

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])




# In[8]:




# In[27]:



DATA_PATH = os.path.join('../MP_Data/Phrases') 

 # Thirty videos worth of data
no_sequences = 30
 # Videos are going to be 30 frames in length
sequence_length = 30
actions_new = np.array([])


# In[28]:


def create(model_to_save):
    # Path for exported data, numpy arrays
    init()

    # Actions that we try to detect
    #demo data1'Emergency','Hello','Fine','wait'
    #demo data2 'deaf','finish','help','good morning'


    x = 0
    n = int(input("Enter number of actions < 8 : "))
    x = x+1
    if n==0:
        print("Enter a number greater than 0 ")
        n = int(input("Enter number of actions < 8 : "))
    else:    
        for i in range(0, n):
            ele = (input("Name of the signs: "))
            #appending to the names file
            # Filename to write
            filename =("Names/"+ model_to_save + ".names")
            
            global actions_new 
            actions_new = np.append(actions_new,ele)
            
            # Open the file with writing permission
            myfile = open(filename, 'a')

            # Write a line to the file
            if x>1 or i>0:
                myfile.write("\n"+ele)
            else:
                myfile.write(ele)
            # Close the file
            myfile.close()
        #crete path
        for action in actions_new: 
            for sequence in range(no_sequences):
                try: 
                    os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
                except:
                    pass


    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic( min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        # NEW LOOP
        for action in actions_new:
        # Loop through actions
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):

                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    print(results)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)

                    # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (60,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        #resizing the cv feed
                        cv2.namedWindow("OpenCV Feed", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("OpenCV Feed", 1280, 720)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(3000)
                        
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),(60,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                      
                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

        
        cap.release()
        cv2.destroyAllWindows()
        
    trainm(model_to_save)

    


# In[29]:


def trainm(model_to_save):    
    
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical

    label_map = {label:num for num, label in enumerate(actions_new)}


    sequences= []
    labels = []
    for action in actions_new:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])


    np.array(sequences).shape

    np.array(labels).shape

    X = np.array(sequences)


    X.shape


    y = to_categorical(labels).astype(int)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    y_test.shape
    # 
    # ## 7. Build and Train LSTM Neural Network

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import TensorBoard

    log_dir = os.path.join('Logs')
    print("\n\n\n your model will now bw trained please wait \n\n\n")

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(actions_new.shape[0], activation='softmax')) 

    
 
    ACCURACY_THRESHOLD = 0.96
    LOSS_THRESHOLD= 0.16
    # Implement callback function to stop training
    # when accuracy reaches e.g. ACCURACY_THRESHOLD = 0.95
    class myCallback(tensorflow.keras.callbacks.Callback): 
        def on_epoch_end(self, epoch, logs={}): 
            if (logs.get('categorical_accuracy') > ACCURACY_THRESHOLD) and (logs.get('loss') < LOSS_THRESHOLD):  
                print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))   
                self.model.stop_training = True


      


    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=2000, callbacks = myCallback())

    model.summary()

    # ## 8. Make Predictions


    res = model.predict(X_test)

    actions_new[np.argmax(res[0])]

    actions_new[np.argmax(y_test[0])]

    # ## 9. Save Weights

    model.save('../models/'+ model_to_save +'.h5')
    
    answer = input("Do you want to add more signs? (Y/N): ") 
    if answer == "Y" or answer == "y": 
        create()      
    elif  answer == "N" or answer == "n": 
        print("Your Model Will Now Be Trained Please Wait.")
        pass

    else:
        print("Please Choose Y(yes)/N(no).")


# In[ ]:


def switch():
    # This will guide the user to choose option
    print("1: Create your own model\n2: Exiting models\n3: delete\n4: Exit")
 
# This will take option from user    
    option = int(input(" your option : "))
# Create your own sign
    def create_model():
        os.system('cls')
        model_to_save=input("What would you like to name the model? ")
        create(model_to_save)
        switch()

# use existing signs
    def existing():
        os.system('cls')
        exec(open('preditc.py').read())
        switch()
    def delete():
        os.system('cls')
        exec(open('delete.py').read())
        switch()
        
    def Exit():
        exit()
# If user enters invalid option then this method will be called 
    def default():
        print("Incorrect option")
        switch()
        
# Dictionary Mapping
    dict = {
        1 : create_model,
        2 : existing,
        3 : delete,
        4 : Exit,      
    }
    dict.get(option,default)() # get() method returns the function matching the argument

switch()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




