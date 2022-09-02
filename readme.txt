###intro##
Real-Time Sign Language Translation is a project based on action detection. The project involved using LSTM deep learning models to train and predict American Sign Language (ASL) signs. 
Enabled custom signs training and evaluating for languages other than ASL.
Works with Tensorflow and Keras to build a deep neural network that leverages LSTM layers to handle the sequence of data (keypoints) collected using Mediapipe Holisitc Framework


###data paths###
The data is stored in MP_data(i.e all sequences collected using open-cv).
lstm models are saved as .h5 files in models after training and can be accessed from the same.

###Code###
1. Create: you can build and train your own model irrespective of the sign language used(asl is not mandatory).
2. existitng: A pre trained model reay to evaluad for 7 phrases is already available.

3. delete: delete moedls if needed