# ISL-to-Text-Converter  
Communication is a basic requirement of an individual to exchange feelings, thoughts, and ideas, but the hearing and speech impaired community finds it difficult to interact with the vast majority of people. Sign language facilitates communication between the hearing and speech impaired person and the rest of society. The Rights of Persons with Disabilities (*RPWD*) Act, 2016, was also passed by the Indian government, which acknowledges Indian Sign Language (*ISL*) and mandates the use of sign language interpreters in all government-aided organizations and the public sector proceedings. Unfortunately, a large percentage of the Indian population is not familiar with the semantics of the gestures associated with ISL.  
To bridge this communication gap, this project proposes a model to identify and classify Indian Sign Language gestures in real-time using Convolutional Neural Networks (*CNN*). The model has been developed using OpenCV and Keras implementation of CNNs and aims to classify 36 ISL gestures representing 0-9 numbers and A-Z alphabets by converting them to their text equivalents. The dataset created and used consists of 300 images for each gesture which were fed into the CNN model for training and testing purposes.  
The proposed CNN model was successfully implemented and achieved 98.68% training and 100% validation accuracy on the training images. On testing the model on test data, an accuracy score of 99.91% was obtained.  

This repository contains the following files:
1. **Data**  
    - *create_gesture_data.py*  
    Code for creating and saving the dataset.
    - *load_data.py*  
    Code to load the dataset.
2. **Models**  
    - *new_model.py*  
    Revised CNN model
    - *old_model.py*  
    Initial CNN model used.
3. **Test**  
    - *plot_model_performance.py*  
    Code to plot model loss and model accuracy curves.
    - *test.py*  
    Code to test the model on test data and print classification report and confusion matrix.
4. **Live camera detection**
    - *gesture_detection.py*  
    Revised code to detect and classify ISL gestures in a live feed from a webcam.
    - *live_feed_detection.py*  
    Code to detect and classify ISL gestures in a live feed from a webcam.
5. **README**


Here is the link to the ISL Dataset used in this project:  
https://drive.google.com/drive/folders/1iyAh6-6vxK9WGvVMXwXl3oEmPNGlRkAe?usp=sharing
