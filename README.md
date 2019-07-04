# AU332-AIproject-HumanActionRecognition
Use kinect and openpose to output the associate action label of a person in a video stream.

## Description
  This is a project for Class AU332 in Shanghai Jiao Tong University. Human behavior recognition has so many applications such as visual surveillance, human-computer interfaces, content based video retrieval etc. It is a challenging research area because the dynamic human body motions have unlimited underlying representations. Our project goal is to construct discriminative underlying human action representations, make computer be aware of human action and recognize human behaviors in various scenarios effectively and effciently.<br>
  
  We used Kinect and [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to complete this project. The Kinect collects video streams and send them into openpose model, which can give back key points of human bones in RGB images. The combination of key points in several frames can represent the shape of a body and its changes. We designed about 8 actions and created a dataset including ~2000 samples, sent them into neural network, and got the trained model parameters. The final testing accuracy reaches 90%+.

## File Functions
**openpose_preprocessing/openpose_test_for_one_frame.py**: Test file, can process a single image and output the result of openpose, used to view the effect<br>
**openpose_preprocessing/cut_video.py**: The frame extraction file, used to obtain the video frame, which has been integrated into the openpose processing file.<br>
**openpose_preprocessing/preprocessing_using_openpose.py**: File for video frame extraction and openpose preprocessing of raw datasets.<br>
**openpose_preprocessing/preprocessing_own_dataset_using_openpose.py**: File for openpose preprocessing of our self-built dataset.<br>
**PyKinect_action_record/record.py**: This code is used to collect dataset of RGB images and depth data.(run: python record.py)<br>
**PyKinect_action_record/PyKinectBodyGame.py**: This code is the real-time action recognition game.(run: python PyKinectBodyGame.py)<br>
**net_and_train/version3/C2D_model.py**: Construct the 2D convolution net.<br>
**net_and_train/version3/skeleton_depth_model.py**: Construct the 2D and 3D convolution net using both skeleton key points and depth images.<br>
**net_and_train/version3/predict.py**: Train the 2D concolution net and save the model.<br>
**net_and_train/version3/PyKinectBodyGame.py**: Real time use of Kinect and net.pkl to predict actions (performs well).<br>
**net_and_train/version3/GradientBoosting.py**: Use GradientBoosting and save the model.<br>
**net_and_train/version3/RandomForest.py**: Use RandomForest.<br>
**net_and_train/version3/PyKinectBodyGame2.py**: Real time use of Kinect and new_net.pkl to predict actions(performs badly).<br>
**net_and_train/version3/net.pkl**: The model saved by predict.py, after using neural network.<br>
**net_and_train/version3/new_net.pkl**: The model saved by GradientBoosting.py<br>
**net_and_train/version3/sample_big.npy**: Samples that include 10 actions<br>
**net_and_train/version3/labels_big.npy**: Labels that include 10 actions<br>

## Dataset
Actions: Standing Still, Waving arms, Drinking, Applaud, Sitting, etc...(See Docs)<br>
The original dataset is too big, we extract them into key points position using openpose. See **sample_big.npy** and **labels_big.npy**.
