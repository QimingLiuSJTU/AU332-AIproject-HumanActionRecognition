# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 19:24:34 2018

@author: Administrator
"""

import cv2
import os
import numpy as np

def get_key_point_from_frame(frame_path, net, temp_list_x, temp_list_y):
       
    # Specify number of points in the model 
    nPoints = 15
        
    im = cv2.imread(frame_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    inWidth = im.shape[1]
    inHeight = im.shape[0]
    
    # Convert image to blob
    netInputSize = (368, 368)
    inpBlob = cv2.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
    net.setInput(inpBlob)

    # Run Inference (forward pass)
    output = net.forward()
    
    # X and Y Scale
    scaleX = float(inWidth) / output.shape[3]
    scaleY = float(inHeight) / output.shape[2]

    # Empty list to store the detected keypoints
    points = []

    # Confidence treshold 
    threshold = 0.1

    for i in range(nPoints):
        # Obtain probability map
        probMap = output[0, i, :, :]
    
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
        # Scale the point to fit on the original image
        x = scaleX * point[0]
        y = scaleY * point[1]

        if prob > threshold : 
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
            temp_list_x.append(int(x))
            temp_list_y.append(int(y))
        else :
            points.append(None)
            temp_list_x.append(None)
            temp_list_y.append(None)
    
    return temp_list_x, temp_list_y


if __name__ == '__main__':
    protoFile = r'C:\Users\Administrator\Desktop\AI_Project\openpose-master\models\pose\mpi\pose_deploy_linevec.prototxt'
    weightsFile = r'C:\Users\Administrator\Desktop\AI_Project\openpose-master\models\pose\mpi\pose_iter_160000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    path = r'E:\ai_data\new_video\new_video'

    arr_mat = []
    arr_lab = []

    for lists in os.listdir(path): # open folder
        sub_path_1 = os.path.join(path, lists)
        for lists2 in os.listdir(sub_path_1): # open folder
            sub_path_2 = os.path.join(sub_path_1, lists2)
            for lists3 in os.listdir(sub_path_2): # open folder
                sub_path_3 = os.path.join(sub_path_2, lists3)
                print(sub_path_3)
                temp_list_x = []
                temp_list_y = []
                ii = 0
                while ii <= 15:
                    filename = os.path.join(sub_path_3, str(ii) + '.jpg')
                    ii += 1
                    print(filename)
                    if filename[-1] == 'g':
                        temp_list_x, temp_list_y = get_key_point_from_frame(filename, net, temp_list_x, temp_list_y)
                temp_list_x.extend(temp_list_y)
                arr_mat.append(temp_list_x)
                arr_lab.append(int(lists))
                print(len(arr_mat))
                print(arr_lab)
            
            print(np.array(arr_mat).shape)
            print(np.array(arr_lab).shape)
            # save in the process in case of accident
            np.save('sample_add_' + str(lists) + '_' + str(lists2) + '.npy', np.array(arr_mat))
            np.save('label_add_' + str(lists) + '_' + str(lists2) + '.npy', np.array(arr_lab))
            
    arr_mat = np.array(arr_mat)
    arr_lab = np.array(arr_lab)
    np.save('final_add_sample.npy', arr_mat)
    np.save('final_add_label.npy', arr_lab)