# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 22:15:45 2018

@author: Liuqiming
"""

import cv2
import os
import random
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

def get_path(path):
    path_list = os.listdir(path)
    path_list.sort()
    full_path_list = []
    for filename in path_list:
        full_path_list.append(os.path.join(path,filename))
    return full_path_list, len(full_path_list)


def get_frame_list(frame_number, get_number):
    if frame_number < get_number:
        flist = list(range(frame_number))
        for k in range(get_number - frame_number):
            flist.append(frame_number - 1)
    elif frame_number < 1.3 * get_number:
        flist = []
        for k in range(get_number):
            flist.insert(0, range(frame_number)[-1 - k])
    else:
        frame_list = list(range(round(frame_number * 0.18), frame_number))
        flist = random.sample(frame_list, get_number)
        flist.sort()
    
    return flist

# cut the video into frames and implement openpose
def cut_video(path, path_folder_naming, arr_mat, arr_lab):
    protoFile = r'C:\Users\Administrator\Desktop\AI_Project\openpose-master\models\pose\mpi\pose_deploy_linevec.prototxt'
    weightsFile = r'C:\Users\Administrator\Desktop\AI_Project\openpose-master\models\pose\mpi\pose_iter_160000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    temp_list_x = []
    temp_list_y = []
    
    cap = cv2.VideoCapture(path)
    category = eval(path[-5])
    path_folder_naming[category - 1] += 1
    naming_num = path_folder_naming[category - 1]
    writing_folder = 'C:\\Users\\Administrator\\Desktop\\AI_Project\\dataset\\' + str(category) + '\\' + str(naming_num) + '\\'
    os.mkdir(writing_folder)
    
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    get_number = 8
    frame_list = get_frame_list(total_frame, get_number)
        
    
    if cap.isOpened():
        print('video successfully opened')
        success = True
               
    for i in range(get_number):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_list[i])
        success, frame = cap.read()
        
        print('Reading a new frame: ', success)
        if success:
            cv2.imwrite(writing_folder + "frame" + "_%d.jpg" % i, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            temp_list_x, temp_list_y = get_key_point_from_frame(frame, net, temp_list_x, temp_list_y)

    temp_list_x.extend(temp_list_y)
    arr_mat.append(temp_list_x)
    arr_lab.append(category)

    cap.release()
    return path_folder_naming, arr_mat, arr_lab


def get_key_point_from_frame(frame, net, temp_list_x, temp_list_y):
       
    # Specify number of points in the model 
    nPoints = 15
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inWidth = im.shape[1]
    inHeight = im.shape[0]
    
    # Convert image to blob
    netInputSize = (368, 368)
    inpBlob = cv2.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
    net.setInput(inpBlob)

    # Run Inference (forward pass)
    output = net.forward()
    scaleX = float(inWidth) / output.shape[3]
    scaleY = float(inHeight) / output.shape[2]

    points = []
    # Confidence treshold 
    threshold = 0.1

    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
        # Scale the point to fit on the original image
        x = scaleX * point[0]
        y = scaleY * point[1]

        if prob > threshold : 
            points.append((int(x), int(y)))
            temp_list_x.append(int(x))
            temp_list_y.append(int(y))
        else :
            points.append(None)
            temp_list_x.append(None)
            temp_list_y.append(None)
    
    return temp_list_x, temp_list_y
    
if __name__ == '__main__':
    folder_path = r'C:\Users\Administrator\Desktop\AI_Project\dataset\Florence_3d_actions'
    video_path_list, video_num = get_path(folder_path)
    path_folder_naming = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    arr_mat = []
    arr_lab = []
    
    for num in range(video_num):
        print('Now slice video', num + 1)
        path_folder_naming, arr_mat, arr_lab = cut_video(video_path_list[num], path_folder_naming, arr_mat, arr_lab)
        print(path_folder_naming)
        
        if num == 100:
            # print(arr_mat)
            print(np.array(arr_mat).shape)
            print(np.array(arr_lab).shape)
            
            np.save('sample_' + str(num) + '.npy', np.array(arr_mat))
            np.save('label_' + str(num) + '.npy', np.array(arr_lab))
    
    arr_mat = np.array(arr_mat)
    arr_lab = np.array(arr_lab)
    np.save('sample.npy', arr_mat)
    np.save('label.npy', arr_lab)