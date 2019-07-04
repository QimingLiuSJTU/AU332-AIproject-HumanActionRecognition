# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:35:15 2018

@author: Administrator
"""

import cv2
import os
import random

def get_path(path):
    path_list = os.listdir(path)
    path_list.sort()
    full_path_list = []
    for filename in path_list:
        full_path_list.append(os.path.join(path,filename))
    return full_path_list, len(full_path_list)

# if the video is so long, we should find the frames that contains key actions
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

# cut vedio into frames
def cut_video(path, path_folder_naming):
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

    cap.release()
    return path_folder_naming


if __name__ == '__main__':
    folder_path = r'C:\Users\Administrator\Desktop\AI_Project\dataset\Florence_3d_actions'
    video_path_list, video_num = get_path(folder_path)
    path_folder_naming = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    for num in range(video_num):
        print('Now slice video', num + 1)
        path_folder_naming = cut_video(video_path_list[num], path_folder_naming)
        print(path_folder_naming)