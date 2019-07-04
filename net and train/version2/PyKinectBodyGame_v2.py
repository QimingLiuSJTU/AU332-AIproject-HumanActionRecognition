from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

from C2D_model import C2D
import torch
import torch.nn as nn
from torch.autograd import Variable

import ctypes
import _ctypes
import pygame
import sys
import math
import numpy as np
import cv2
import time
import torch

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# colors for drawing different bodies 
SKELETON_COLORS = [pygame.color.THECOLORS["red"], 
                  pygame.color.THECOLORS["blue"], 
                  pygame.color.THECOLORS["green"], 
                  pygame.color.THECOLORS["orange"], 
                  pygame.color.THECOLORS["purple"], 
                  pygame.color.THECOLORS["yellow"], 
                  pygame.color.THECOLORS["violet"]]


class BodyGameRuntime(object):
    def __init__(self):
        # initialize the game
        pygame.init()

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Action Recognition with Kinect")

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data 
        self._bodies = None

        # the number of bodies tracked
        self._body_num = 0

        # pose of user
        self._pose = ["stand", "stand"]

        #to get data
        self._is_get = [False, False]

        # data
        self._x = [[], []] # x coordiates of joints points
        self._y = [[], []] # y coordiates of joints points
        self.labels = ['wave', 'drink',"call",  "applause", "stand", "sit", "stand still"]

        # interval
        self._inter_range = 100
        self._interval = [0, 0]

        # build the net by loading pytorch model
        self.net = torch.load('net.pkl')
        print("model loaded")

    # show a bone on the screen
    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # get the endpoints
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)

        try:
            pygame.draw.line(self._frame_surface, color, start, end, 8)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

    # draw the total body on the screen
    def draw_body(self, joints, jointPoints, color):
        # Torso
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);
    
        # Right Arm    
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight);

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft);

        # Right Leg
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight);

        # Left Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft);


    # draw the box on the screen
    def draw_box(self, joints, jointPoints, color, number):
        '''
        function1: draw the box of body, according to the boundary of coordinates
        function2: store the data if it is time to sample
        '''
        X = []
        Y = []
        # calculate the interval from last sampling
        if number <= 1:
            self._interval[number] += self._clock.get_time()
            # if it is time to sample
            if self._interval[number] > self._inter_range:
                self._interval[number] = 0
                self._is_get[number] = True
        # [3] [20] [8] [9] [10] [4] [5] [6] [16] [17] [18] [12] [13] [14] [1]
        # the first 15 joints are the data we need to input to the model
        # they have the same order with training datasets
        points = [PyKinectV2.JointType_Head, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight,
                PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ShoulderLeft,
                PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HipRight,
                PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_HipLeft,
                PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_SpineMid,
                PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HandRight,
                PyKinectV2.JointType_HandTipRight, PyKinectV2.JointType_ThumbRight, PyKinectV2.JointType_HandLeft,
                PyKinectV2.JointType_HandTipLeft, PyKinectV2.JointType_ThumbLeft, PyKinectV2.JointType_FootRight,
                PyKinectV2.JointType_FootLeft]
        # initialize the boundary
        left_most = float('inf')
        right_most = 0
        up_most = float('inf')
        down_most = 0
        # traverse all the joints
        for No in range(25):
            pt = points[No]
            jointState = joints[pt].TrackingState
            # if the joint is not tracked(missing) and we need to get it
            # store the coordinates as None and continue
            if jointState == PyKinectV2.TrackingState_NotTracked or jointState == PyKinectV2.TrackingState_Inferred:
                if number <= 1:
                    if self._is_get[number] == True and No < 15:
                        X.append(None)
                        Y.append(None)
                    continue
            # update the boundary
            x = jointPoints[pt].x
            y = jointPoints[pt].y
            if x < left_most:
                left_most = x
            if x > right_most:
                right_most = x
            if y < up_most:
                up_most = y
            if y > down_most:
                down_most = y
            # if it is time to sample
            if number <= 1:
                if self._is_get == True and No < 15:
                    # if out of range, append None
                    if x < 0 or x > 1920:
                        X.append(None)
                    else:
                        # change the size of coordinates and store
                        X.append(600 - int(x * 600 / 1920))
                    if y < 0 or y > 1080:
                        Y.append(None)
                    else:
                        Y.append(int(y * 480 / 1080))                
                
        # append the data of this frame to the total list
        if self._is_get and number <= 1:
            if len(X)==15 and len(Y)==15:
                self._x[number] += X
                self._y[number] += Y
            # if data is not complete, sample next frame
            # but I think it is not gonna happen
            else:
                self._interval[number] = self._inter_range
        height = down_most - up_most
        width = right_most - left_most

        # since the joints is not the strict boundary
        # change the coordinates
        corner1 = (left_most - 0.7*math.sqrt(abs(width)), up_most - 0.15*height)
        corner2 = (right_most + 0.7*math.sqrt(abs(width)), up_most - 0.15*height)
        corner3 = (right_most + 0.7*math.sqrt(abs(width)), down_most + 0.15*height)
        corner4 = (left_most - 0.7*math.sqrt(abs(width)), down_most + 0.15*height)
        # draw the box
        try:
            pygame.draw.line(self._frame_surface, color, corner1, corner2, 8)
            pygame.draw.line(self._frame_surface, color, corner2, corner3, 8)
            pygame.draw.line(self._frame_surface, color, corner3, corner4, 8)
            pygame.draw.line(self._frame_surface, color, corner4, corner1, 8)

        except: # need to catch it due to possible invalid positions (with inf)
            pass

    # filter the None data
    '''
    if we get None in draw_box()
    set them with approximate values
    '''
    def filter_none_data(self, features):
        for i in range(len(features)):
            # set None to 0
            non_pos = []
            for j in range(len(features[i])):
                if str(features[i][j]) == str(None):
                    non_pos.append(j)
                    features[i][j] = 0
            # if it is 0, set it with average of neighbers
            for k in range(10): #value iteration times
                for m in range(len(non_pos)):
                    if non_pos[m]%15 == 0:  #begin position
                        features[i][non_pos[m]] = features[i][non_pos[m]+1]
                    elif (non_pos[m]+1) % 15 == 0: #end position
                        features[i][non_pos[m]] = features[i][non_pos[m]-1]
                    else:#medium position
                        features[i][non_pos[m]] = (features[i][non_pos[m]+1] + features[i][non_pos[m]-1]) / 2
        return features
            
    
    def update_pose(self):
        '''
        update the pose by feed the data into the model
        show the label in the screen
        '''
        for i in range(2):
            # if data is not enough, skip
            if len(self._x[i]) < 240 or len(self._y[i]) < 240:
                continue
            # if data out of range, raise error
            if len(self._x[i]) > 240 or len(self._y[i]) > 240:
                raise ValueError("out of range")
            Time = time.time()
            data = np.array(self._x[i] + self._y[i])
            data = np.reshape(data, (1, 480))
            data = self.filter_none_data(data)
            self._x[i] = []
            self._y[i] = []
            
            data = data.astype(np.int32)
            # reshape the data to fit the model
            data = data.reshape(-1,2,16,15)
            # turn into torch tensor
            data = torch.Tensor(data)

            predict = np.argmax(self.net(data).data.numpy())
            # update the label
            self._pose[i] = self.labels[predict]
            print(self._pose[i])
            # restart the interval
            self._interval[i] = 0
            # show the time of predict
            print(time.time()-Time)
        

    # draw the frame on the screen
    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    # run pygame
    def run(self):
        # -------- Main Program Loop -----------
        while not self._done:
            # Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                    

            # Getting frames and drawing  
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()

                # it can be reshape into image matrix(RGB-D)
                # image = frame.reshape((self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.width, 4),order='C')
                # image = image[:, :, :3]
                # cv2.imshow('test', image)

                self.draw_color_frame(frame, self._frame_surface)
                frame = None

            # We have a body frame, so can get skeletons
            if self._kinect.has_new_body_frame(): 
                self._bodies = self._kinect.get_last_body_frame()

            # draw skeletons to _frame_surface
            if self._bodies is not None:
                self._body_num = self._kinect.max_body_count
                # traverse the bodies
                for i in range(0, self._kinect.max_body_count):
                    body = self._bodies.bodies[i]
                    if not body.is_tracked: 
                        self._body_num -= 1
                        continue 
                    
                    joints = body.joints 
                    # convert joint coordinates to color space 
                    joint_points = self._kinect.body_joints_to_color_space(joints)
                    # draw the box and store the data
                    self.draw_box(joints, joint_points, SKELETON_COLORS[i], i)

            # copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
            # (screen size may be different from Kinect's color frame size)
            self.update_pose()
            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height))
            
            # Display the action label
            font = pygame.font.Font(None, 60)
            if self._body_num == 1:
                text = font.render(self._pose[0], 1, (255, 10, 10))
            else:
                text = font.render(self._pose[0] + '    ' + self._pose[1], 1, (255, 10, 10))
            textpos = text.get_rect()
            textpos.centerx = surface_to_draw.get_rect().centerx
            surface_to_draw.blit(text, textpos)
            self._screen.blit(surface_to_draw, (0,0))
            # update the frame
            pygame.display.update()

            # Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # Limit the fps
            self._clock.tick(20)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "Kinect v2 Body Game"
game = BodyGameRuntime();
game.run();

