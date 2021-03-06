from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime


import ctypes
import _ctypes
import pygame
import sys
import math
import numpy as np
import cv2
import time
import os

'''
This py code is used to collect our dataset containing rgb images and depth data
it is based on a demo of drawing box (not the final version)

you need to change PATH to your destination, it will keep collecting samples,
every sample is saved in a dist

the function is done at # 310~340, So I only wrote notation at that part
the full notation is at done in the final virsion demo
'''

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# path to store the data
PATH = "E:/AI/final/PyKinect2/examples/video/10/4/"

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

        # pose of user
        self._pose = "stand"

        #to get data
        self._is_get = False

        #begin record
        self._frame_num = 0
        if os.listdir(PATH) == []:
            self._group_num = 0
        else:
            n = 0
            l = os.listdir(PATH)
            for i in l:
                m = int(i)
                if m > n:
                    n = m
            self._group_num = n + 1

        self.depth = np.zeros((1,600,480))
        # data
        self._x = []
        self._y = []
        self.labels = ["wave", "applause", "sit", "stand", "watch wrist"]

        # interval
        self._inter_range = 100
        self._interval = 0

        print("model loaded")

    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good 
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)

        try:
            pygame.draw.line(self._frame_surface, color, start, end, 8)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

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


    def draw_box(self, joints, jointPoints, color):
        X = []
        Y = []
        self._interval += self._clock.get_time()
        if self._interval > self._inter_range:
            self._interval = 0
            self._is_get = True
        # [3] [20] [8] [9] [10] [4] [5] [6] [16] [17] [18] [12] [13] [14] [1]
        points = [PyKinectV2.JointType_Head, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight,
                PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ShoulderLeft,
                PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HipRight,
                PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_HipLeft,
                PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_SpineMid,
                PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HandRight,
                PyKinectV2.JointType_HandTipRight, PyKinectV2.JointType_ThumbRight, PyKinectV2.JointType_HandLeft,
                PyKinectV2.JointType_HandTipLeft, PyKinectV2.JointType_ThumbLeft, PyKinectV2.JointType_FootRight,
                PyKinectV2.JointType_FootLeft]
        left_most = float('inf')
        right_most = 0
        up_most = float('inf')
        down_most = 0
        for No in range(25):
            pt = points[No]
            jointState = joints[pt].TrackingState;
            if jointState == PyKinectV2.TrackingState_NotTracked or jointState == PyKinectV2.TrackingState_Inferred:
                if self._is_get == True and No < 15:
                    X.append(None)
                    Y.append(None)
                continue
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
            if self._is_get == True and No < 15:
                # print(No)
                # print(pt)
                # print(x, y)
                if x < 0 or x > 1920:
                    X.append(None)
                else:
                    X.append(int(x * 600 / 1920))
                    # self._is_get = False
                    # self._interval = self._inter_range
                if y < 0 or y > 1080:
                    Y.append(None)
                else:
                    Y.append(int(y * 480 / 1080))
                    # self._is_get = False
                    # self._interval = self._inter_range
                
                
        if self._is_get:
            if len(X)==15 and len(Y)==15:
                self._x += X
                self._y += Y
            else:
                self._interval = self._inter_range
            self._is_get = False
            # print(len(self._x), len(self._y))
            # print(self._x)
            # print(self._y)
        height = down_most - up_most
        width = right_most - left_most

        corner1 = (left_most - 0.7*math.sqrt(abs(width)), up_most - 0.15*height)
        corner2 = (right_most + 0.7*math.sqrt(abs(width)), up_most - 0.15*height)
        corner3 = (right_most + 0.7*math.sqrt(abs(width)), down_most + 0.15*height)
        corner4 = (left_most - 0.7*math.sqrt(abs(width)), down_most + 0.15*height)
        try:
            pygame.draw.line(self._frame_surface, color, corner1, corner2, 8)
            pygame.draw.line(self._frame_surface, color, corner2, corner3, 8)
            pygame.draw.line(self._frame_surface, color, corner3, corner4, 8)
            pygame.draw.line(self._frame_surface, color, corner4, corner1, 8)

        except: # need to catch it due to possible invalid positions (with inf)
            pass
        
    def update_pose(self):
        if len(self._x) < 120 or len(self._y) < 120:
            return
        if len(self._x) > 120 or len(self._y) > 120:
            print("range error!")
            self._x = []
            self._y = []
            return
        data = np.array(self._x + self._y)
        data = np.reshape(data, (1, 240))
        self._x = []
        self._y = []
        
        data = data.reshape(-1,2,8,15)
        data = torch.Tensor(data)

        predict = np.argmax(self.net(data).data.numpy())
        self._pose = self.labels[predict]
        
        self._interval = 0
        

    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def run(self):
        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                    
            # --- Game logic should go here

            # --- Getting frames and drawing  

            # --- Cool! We have a body frame, so can get skeletons
            if self._kinect.has_new_body_frame(): 
                self._bodies = self._kinect.get_last_body_frame()

            # --- draw skeletons to _frame_surface
            if self._bodies is not None:
                for i in range(0, self._kinect.max_body_count):
                    body = self._bodies.bodies[i]
                    if not body.is_tracked: 
                        continue 
                    
                    joints = body.joints 
                    # convert joint coordinates to color space 
                    joint_points = self._kinect.body_joints_to_color_space(joints)
                    #self.draw_body(joints, joint_points, SKELETON_COLORS[i])


            # --- Woohoo! We've got a color frame! Let's fill out back buffer surface with frame's data 
            if self._kinect.has_new_color_frame():
                self._interval += self._clock.get_time()
                if self._interval > self._inter_range:
                    self._interval = 0
                    self._is_get = True
                


                '''
                here is the main code to collect the dataset
                '''
                frame = self._kinect.get_last_color_frame()
                # reshape the frame into image shape
                image = frame.reshape((self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.width, 4),order='C')
                # get the rgb part and shrink them
                rgb_image = image[:, :, :3].copy()
                rgb_image = cv2.resize(rgb_image, (600, 480))
                # get the depth part and shrink it
                d_image = image[:, :, 3].copy()
                d_image = cv2.resize(d_image, (600, 480))
                d_image = np.reshape(d_image, (1,600,480))

                if self._is_get:
                    if self._frame_num == 0:
                        os.mkdir(PATH+str(self._group_num)+'/')
                    path = PATH + str(self._group_num) + '/'
                    # save the rgb image
                    cv2.imwrite(path+str(self._frame_num)+".jpg", rgb_image)
                    # concatenate 16 depth data together
                    self.depth = np.concatenate((self.depth, d_image), axis=0)
                    self._frame_num += 1
                    if self._frame_num == 16:
                        self.depth = np.delete(self.depth, 0, axis=0)
                        # save all the 16 frames' depth data
                        np.save(path+"depth.npy", self.depth)
                        self.depth = np.zeros((1,600,480))
                        self._frame_num = 0
                        self._group_num += 1

                

                # cv2.imshow('test', image)
                self.draw_color_frame(frame, self._frame_surface)
                frame = None

                    

            # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
            # --- (screen size may be different from Kinect's color frame size)
            
            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height))
            
            # # Display some text
            # font = pygame.font.Font(None, 60)
            # text = font.render(self._pose, 1, (10, 10, 10))
            # textpos = text.get_rect()
            # textpos.centerx = surface_to_draw.get_rect().centerx
            # surface_to_draw.blit(text, textpos)

            self._screen.blit(surface_to_draw, (0,0))
            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


if __main__ = "Kinect v2 Body Game":
    game = BodyGameRuntime()
    game.run()

