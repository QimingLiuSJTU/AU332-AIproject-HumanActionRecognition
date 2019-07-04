# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 17:12:05 2018
@author: lenovo
"""

import torch.nn as nn
import numpy as np

class C2D(nn.Module):

    def __init__(self):
        super(C2D, self).__init__()
        
        
        #input shape: (16, 15)
        self.conv21 = nn.Conv2d(2,  6,  kernel_size=(3,3), padding=(1,1), stride=(1,1)) #(16, 15)
        self.conv22 = nn.Conv2d(6,  10, kernel_size=(3,3), padding=(1,1), stride=(1,1)) #(16, 15)
        self.pool21 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)) #(8, 7)
        
        self.conv23 = nn.Conv2d(10, 16, kernel_size=(3,3), padding=(1,1), stride=(1,1)) #(8, 7)
        self.conv24 = nn.Conv2d(16, 24, kernel_size=(3,3), padding=(1,1), stride=(1,1)) #(8, 7)
        self.pool22 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)) #(4, 3)
        
        self.conv25 = nn.Conv2d(24, 38, kernel_size=(3,3), padding=(1,1), stride=(1,1)) #(4, 3)
        self.conv26 = nn.Conv2d(38, 64, kernel_size=(3,3), padding=(1,1), stride=(1,1)) #(4, 3)
        self.pool23 = nn.MaxPool2d(kernel_size=(2,1),stride=(2,1)) #(2, 3)
        #64 * 2 * 3 = 384
        
        #input shape: (16, 150, 120)
        self.conv31 = nn.Conv3d(1, 2, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1)) #(16, 480, 600)
        self.conv32 = nn.Conv3d(2, 4, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1)) #(16, 480, 600)
        self.pool31 = nn.MaxPool3d(kernel_size=(2,3,3), stride=(2,3,3)) #(8,50,40)
        
        self.conv33 = nn.Conv3d(4, 8, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1))#(8,50,40)
        self.conv34 = nn.Conv3d(8, 12, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1)) #(8,50,40)
        self.pool32 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)) #(4,25,20)
        
        self.conv35 = nn.Conv3d(12, 15, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1))#(4,25,20)
        self.conv36 = nn.Conv3d(15, 18, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1))#(4,25,20)
        self.pool33 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)) #(2,12,10)
        
        self.conv37 = nn.Conv3d(18, 20, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1))#(2,12,10)
        self.conv38 = nn.Conv3d(20, 24, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1))#(2,12,10)
        self.pool34 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) #(2.6,5)
        
        self.conv39 = nn.Conv3d(24, 28, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1)) #(2,6,5)
        self.conv310 = nn.Conv3d(28, 32, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1)) #(2,6,5)
        self.pool35 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) #(2,3,2)       
        #32 * 2 * 3 * 2 = 384

        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192,  96)
        self.fc4 = nn.Linear(96,  48)
        self.fc5 = nn.Linear(48,  24)
        self.fc6 = nn.Linear(24,  12)
        self.fc7 = nn.Linear(12,  9)
        #output shape: 9 labels
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x1, x2):

        #convolution and pooling layers
        h = self.relu(self.conv21(x1))
        h = self.relu(self.conv22(h))
        h = self.pool21(h)

        h = self.relu(self.conv23(h))
        h = self.relu(self.conv24(h))
        h = self.pool22(h)
        
        h = self.relu(self.conv25(h))
        h = self.relu(self.conv26(h))
        h = self.pool23(h)
        
        
        g = self.relu(self.conv31(x2))
        g = self.relu(self.conv32(g))
        g = self.pool31(g)
        
        g = self.relu(self.conv33(g))
        g = self.relu(self.conv34(g))
        g = self.pool32(g)
        
        g = self.relu(self.conv35(g))
        g = self.relu(self.conv36(g))
        g = self.pool33(g)
        
        g = self.relu(self.conv37(g))
        g = self.relu(self.conv38(g))
        g = self.pool34(g)
        
        g = self.relu(self.conv39(g))
        g = self.relu(self.conv310(g))
        g = self.pool35(g)
        

        #flatten layer
        h = h.view(-1, 384)
        g = g.view(-1, 384)
        f = np.concatenate((h,g), axis = 1)
        
        
        #full connection layers
        f = self.relu(self.fc1(f))
        f = self.dropout5(f)
        f = self.relu(self.fc2(f))
        f = self.dropout5(f)
        f = self.relu(self.fc3(h))
        f = self.dropout4(f)
        f = self.relu(self.fc4(h))
        f = self.dropout3(f)  
        f = self.relu(self.fc5(h))
        f = self.dropout2(f) 
        f = self.relu(self.fc6(h))
        f = self.dropout1(f) 
        
        logits = self.fc7(f)
        
        probs = self.softmax(logits)

        return probs

