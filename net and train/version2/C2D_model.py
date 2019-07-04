import torch.nn as nn

class C2D(nn.Module):

    def __init__(self):
        super(C2D, self).__init__()
        
        #input shape: (16, 15)
        self.conv1 = nn.Conv2d(2,  4, kernel_size=(3,3), padding=(1,1), stride=(1,1))#(16, 15)
        self.conv2 = nn.Conv2d(4,  8, kernel_size=(3,3), padding=(1,1), stride=(1,1))#(16, 15)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))#(8, 7)
        
        self.conv3 = nn.Conv2d(8, 12, kernel_size=(3,3), padding=(1,1), stride=(1,1))#(8, 7)
        self.conv4 = nn.Conv2d(12,16, kernel_size=(3,3), padding=(1,1), stride=(1,1))#(8, 7)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))#(4, 3)
        
        self.conv5 = nn.Conv2d(16,20, kernel_size=(3,3), padding=(1,1), stride=(1,1))#(4, 3)
        self.conv6 = nn.Conv2d(20,24, kernel_size=(3,3), padding=(1,1), stride=(1,1))#(4, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,1),stride=(2,1))#(2, 3)
        
        self.fc1 = nn.Linear(144, 72)
        self.fc2 = nn.Linear(72, 36)
        self.fc3 = nn.Linear(36,  18)
        self.fc4 = nn.Linear(18,  7)
        #output labels: 7
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):

        #convolution and pooling layers
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.pool1(h)

        h = self.relu(self.conv3(h))
        h = self.relu(self.conv4(h))
        h = self.pool2(h)
        
        h = self.relu(self.conv5(h))
        h = self.relu(self.conv6(h))
        h = self.pool3(h)
        
        #flatten layer
        h = h.view(-1, 144)
        
        #full connection layers
        h = self.relu(self.fc1(h))
        h = self.dropout3(h)
        h = self.relu(self.fc2(h))
        h = self.dropout3(h)
        h = self.relu(self.fc3(h))
        h = self.dropout2(h)
        logits = self.fc4(h)
        
        probs = self.softmax(logits)

        return probs
