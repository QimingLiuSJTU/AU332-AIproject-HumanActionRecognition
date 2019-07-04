import torch.nn as nn

class C3D(nn.Module):

    def __init__(self):
        super(C3D, self).__init__()
        
        #Input shsape: (8,15)
        self.conv1 = nn.Conv2d(2,  8, kernel_size=(3,3), padding=(1,1), stride=(1,1))#(8,15)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))#(4,7)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3,3), padding=(1,1), stride=(1,1))#(4,7)
        self.conv3 = nn.Conv2d(16,32, kernel_size=(3,3), padding=(1,1), stride=(1,1))#(4,7)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))#(2,3)
        
        self.fc1 = nn.Linear(192,96)
        self.fc2 = nn.Linear(96,48)
        self.fc3 = nn.Linear(48,24)
        self.fc4 = nn.Linear(24,12)
        self.fc5 = nn.Linear(12,5)
        #output labels: 5
        
        self.dropout = nn.Dropout(p=0.1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        
        #convolution and pooling layers
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))       
        h = self.relu(self.conv3(h))
        h = self.pool3(h)
        
        #fully connection layers
        h = h.view(-1, 192)
        h = self.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.relu(self.fc2(h))
        h = self.dropout(h)
        h = self.relu(self.fc3(h))
        h = self.dropout(h)
        h = self.relu(self.fc4(h))
        h = self.dropout(h)
        logits = self.fc5(h)
        #softmax layer
        probs = self.softmax(logits)

        return probs