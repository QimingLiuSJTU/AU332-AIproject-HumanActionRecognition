import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from C2D_model import C2D

#calculate the accuracy
#y is actual label and z is prediction result(haven't changed to label)
def accuracy_cal(y, z):
    batch_acc = 0
    z_label = torch.argmax(z,1).numpy()
    for m in range(len(y)):
        if int(y[m]) == int(z_label[m]):
            batch_acc += 1
    return batch_acc        

#testing set
def testing(net, testing_samples, testing_labels, loss_fn):
    test_size = 1 
    test_iterate = int(len(testing_samples) / test_size)
    test_acc = 0
    test_loss = 0
    for m in range(test_iterate):
        y_pred = net(testing_samples[m*test_size:(m+1)*test_size])
        y = testing_labels[m*test_size:(m+1)*test_size]
        #calculate testing loss
        loss = loss_fn(y_pred, y.type(torch.long)) 
        test_loss += loss.item()
        #calculate testing accuracy
        test_acc += accuracy_cal(y,y_pred)
    
    return test_loss, test_acc

#training process
def training_process(net, training_samples, training_labels, testing_samples, testing_labels, max_iterations, batch_size, learning_rate):
    
    #set the parameters, batch size, max_iterations, test_size    
    batch_size = batch_size 
    max_iterations = max_iterations  #200
    iters_per_iterate = int(np.ceil(len(training_samples)/batch_size))
    
    #define loss function
    loss_fn = nn.CrossEntropyLoss(reduction = 'sum')
    
    #define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate) #0.001

    ################
    #begin training#
    ################
    #store the training and testing accuracy and loss information
    training_loss, testing_loss = [], []
    training_accuracy, testing_accuracy = [], []

    for iterate in range(max_iterations):
        iterate_loss = 0
        iterate_acc = 0
        for i in range(iters_per_iterate):
            #if come to the end
            if i == (iters_per_iterate -1) :
                x = (training_samples[i*batch_size: len(training_samples)])
                y = training_labels[i*batch_size: len(training_samples)]
            else:
                x = (training_samples[i*batch_size: (i+1)*batch_size])
                y = training_labels[i*batch_size: (i+1)*batch_size]
            z = net(x)
            
            #calculate the batch loss
            loss_val = loss_fn(z, (y.type(torch.long)))
            iterate_loss += loss_val.item()
            #calculate the batch accuracy
            iterate_acc += accuracy_cal(y,z)
            
            #update the net parameters, BP algorithm
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
        #update the acc and loss list
        training_loss.append(iterate_loss/len(training_samples))
        training_accuracy.append(100 * iterate_acc / len(training_samples))
        
        #fit testing set 
        test_loss, test_acc = testing(net, testing_samples, testing_labels, loss_fn)
        testing_loss.append(test_loss / len(testing_samples))
        testing_accuracy.append(100 * test_acc / len(testing_samples))
        print(iterate, training_accuracy[-1], testing_accuracy[-1])
        
    #plot the images
    plot_image(training_loss, training_accuracy, testing_loss, testing_accuracy)
    #find the best model's number
    print('The best model is in:', np.argmax(np.array(testing_accuracy)))
    return np.mean(np.array(training_accuracy[-1])), np.mean(np.array(testing_accuracy[-1]))
    
#plot the image of training and testing accuracy and loss in the training process        
def plot_image(training_loss, training_accuracy, testing_loss, testing_accuracy):
    
    figure, ax = plt.subplots(figsize = [9,10])
    x= range(len(training_loss))
    
    #plot the loss
    plt.subplot(2,1,1)
    line1, =plt.plot(x, training_loss, 'r-', label = "Training loss")
    plt.title('Loss')

    plt.subplot(2,1,1)
    line2, =plt.plot(x, testing_loss, 'b-', label = 'Testing loss')
    plt.legend(handles =[line1, line2], loc = 0)
    
    #plot the accuracy
    plt.subplot(2,1,2)
    line3, =plt.plot(x,training_accuracy, 'r-', label = 'Training accuracy')
    plt.title('Accuracy')

    plt.subplot(2,1,2)
    line4, =plt.plot(x,testing_accuracy, 'b-', label = 'Testing accuracy')
    plt.legend(handles = [line3, line4], loc = 0)
    #show the figure
    plt.show()
      


def training(net, features, labels, max_iterations, batch_size, learning_rate):
    
    #N: the number of samples, C: 2 channels (x,y), F: frames, H: height, actually 15 key points
    N, C, F, H = features.shape
    
    #reorder the trainig set
    np.random.seed(6)
    np.random.shuffle(features)
    np.random.seed(6)
    np.random.shuffle(labels)

    #define the number of training set and testing set
    training_num = int(N * 0.8)
    #get the training set and testing set
    training_samples = torch.Tensor(features[:training_num])
    training_labels = torch.Tensor(labels[:training_num])
    testing_samples = torch.Tensor(features[training_num:])
    testing_labels = torch.Tensor(labels[training_num:])
    #train the net
    train_acc, test_acc =  training_process(net, training_samples, training_labels, testing_samples, testing_labels, max_iterations, batch_size, learning_rate)

    #show the final accuracy
    training_acc, testing_acc = [], []
    training_acc.append(train_acc)
    testing_acc.append(test_acc)
    show = pd.DataFrame(columns = ('train_acc','test_acc'))
    show['train_acc'] = training_acc
    show['test_acc'] = testing_acc
    print(show)
    
    #save the model
    torch.save(net, 'net.pkl')
    
#pre_treat the features and labels
def pre_treat(features, labels): #pre_treat the data
    
    #delete some labels samples(which have many None data)
    del_row = []
    for i in range(len(labels)):
        if int(labels[i]) in [4]:
            del_row.append(i)
    features = np.delete(features, del_row, 0)
    labels = np.delete(labels, del_row, 0)
    
    
    #change the remain labels to right order  
    for i in range(len(labels)):
        if int(labels[i]) >= 1 and int(labels[i]) <= 3:
            labels[i] = int(labels[i]) - 1
        else:
            labels[i] = int(labels[i]) - 2

    return features, labels

#using value iteration to process None data
def filter_none_data(features):
    
    #get the None data's position and set them to 0 initially
    for i in range(len(features)):
        non_pos = []
        for j in range(len(features[i])):
            if str(features[i][j]) == str(None):
                non_pos.append(j)
                features[i][j] = 0
        
        #using 10 times value iteration to padding the missing values
        for k in range(10): #value iteration times
            for m in range(len(non_pos)):
                if non_pos[m]%15 == 0:  #begin position
                    features[i][non_pos[m]] = features[i][non_pos[m]+1]
                elif (non_pos[m]+1) % 15 == 0: #end position
                    features[i][non_pos[m]] = features[i][non_pos[m]-1]
                else:#medium position
                    features[i][non_pos[m]] = (features[i][non_pos[m]+1] + features[i][non_pos[m]-1]) / 2
    return features

if __name__ == '__main__':

    #load data
    features = np.load('sample_expend.npy')
    labels = np.load('label_expend.npy')
    
    #preprocess the features and labels
    features, labels = pre_treat(features, labels)
    features = filter_none_data(features)
    
    #change the features to appropriate type
    features = features.astype(np.int32)
    
    #N,C,H,W, actually frames as H
    features = features.reshape(-1,2,16,15)
    
    #initial the net
    net = C2D()

    #assign the tunning parameters
    max_iterations = 300 #300
    batch_size = 5 # 4
    learning_rate = 0.00005  #0.00001
    
    #train the net parameters
    training(net, features, labels, max_iterations, batch_size, learning_rate)
    print('max_iterations\t', 'batch_size\t', 'learning_rate')
    print(max_iterations,'\t', batch_size, '\t', learning_rate)
    