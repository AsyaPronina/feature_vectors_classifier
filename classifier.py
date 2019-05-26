# -*- coding: cp1251 -*-
import torch
import torch.nn as nn
import torch.utils.data as torch_data
from torch.autograd import Variable

import topology
import data
import validator

model = topology.model

# construct the loss function
criterion = torch.nn.MSELoss()

# construct the optimizer (stochastic gradient descent in this case)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) #step in gradient descent = [10e-3, 10e-1]

# set epochs
epochs = 2000
 
#after every epoch run validation and check that accuracy do not do slown
#may be network are super learned on determenistic distributuion
#add regularization, drop-out for hidden layers (input? output?), l2dk
#stepdk for learning rate
#may insert batchnorm instead of drop-out(as they are badly work together)

#may be try Adam instead of SGD (as it is simpler)

#May be Egor is right with method of training about my concern about batch and class? - yes
#But my idea has sense

# Training data loader
#train_batch_size = 4
train_batch_size = 4
train_data = data.FeatureVectors("train_data.json")
train_data_loader = torch_data.DataLoader(train_data, batch_size=train_batch_size, num_workers=0)

# Test data loader
test_batch_size = 1
test_data = data.FeatureVectors("test_data.json")
test_data_loader = torch_data.DataLoader(test_data, batch_size=test_batch_size, num_workers=0)

# training routine
for epoch in range(epochs):
    for i, (tr_data, tr_target) in enumerate(train_data_loader):
        tr_data, tr_target = Variable(tr_data), Variable(tr_target)

        # zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        
        # forward pass: compute predicted y by passing x to the model
        pred = model(tr_data)

        # compute and print loss
        loss = criterion(pred, tr_target)
        
        #print('epoch: ', epoch,' loss: ', loss.item())        
        # perform a backward pass (backpropagation)
        loss.backward()
        #print('target: ', tr_target)
        #print('pred: ', pred)
        
        # update the parameters
        optimizer.step()

    train_accuracy = validator.validate_model(model, train_data_loader, train_data)
    test_accuracy = validator.validate_model(model, test_data_loader, test_data)
    print('Epoch: {0}, train accuracy: {1}, test accuracy: {2}%'.format(epoch, train_accuracy, test_accuracy))

    if test_accuracy >= 99:
        print('Accuracy is greater or equal to 99. Stopping the training.')
        torch.save(model.state_dict(), 'classifier.pt')
        exit(0)

    train_data.shuffle()

torch.save(model.state_dict(), 'classifier.pt')
