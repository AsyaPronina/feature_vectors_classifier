# -*- coding: cp1251 -*-
import json

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

import numpy

# Classes dict
classes_map = { "Asyok"     : [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "daryafret" : [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "Nastya"    : [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "Malinka"   : [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "Ion"       : [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                "Unknown"   : [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}

# Class for training data loader
# trying to implement logic for batch size = 10
class FeatureVectors(data.Dataset):
    def __init__(self, filename):
        self.json_data = None
        with open(filename) as json_file:
            self.json_data = json.load(json_file)

        self.feature_vectors = []
        self.labels = []

        fv_classes = self.json_data["labels"]
        fv_classes.remove("Unknown")

        #it is expected that each class has the same amount of feature vectors besides Unknown.
        size = len(self.json_data[fv_classes[0]])
        uknown_size = len(self.json_data["Unknown"])

        if uknown_size % size != 0:
            raise RuntimeError("Size of feature vectors array for 'Unknown' class is not dividable by size of other classes arrays!")

        unknown_in_one_batch = int(uknown_size / size)

        for i in range(0, size):
            for fv_class in fv_classes:
                feature_vector = self.json_data[fv_class][i]

                self.feature_vectors.append([ float(n) for n in feature_vector.split(', ') ])
                self.labels.append(fv_class)

            for j in range(0, unknown_in_one_batch):
                feature_vector = self.json_data["Unknown"][unknown_in_one_batch * i + j]

                self.feature_vectors.append([ float(n) for n in feature_vector.split(', ') ])
                self.labels.append("Unknown")

        self.size = len(self.feature_vectors)

        #p = numpy.random.permutation(self.size)
        #self.feature_vectors = numpy.array(self.feature_vectors)[p].tolist()
        #self.labels = numpy.array(self.labels)[p].tolist()
        
    
    def __len__(self):   # Length of the dataset.
        return self.size
    
    def __getitem__(self, index):   # Function that returns one point and one label.
        return torch.Tensor(self.feature_vectors[index]), torch.Tensor(classes_map[self.labels[index]])

# Training data loader

batch_size = 10 # here output will be formed from 10 vecs
# batch_size is set to 10 as we have 6 classes: and form vector using 1 element from each of 5 classes and 5 elements of the latest.
# it is used because of sizes mismatch

train_data = FeatureVectors("train_data.json")
train_data_loader = data.DataLoader(train_data, batch_size=batch_size, num_workers=0)

# Defining input size, hidden layer 1 size, hidden layer 2 size, output size respectively
n_in, n_h1, n_h2, n_out = 512, 200, 200, 6

# Create a model
model = nn.Sequential(nn.Linear(n_in, n_h1),
                      nn.ReLU(),
                      nn.Linear(n_h1, n_h2),
                      nn.ReLU(),
                      nn.Linear(n_h2, n_out),
                      nn.Softmax()) #fails with dim=1 - Why?

# construct the loss function
criterion = torch.nn.MSELoss()

# construct the optimizer (stochastic gradient descent in this case)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) #step in gradient descent = [10e-3, 10e-1]

# set epochs
epochs = 1500

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
        
        print('epoch: ', epoch,' loss: ', loss.item())      
        
        # perform a backward pass (backpropagation)
        loss.backward()
        print('target: ', tr_target)
        print('pred: ', pred)
        
        # update the parameters
        optimizer.step()


torch.save(model.state_dict(), 'classifier.pt')
