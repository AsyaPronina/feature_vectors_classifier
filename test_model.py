# -*- coding: cp1251 -*-
import json

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

import numpy

#turn training data loader to be similar to this:
class FeatureVectors(data.Dataset):
    def __init__(self, filename):
        self.json_data = None
        with open(filename) as json_file:
            self.json_data = json.load(json_file)

        self.feature_vectors = []
        self.labels = []

        fv_classes = self.json_data["labels"]

        for fv_class in fv_classes:
            feature_vectors = self.json_data[fv_class]
            self.feature_vectors.update([ [ float(n) for n in v.split(', ') ] for v in feature_vectors ])
            self.labels.update([fv_class] * len(feature_vectors))

        self.size = len(self.feature_vectors)

        p = numpy.random.permutation(self.size)
        self.feature_vectors = numpy.array(self.feature_vectors)[p].tolist()
        self.labels = numpy.array(self.labels)[p].tolist()
        pass
        
    
    def __len__(self):   # Length of the dataset.
        return self.size
    
    def __getitem__(self, index):   # Function that returns one point and one label.
        return torch.Tensor(self.feature_vectors[index]), torch.Tensor(classes_map[self.labels[index]])

# Training data loader

batch_size = 1 # here output will be formed from 1 vec
# old logic:
# batch_size is set to 10 as we have 6 classes: and form vector using 1 element from each of 5 classes and 5 elements of the latest.
# it is used because of sizes mismatch

test_data = FeatureVectors("test_data.json")
test_data_loader = data.DataLoader(test_data, batch_size=batch_size, num_workers=0)

# Defining input size, hidden layer 1 size, hidden layer 2 size, output size respectively
n_in, n_h1, n_h2, n_out = 512, 200, 200, 6

model = nn.Sequential(nn.Linear(n_in, n_h1),
                      nn.ReLU(),
                      nn.Linear(n_h1, n_h2),
                      nn.ReLU(),
                      nn.Linear(n_h2, n_out),
                      nn.Softmax()) #fails with dim=1 - Why?

model.load_state_dict(torch.load('classifier.pt'))

correct_preds = 0

for i, (test_data, test_target) in enumerate(test_data_loader):
        test_data, test_target = Variable(test_data), Variable(test_target)

        pred = model(test_data)

        pred_max = 0
        pred_max_i = 0

        target_max = 0
        target_max_i = 0

        for i in range(0, len(pred)):
            if pred[i] > pred_max:
                pred_max = pred[i]
                pred_max_i = i

            if test_target[i] > target_max:
                target_max = test_target[i]
                target_max_i = i

        if pred_max_i == target_max_i:
            correct_preds += 1

print('Accuracy: ' + str((correct_preds * 100) / test_data.size) + '%')
