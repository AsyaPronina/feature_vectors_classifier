import torch
import torch.nn as nn

# Defining input size, hidden layer 1 size, hidden layer 2 size, output size respectively
#n_in, n_h1, n_h2, n_out = 512, 200, 200, 6
n_in, n_h1, n_h2, n_out = 512, 200, 200, 6

model = nn.Sequential(nn.Linear(n_in, n_h1),
                      nn.ReLU(),
                      nn.Linear(n_h1, n_h2),
                      nn.ReLU(),
                      nn.Linear(n_h2, n_out),
                      nn.Softmax()) #fails with dim=1 - Why?

#model = nn.Sequential(nn.Linear(n_in, n_out),
#                      #nn.Sigmoid(),
#                      #nn.Linear(n_h1, n_h2),
#                      #nn.ReLU(),
#                      #nn.Linear(n_h1, n_out),
#                      nn.Sigmoid()) #fails with dim=1 - Why?
