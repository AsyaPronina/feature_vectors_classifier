import numpy as np
import torch
from torch import nn
import torch.onnx

import io

n_in, n_h1, n_h2, n_out = 512, 200, 200, 6

model = nn.Sequential(nn.Linear(n_in, n_h1),
                      nn.ReLU(),
                      nn.Linear(n_h1, n_h2),
                      nn.ReLU(),
                      nn.Linear(n_h2, n_out),
                      nn.Softmax())


# Initialize model with the pretrained weights
model.load_state_dict(torch.load('classifier.pt'))
model.eval()

x = torch.randn(1, n_in)

# Export the model
torch_out = torch.onnx._export(model,             # model being run
                               x,                       # model input (or a tuple for multiple inputs)
                               "classifier.onnx",       # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file
