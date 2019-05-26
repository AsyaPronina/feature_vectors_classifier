import json

import torch
import torch.utils.data as data

import numpy

# Classes dict
classes_map = { "Asyok"     : [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "daryafret" : [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "Nastya"    : [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "Malinka"   : [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "Ion"       : [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                "Unknown"   : [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}

# Class for training data loader
# trying to implement logic for batch size = 10 (deterministic()
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
            self.feature_vectors.extend([ [float(n) for n in v.split(', ') ] for v in feature_vectors])
            self.labels.extend([fv_class] * len(feature_vectors))

        self.size = len(self.feature_vectors)

        p = numpy.random.permutation(self.size)
        self.feature_vectors = numpy.array(self.feature_vectors)[p].tolist()
        self.labels = numpy.array(self.labels)[p].tolist()
        pass
        
    def __len__(self):   # Length of the dataset.
        return self.size
    
    def __getitem__(self, index):
        return torch.Tensor(self.feature_vectors[index]), torch.Tensor(classes_map[self.labels[index]])

    def shuffle(self):
        p = numpy.random.permutation(self.size)
        self.feature_vectors = numpy.array(self.feature_vectors)[p].tolist()
        self.labels = numpy.array(self.labels)[p].tolist()
        pass
