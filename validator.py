# -*- coding: cp1251 -*-
import torch
import torch.nn as nn
import torch.utils.data as torch_data
from torch.autograd import Variable

import topology
import data

def validate_model(model, test_data_loader, test_data, mode='test'):
    classes_stat = {
                        "Asyok"     : ( 0, 0 ),
                        "daryafret" : ( 0, 0 ),
                        "Nastya"    : ( 0, 0 ),
                        "Malinka"   : ( 0, 0 ),
                        "Ion"       : ( 0, 0 ),
                        "Unknown"   : ( 0, 0 ) 
                  }

    classes_list = [ "Asyok", "daryafret", "Nastya", "Malinka", "Ion", "Unknown" ]

    correct_preds = 0
    for i, (test_vector, test_target) in enumerate(test_data_loader):
            test_vector, test_target = Variable(test_vector), Variable(test_target)

            pred = model(test_vector)

            for i in range(0, len(pred)):
                pred_max = 0
                pred_max_ind = 0

                target_max = 0
                target_max_ind = 0

                current_pred = pred[i]
                current_target = test_target[i]

                for j in range(0, len(current_pred)):
                    if current_pred[j] > pred_max:
                        pred_max = current_pred[j]
                        pred_max_ind = j

                    if current_target[j] > target_max:
                        target_max = current_target[j]
                        target_max_ind = j

                total, error = classes_stat[classes_list[target_max_ind]]
                total += 1

                if pred_max_ind == target_max_ind:
                    correct_preds += 1
                else:
                     error += 1

                classes_stat[classes_list[target_max_ind]] = total, error
                #else:
                #    print('Mismatch!')
                #    print(pred)
                #    print(pred_max_ind)
                #    print(test_target)
                #    print(target_max_ind)

    print(classes_stat)
    return (correct_preds * 100) / test_data.size

if __name__ == '__main__':
    model = topology.model
    model.load_state_dict(torch.load('classifier.pt'))

    batch_size = 4
    test_data = data.FeatureVectors("test_data.json")
    test_data_loader = torch_data.DataLoader(test_data, batch_size=batch_size, num_workers=0)

    accuracy = validate_model(model, test_data_loader, test_data)
    print('Accuracy: ' + str(accuracy) + '%')

