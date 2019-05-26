from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import numpy as np

import data

classes = {
              "Asyok"     : 0,
              "daryafret" : 1,
              "Nastya"    : 2,
              "Malinka"   : 3,
              "Ion"       : 4,
              "Unknown"   : 5
          }

model = LogisticRegression(random_state=0, solver='sag',
                           multi_class='multinomial')

#model = KNeighborsClassifier(n_neighbors=3)

#model = XGBClassifier()

train_data = data.FeatureVectors("train_data.json")
test_data = data.FeatureVectors("test_data.json")

X_train = np.array(train_data.feature_vectors)
y_train = np.array([ classes[l] for l in train_data.labels ])

X_test = np.array(test_data.feature_vectors)
y_test = np.array([ classes[l] for l in test_data.labels ])

model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))