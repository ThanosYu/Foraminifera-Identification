from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.utils import to_categorical
import numpy as np
from keras.callbacks import LearningRateScheduler
import math
import pickle
from sklearn.utils import shuffle

foram_fc_model_path = './foram_fc.h5'
batch_size = 32
epochs = 60

data = pickle.load(open('forams_features.p', "rb"))
features = np.squeeze(data['features'])
labels = data['labels']
class_count = data['class_count']
features, labels = shuffle(features, labels)
one_hot_labels = to_categorical(labels)
label2class = data['label2class']
feature_shape = features.shape

print(feature_shape)
print(one_hot_labels.shape)
print(label2class)

foram_fc_model = Sequential()
foram_fc_model.add(Dropout(0.05, input_shape=feature_shape[1:]))
foram_fc_model.add(Dense(512, activation='relu'))
foram_fc_model.add(Dropout(0.15))
foram_fc_model.add(Dense(512, activation='relu'))
foram_fc_model.add(Dense(7, activation='softmax'))

foram_fc_model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])

initial_lrate = 1e-3


def step_decay(epoch):
    drop = 0.95
    epochs_drop = 1
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


# learning schedule callback
lrate = LearningRateScheduler(step_decay)
foram_fc_model.fit(features, one_hot_labels,
                   epochs=epochs,
                   batch_size=batch_size,
                   shuffle=True,
                   validation_split=0.2,
                   class_weight={key: 1000 / class_count[key] for key in class_count},
                   callbacks=[lrate])
