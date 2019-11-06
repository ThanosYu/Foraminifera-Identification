from keras.applications import resnet50
from keras.applications import vgg16
import os
import numpy as np
from natsort import natsorted
import cv2
import pickle
import glob

# data_dir = './NCSU-CUB Foram Images 01/'
data_dir = './NCSU-CUB_Foram_Images_G-bulloides/'
img_shape = (224, 224)

# build the pretrained model
vgg16_model = vgg16.VGG16(include_top=False, pooling='avg')
resnet50_model = resnet50.ResNet50(include_top=False, pooling='avg')
print('Pre-trained Model loaded.')

forams_features = []
forams_labels = []
label2class = {}
class_count = {}
class_list = natsorted([os.path.join(data_dir, folder) for folder in
                        os.listdir(data_dir) if not folder.endswith('.txt')])
for class_id, class_folder in enumerate(class_list):

    sample_list = natsorted([os.path.join(class_folder, folder)
                             for folder in os.listdir(class_folder)])
    class_count[class_id] = len(sample_list) / 1000
    for sample_folder in sample_list:

        img_filenames = natsorted([os.path.join(sample_folder, file) for file in
                                   os.listdir(sample_folder) if file.endswith('.png')])
        group_images = np.zeros(img_shape + (len(img_filenames),))
        for i, img_file in enumerate(img_filenames):
            img = cv2.imread(img_file, 0)
            img = cv2.resize(img, img_shape, interpolation=cv2.INTER_CUBIC)
            group_images[:, :, i] = img
        img90 = np.expand_dims(np.percentile(group_images, 90, axis=-1), axis=-1)
        img50 = np.expand_dims(np.percentile(group_images, 50, axis=-1), axis=-1)
        img10 = np.expand_dims(np.percentile(group_images, 10, axis=-1), axis=-1)
        img = np.concatenate((img10, img50, img90), axis=-1)
        img = np.expand_dims(img, axis=0)
        fea_vgg16 = vgg16_model.predict_on_batch(vgg16.preprocess_input(img))
        fea_resnet50 = resnet50_model.predict_on_batch(resnet50.preprocess_input(img))
        fea = np.concatenate((fea_vgg16, fea_resnet50), axis=1)
        forams_features.append(fea)
        forams_labels.append(class_id)
        label2class[class_id] = class_folder.split('/')[-1]

forams_features = np.array(forams_features)
forams_labels = np.array(forams_labels)
print("================forams_features: ", forams_features.shape)
print("================forams_labels: ", forams_labels.shape)

with open('./forams_features.p', 'wb') as f:
    pickle.dump({'features': forams_features, 'labels': forams_labels,
                 'label2class': label2class, 'class_count': class_count}, f)
