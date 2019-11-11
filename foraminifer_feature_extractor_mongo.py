from keras.applications import vgg16
import os
import numpy as np
import cv2
import pickle

import umap
import pandas as pd
from pymongo import MongoClient
import bson.binary

forams_features = []
forams_labels = []
label2class = {}
class_count = {}

# folder = '/home/q/q/CNOOC/foraminifer'
# folder = '/home/q/q/CNOOC/foraminifer/NCSU-CUB_Foram_Images_01/'
vgg16_model = vgg16.VGG16(include_top=False, pooling='avg')
# resnet50_model = resnet50.ResNet50(include_top=False, pooling='avg')
print('Pre-trained Model loaded.')

# data_dir = './NCSU-CUB_Foram_Images_01/'
# data_dir = './NCSU-CUB_Foram_Images_01/NCSU-CUB_Foram_Images_Globigerina_bulloides/'
# data_dir = './NCSU-CUB_Foram_Images_01/NCSU-CUB_Foram_Images_Globigerinoides_ruber/'
data_dir = './NCSU-CUB_Foram_Images_01/NCSU-CUB_Foram_Images_Globigerinoides_sacculifer/'
# data_dir = './NCSU-CUB_Foram_Images_01/NCSU-CUB_Foram_Images_Neogloboquadrina_dutertrei/'
# data_dir = './NCSU-CUB_Foram_Images_01/NCSU-CUB_Foram_Images_Neogloboquadrina_incompta/'
# data_dir = './NCSU-CUB_Foram_Images_01/NCSU-CUB_Foram_Images_Neogloboquadrina_pachyderma/'
# data_dir = './NCSU-CUB_Foram_Images_01/NCSU-CUB_Foram_Images_Others/'
# print('Folders  ', [folder for folder in os.listdir(data_dir)])

prefix = 'NCSU-CUB_Foram_Images_'
class_id = ['Globigerina_bulloides',
            'Globigerinoides_ruber', 'Globigerinoides_sacculifer',
            'Neogloboquadrina_dutertrei', 'Neogloboquadrina_incompta', 'Neogloboquadrina_pachyderma',
            'Others']

img_shape = (224, 224)

df_dict = {}
df_dict['Category'] = []
df_dict['Feature_1'] = []
df_dict['Feature_2'] = []
df_dict['Feature_3'] = []
df_dict['Feature_4'] = []
df_dict['Data'] = []
count = 0
for dirs, subdirs, files in os.walk(data_dir):
    # if count > 3:
    #     continue
    # count += 1

    print('In folder = ', dirs, ' , found file = ', files)

    if len(files) == 0:
        continue
    # get class label of files
    this_id = -1
    for num, id in enumerate(class_id):
        if id in dirs:
            print('Found class = ', id)
            this_id = num

    group_images = np.zeros(img_shape + (len(files),))
    # print ('shape of group_images ', group_images.shape)

    for i, img_file in enumerate(files):
        forams_labels.append(this_id)
        if not img_file.endswith('.png'):
            print('Not an Image file')
            continue
        else:
            pass

        img_file = os.path.join(dirs, img_file)
        img = cv2.imread(img_file, 0)
        img = cv2.resize(img, img_shape, interpolation=cv2.INTER_CUBIC)
        group_images[:, :, i] = img

        img90 = np.expand_dims(np.percentile(group_images, 90, axis=-1), axis=-1)
        img50 = np.expand_dims(np.percentile(group_images, 50, axis=-1), axis=-1)
        img10 = np.expand_dims(np.percentile(group_images, 10, axis=-1), axis=-1)
        img = np.concatenate((img10, img50, img90), axis=-1)
        img = np.expand_dims(img, axis=0)
        fea_vgg16 = vgg16_model.predict_on_batch(vgg16.preprocess_input(img))
        fea = fea_vgg16
        forams_features.append(fea)
        df_dict['Feature_1'].append(fea[0, 0])
        df_dict['Feature_2'].append(fea[0, 1])

        # placeholder
        df_dict['Feature_3'].append(fea[0, 0])
        df_dict['Feature_4'].append(fea[0, 1])
        df_dict['Category'].append(class_id[this_id])

        # print('Number of Features = ', fea.shape[1], ' for image = ', img_file)
        s = bson.binary.Binary(open(img_file, 'rb').read())
        df_dict['Data'].append(s)

print('**********************load into df success')
forams_features = np.array(forams_features).reshape(len(forams_features), -1)
forams_labels = np.array(forams_labels)
print(forams_features.shape)
print(forams_labels.shape)

permuted = np.random.permutation(len(forams_features))
forams_features = forams_features[permuted, :]
forams_labels = forams_labels[permuted]

df = pd.DataFrame(df_dict)
df = df.reindex(permuted)

train, test = forams_features[:int(len(forams_features) * 0.6)], forams_features[int(len(forams_features) * .6):]
reducer = umap.UMAP(n_neighbors=5, random_state=412, verbose=1).fit(train)
with open('reducer.pkl', 'wb') as f:
    pickle.dump(reducer, f, pickle.HIGHEST_PROTOCOL)

reducer = []
with open('reducer.pkl', 'rb') as f:
    reducer = pickle.load(f)

embedding = reducer.transform(forams_features)

df['Feature_3'] = embedding[:, 0]
df['Feature_4'] = embedding[:, 1]

df = df.reset_index(drop=True)
# print('DataFrame ', df)

conn = MongoClient('10.192.30.96', 27022)
db = conn.images
col = db.ForaminiferImage
print('**********************start insert')
for row in range(len(df)):
    col.insert({'Category': df.loc[row, 'Category'], 'Feature_1': float(df.loc[row, 'Feature_1']),
                'Feature_2': float(df.loc[row, 'Feature_2']), 'Feature_3': float(df.loc[row, 'Feature_3']),
                'Feature_4': float(df.loc[row, 'Feature_4']), 'Data': df.loc[row, 'Data']})
    if row % 10 == 0:
        print('**********************row: ', row)
print('**********************finish insert')
