from keras.applications import resnet50
from keras.applications import vgg16
import os
import numpy as np
from natsort import natsorted
import cv2
import pickle

import umap
import pandas as pd

import base64

from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy

cluster = Cluster(contact_points=['10.192.27.232'],
                  port=9042,
                  load_balancing_policy=RoundRobinPolicy())
session = cluster.connect(keyspace='bbac')

forams_features = []
forams_labels = []
label2class = {}
class_count = {}

folder = '/home/q/q/CNOOC/foraminifer'
vgg16_model = vgg16.VGG16(include_top=False, pooling='avg')
# resnet50_model = resnet50.ResNet50(include_top=False, pooling='avg')
print('Pre-trained Model loaded.')

data_dir = './NCSU-CUB_Foram_Images_01/'
print('Folders  ', [folder for folder in os.listdir(data_dir)])

prefix = 'NCSU-CUB_Foram_Images_'
class_id = ['Globigerina_bulloides',
            'Globigerinoides_ruber', 'Globigerinoides_sacculifer',
            'Neogloboquadrina_dutertrei', 'Neogloboquadrina_incompta', 'Neogloboquadrina_pachyderma',
            'Others']

img_shape = (224, 224)
# mm = '/home/q/q/CNOOC/foraminifer/NCSU-CUB_Foram_Images_01/NCSU-CUB_Foram_Images_Neogloboquadrina_pachyderma/2-3-17_Trial_2 N. Pachy from AJ 250-320 micro/imgray_4.png'
#
# img = cv2.imread(mm, 0)
# print ('MM ',img)

df_dict = {}
df_dict['Class'] = []
df_dict['Feature_1'] = []
df_dict['Feature_2'] = []
df_dict['Feature_3'] = []
df_dict['Feature_4'] = []
df_dict['Base64'] = []
count = 0
for dirs, subdirs, files in os.walk(data_dir):
    if count > 3:
        continue
    count += 1

    print('')
    print('')

    print('In folder = ', dirs, ' , found file = ', files)

    if len(files) == 0: continue
    this_id = -1  # get class label of files
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
            ##print ('Image = ',img_file)

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
        df_dict['Class'].append(class_id[this_id])

        # print('Number of Features = ', fea.shape[1], ' for image = ', img_file)
        base64_data = base64.b64encode(open(img_file, 'rb').read())
        s = base64_data.decode()
        df_dict['Base64'].append(s)
        # print('data:image/jpeg;base64,', s)

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

# print (df['Feature_3'])
df['Feature_3'] = embedding[:, 0]
df['Feature_4'] = embedding[:, 1]

# print ('---- ')
# print (df['Feature_3'])

df = df.reset_index(drop=True)
# print('DataFrame ', df)

print('**********************start insert')
for row in range(len(df)):
    # print(df.loc[row, 'Class'])
    insertSql = 'insert into foraminifer_image(class,feature_1,feature_2,feature_3,feature_4,base64) values (%s,%s,' \
                '%s,%s,%s,%s) '
    session.execute(insertSql,
                    (df.loc[row, 'Class'], df.loc[row, 'Feature_1'], df.loc[row, 'Feature_2'], df.loc[row, 'Feature_3'],
                     df.loc[row, 'Feature_4'], df.loc[row, 'Base64']))
    if row % 10 == 0:
        print('**********************row: ', row)

print('**********************finish insert')
'''

    #print (dir)
    #print ('label ',label)
    #class_id = list(set(class_id.append(label)))




    # label = dir.replace(prefix,'',1)
    # print (dir)
    # print ('label ',label)
    # class_id = list(set(class_id.append(label)))
    # print ('class - id ',class_id )


# build the pretrained model
vgg16_model = vgg16.VGG16(include_top=False, pooling='avg')
resnet50_model = resnet50.ResNet50(include_top=False, pooling='avg')
print('Pre-trained Model loaded.')


forams_features = []
forams_labels = []
label2class = {}
class_count = {}
class_list = natsorted([os.path.join(data_dir, folder) for folder in \
                            os.listdir(data_dir) if not folder.endswith('.txt')])




# for folder in os.listdir(data_dir):
#     print ('In folder = ', folder)
#     label = folder.replace(prefix, "", 1)




img_shape = (224, 224)

# build the pretrained model
vgg16_model = vgg16.VGG16(include_top=False, pooling='avg')
resnet50_model = resnet50.ResNet50(include_top=False, pooling='avg')
print('Pre-trained Model loaded.')

forams_features = []
forams_labels = []
label2class = {}
class_count = {}
class_list = natsorted([os.path.join(data_dir, folder) for folder in \
                            os.listdir(data_dir) if not folder.endswith('.txt')])




for class_id, class_folder in enumerate(class_list):
    sample_list = natsorted([os.path.join(class_folder, folder) \
                                    for folder in os.listdir(class_folder)])
    class_count[class_id] = len(sample_list) / 1000
    for sample_folder in sample_list:
        img_filenames = natsorted([os.path.join(sample_folder, file) for file in \
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
print(forams_features.shape)
print(forams_labels.shape)

with open('./forams_features.p', 'wb') as f:
    pickle.dump({'features':forams_features, 'labels':forams_labels, \
                    'label2class':label2class, 'class_count':class_count}, f)
'''
