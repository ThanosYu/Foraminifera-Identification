from pymongo import MongoClient
import bson.binary
from bson.objectid import ObjectId
from gridfs import *
import requests

conn = MongoClient('10.192.30.96', 27022)
db = conn.images

# *****write into mongo with bjson********
# myset = db.img
# with open('test1.png', 'rb') as f:
#     content = bson.binary.Binary(f.read())
# print(content)
#     dic1 = {'filename': f.name, 'data': content, 'title': 'title'}
#     dic2 = {'filename': f.name, 'data': content, 'title': 'title'}
#     list = []
#     list.append(dic1)
#     list.append(dic2)
#     myset.insert_many(list)

# read from mongo
# col = db.ForaminiferImage
# data = col.find_one({'_id': ObjectId("5dc51ec805f59246e632c009")})
# print(data['Data'])
# with open('test1.png', 'wb') as f:
#     f.write(data['Data'])

# *****write into mongo with gridfs********
fs = GridFS(db, collection="gridfsimages")
# dic = {}
# dic['filename'] = 'testfile'
# dic['tile'] = 'testfitle'
# content = open('test1.png', 'rb').read()
# fs.put(content, **dic)

# for cursor in fs.find():
#         filename = cursor.filename
#         content = cursor.read()
#         print(content)
#         with open('gridFsImage.png', 'wb') as f:
#             f.write(content)

content = fs.get(file_id=ObjectId('5dca067fc9cf2f7560d509fd')).read()
print(content)