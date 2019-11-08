from pymongo import MongoClient
import bson.binary

conn = MongoClient('10.192.30.96', 27022)
db = conn.images
myset = db.img

with open('test.png', 'rb') as f:
    content = bson.binary.Binary(f.read())
    dic1 = {'filename': f.name, 'data': content, 'title': 'title'}
    dic2 = {'filename': f.name, 'data': content, 'title': 'title'}
    list = []
    list.append(dic1)
    list.append(dic2)
    myset.insert_many(list)

# data = myset.find_one({'filename': 'test.png'})
# with open(data['filename'], 'wb') as f:
#     f.write(data['data'])
