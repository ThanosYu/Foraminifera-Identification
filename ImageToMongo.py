from pymongo import MongoClient
import bson.binary
from bson.objectid import ObjectId

conn = MongoClient('10.192.30.96', 27022)
db = conn.images
myset = db.img

# with open('test1.png', 'rb') as f:
#     content = bson.binary.Binary(f.read())
    # print(content)
#     dic1 = {'filename': f.name, 'data': content, 'title': 'title'}
#     dic2 = {'filename': f.name, 'data': content, 'title': 'title'}
#     list = []
#     list.append(dic1)
#     list.append(dic2)
#     myset.insert_many(list)

col = db.stoneImage
data = col.find_one({'_id': ObjectId("5dc5158c05f592329774374c")})
print(data['Data'])
with open('test1.jpg', 'wb') as f:
    f.write(data['Data'])
