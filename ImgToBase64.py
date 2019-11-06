import base64

with open("C:\\Users\\z003zdbc\\Desktop\\New folder\\test.png", 'rb') as f:
    base64_data = base64.b64encode(f.read())
    s = base64_data.decode()
    print('data:image/jpeg;base64,', s)
